"""Independent PPO (IPPO) for the token negotiation environment.

Each agent owns an actor-critic network and a PPO optimizer, and never sees the
other agent's parameters, observations or rewards: from its point of view the
opponent is just part of the environment.

Transitions: a seat records a transition only on steps where its action is used
(my_turn), since the env ignores the rest. A transition runs from one of the
seat's decisions to its next, and any reward that arrives in between (such as
the opponent accepting its offer) is credited to it.

Reward: the agent's own payout, discount**delay * valuation * tokens, exactly as
the env emits it. The round discount is already inside the reward, so PPO uses
gamma = 1 to avoid discounting delay twice. Rewards are divided by pool_size for
optimization only; everything logged is in raw payoff units.

Action distribution, matching the env's Dict action space:
    response  Categorical over {REJECT, ACCEPT}
    offer     Beta over the share of the pool the agent asks for, sent to the
              env as (share, 1 - share)
    message   Categorical over the cheap-talk vocabulary (with --message-vocab)
Only the parts of an action the env actually uses, per its can_accept,
can_offer and can_talk flags, enter the log-probability and entropy. That keeps
the learner protocol-agnostic: under alternating offers the response is moot in
round 0 and the offer is moot on acceptance; under sealed bids only the offer
counts; on talk steps only the message does.

Protocols: train under one with --protocol. After training, the same agents are
evaluated under every protocol, which is how to compare emergent behavior across
protocols; --checkpoint reruns that comparison for an already-trained run.

Policies: --policy mlp (the default) judges every offer on the current
observation alone; --policy gru carries a hidden state along the agent's own
decisions, so it can condition on the whole negotiation so far. Both are in
networks.py. A recurrent policy trains on whole episodes rather than shuffled
steps, which the PPO update handles by replaying each episode through the GRU.

Around this file: tournament.py plays a checkpoint against scripted negotiators
(baselines.py), and experiments.py runs several seeds per arm and compares them.

Cheap talk: with --message-vocab N, a talk step precedes every bargaining move
and the network also sees the one-hot messages last heard and sent. metrics.csv
then tracks, per agent, message entropy and how much its messages reveal about
its valuation and its next offer (see message_metrics); transcripts.py prints
episodes and checks whether listeners rely on the messages.

The rollout collector takes a matchmaker that decides who plays each seat, so
self_play.py reuses everything here with an opponent pool instead of a fixed pair.

Logs, written to runs/<run>/:
    episodes.csv  one row per episode: payout and tokens per agent, outcome, rounds
    metrics.csv   one row per update: mean episode reward per agent, agreement /
                  disagreement / timeout rates, mean rounds-to-agreement (the
                  round the deal was reached in; 1 = the first offer or bid),
                  mean Gini of agreed splits, PPO losses, entropy, approximate
                  KL, and the message metrics when cheap talk is on
    agents.pt     both agents' network weights and the config
    eval.json     evaluation under every protocol, with sampled and greedy actions

Usage:
    python ippo.py
    python ippo.py --protocol sealed_bid --total-steps 300000
    python ippo.py --message-vocab 5
    python ippo.py --checkpoint runs/<run>/agents.pt
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from negotiation_env import ACCEPT, PROTOCOLS, REJECT, NegotiationEnv
from networks import build_network

AGENTS = ("agent_0", "agent_1")
OUTCOMES = ("agreement", "disagreement", "timeout")
FLAGS = ("can_accept", "can_offer", "can_talk")
SHARE_EPS = 1e-4  # keeps sampled shares off {0, 1}, where Beta log-probs blow up


@dataclass
class Config:
    protocol: str = "alternating_offers"  # protocol to train under; evaluation covers all of them
    policy: str = "mlp"  # mlp sees only the current observation; gru remembers the negotiation so far
    total_steps: int = 1_000_000  # env steps, summed over all envs
    steps_per_update: int = 4096  # env steps per rollout; in-flight episodes then run to completion
    num_envs: int = 32
    hidden_size: int = 64
    lr: float = 3e-4
    gamma: float = 1.0  # the env's round discount is already in the reward
    gae_lambda: float = 0.95
    update_epochs: int = 10
    num_minibatches: int = 8
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    ent_coef: float = 0.02  # entropy bonus at the start of training...
    ent_coef_final: float = 0.0  # ...annealed linearly to this...
    ent_anneal_frac: float = 0.5  # ...over this fraction of training
    max_rounds: int = 20
    discount: float = 0.95
    message_vocab: int = 0  # cheap-talk vocabulary size; 0 = no channel
    observe_round: int = 1  # 0 hides the round counter, so only a policy with memory knows the deadline
    eval_episodes: int = 1000
    seed: int = 0
    device: str = "cpu"
    run_dir: str = ""  # default: runs/<script>_<timestamp>
    checkpoint: str = ""  # skip training and compare this agents.pt across protocols


def feature_count(message_vocab):
    """Network input size: 7 bargaining features, plus the heard and sent one-hot messages and can_talk."""
    return 7 + (2 * message_vocab + 1 if message_vocab else 0)


def policy_inputs(observations, env):
    """Batch one seat's Dict observations into network features and the env's can_* action flags."""
    rows = []
    for obs in observations:
        row = [
            obs["balance"][0] / env.pool_size,
            obs["valuation"][0],
            obs["last_offer"][0] / env.pool_size,
            obs["last_offer"][1] / env.pool_size,
            obs["round"] / env.max_rounds,
            obs["rounds_remaining"] / env.max_rounds,
            obs["my_turn"],
        ]
        if env.message_vocab:
            row += [*obs["heard_message"], *obs["sent_message"], obs["can_talk"]]
        rows.append(row)
    flags = {key: np.array([bool(obs.get(key, 0)) for obs in observations]) for key in FLAGS}
    return np.array(rows, dtype=np.float32), flags


def env_action(out, j, message_vocab):
    """The env action for row j of a Policy.act result."""
    share = out["share"][j]
    action = {"response": int(out["response"][j]), "offer": np.array([share, 1.0 - share], dtype=np.float32)}
    if message_vocab:
        action["message"] = int(out["message"][j])
    return action


# The networks live in networks.py: ActorCritic (feed-forward) and RecurrentActorCritic (a GRU over the
# negotiation so far). build_network picks between them from --policy.


def masked_logp_entropy(dists, actions, flags):
    """Log-prob, and entropy per part, of only the action parts the env uses per its can_* flags."""
    available = {"response": flags["can_accept"], "share": flags["can_offer"], "message": flags["can_talk"]}
    used = dict(available, share=flags["can_offer"] & ((actions["response"] == REJECT) | ~flags["can_accept"]))
    logp = sum(torch.where(used[part], dist.log_prob(actions[part]), 0.0) for part, dist in dists.items())
    entropies = {part: torch.where(available[part], dist.entropy(), 0.0) for part, dist in dists.items()}
    return logp, entropies


class Policy:
    """An actor-critic network that can act in the env. A plain Policy only plays; a PPOAgent also learns."""

    def __init__(self, net, device):
        self.net, self.device = net, device

    @torch.no_grad()
    def act(self, features, flags, greedy=False, capture=False, state=None):
        """Choose actions for a batch of observations.

        Returns numpy arrays keyed response, share, message, logp, value and message_entropy (nats; zeros
        without a cheap-talk channel), plus "state": the recurrent hidden state to pass back in on this
        stream's next decision (None for the feed-forward policy). Greedy means the most likely response
        and message and the mean share. With capture, the result also carries what the policy was thinking,
        for the demo dashboard: accept_prob, beta (the offer distribution's concentrations), message_probs
        and activations.
        """
        obs = torch.as_tensor(features, device=self.device)
        flags = {key: torch.as_tensor(value, device=self.device) for key, value in flags.items()}
        dists, value, next_state, activations = self.net.step(obs, state, capture=capture)
        if greedy:
            actions = {"response": dists["response"].probs.argmax(-1), "share": dists["share"].mean}
        else:
            actions = {"response": dists["response"].sample(), "share": dists["share"].sample()}
        actions["share"] = actions["share"].clamp(SHARE_EPS, 1 - SHARE_EPS)
        if "message" in dists:
            message = dists["message"]
            actions["message"] = message.probs.argmax(-1) if greedy else message.sample()
            message_entropy = message.entropy()
        else:
            actions["message"] = torch.zeros_like(actions["response"])
            message_entropy = torch.zeros_like(actions["share"])
        logp, _ = masked_logp_entropy(dists, actions, flags)
        out = {key: sampled.cpu().numpy() for key, sampled in actions.items()}
        out["logp"] = logp.cpu().numpy()
        out["value"] = value.cpu().numpy()
        out["message_entropy"] = message_entropy.cpu().numpy()
        out["state"] = next_state
        if capture:
            out["accept_prob"] = dists["response"].probs[..., ACCEPT].cpu().numpy()
            out["beta"] = torch.stack([dists["share"].concentration1, dists["share"].concentration0], -1).cpu().numpy()
            if "message" in dists:
                out["message_probs"] = dists["message"].probs.cpu().numpy()
            out["activations"] = {"input": features, **{key: a.cpu().numpy() for key, a in activations.items()}}
        return out


class PPOAgent(Policy):
    """A learning agent: its own actor-critic network, optimizer and PPO update. Shares nothing with others."""

    def __init__(self, name, cfg, device):
        net = build_network(feature_count(cfg.message_vocab), cfg.hidden_size, cfg.message_vocab, cfg.policy)
        super().__init__(net.to(device), device)
        self.name, self.cfg = name, cfg
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)

    def snapshot(self):
        """A frozen copy of the current policy, which plays but never learns."""
        return Policy(copy.deepcopy(self.net), self.device)

    def update(self, batch, ent_coef):
        """One PPO update. A memoryless policy learns from shuffled steps, a recurrent one from whole episodes."""
        lengths = np.asarray(batch.get("episode_lengths", []), dtype=np.int64)
        data = {
            key: torch.as_tensor(np.asarray(values), device=self.device)
            for key, values in batch.items()
            if key != "episode_lengths"
        }
        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": []}
        if self.net.recurrent:
            self._update_episodes(data, lengths, ent_coef, stats)
        else:
            self._update_steps(data, ent_coef, stats)
        return {key: float(np.mean(values)) if values else float("nan") for key, values in stats.items()}

    def _update_steps(self, data, ent_coef, stats):
        cfg = self.cfg
        n = len(data["logp"])
        minibatch_size = max(2, n // cfg.num_minibatches)
        for _ in range(cfg.update_epochs):
            order = torch.randperm(n, device=self.device)
            for start in range(0, n - minibatch_size + 1, minibatch_size):
                mb = order[start : start + minibatch_size]
                dists = self.net.policy(data["obs"][mb])
                actions = {part: data[part][mb] for part in ("response", "share", "message")}
                logp, entropies = masked_logp_entropy(dists, actions, {key: data[key][mb] for key in FLAGS})
                value = self.net.value(data["obs"][mb])
                self._apply_losses(
                    logp, data["logp"][mb], value, data["return"][mb], data["advantage"][mb],
                    sum(entropies.values()), ent_coef, stats,
                )

    def _update_episodes(self, data, lengths, ent_coef, stats):
        """PPO over whole episodes: the GRU replays each one from its zero state, so what it learns from is
        the memory it actually had. Episodes are padded to the longest in the minibatch and padding is masked."""
        cfg = self.cfg
        if not len(lengths) or not lengths.max():
            return
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        longest, episodes = int(lengths.max()), len(lengths)
        index = np.zeros((longest, episodes), dtype=np.int64)
        valid = np.zeros((longest, episodes), dtype=bool)
        for column, (start, length) in enumerate(zip(starts, lengths)):
            index[:length, column] = np.arange(start, start + length)
            valid[:length, column] = True
        index = torch.as_tensor(index, device=self.device)
        valid = torch.as_tensor(valid, device=self.device)
        per_minibatch = max(1, episodes // cfg.num_minibatches)

        for _ in range(cfg.update_epochs):
            order = torch.randperm(episodes, device=self.device)
            for start in range(0, episodes - per_minibatch + 1, per_minibatch):
                columns = order[start : start + per_minibatch]
                rows = index[:, columns].reshape(-1)
                keep = valid[:, columns].reshape(-1)
                dists, value = self.net.sequence(data["obs"][index[:, columns]])
                actions = {part: data[part][rows] for part in ("response", "share", "message")}
                logp, entropies = masked_logp_entropy(dists, actions, {key: data[key][rows] for key in FLAGS})
                self._apply_losses(
                    logp[keep], data["logp"][rows][keep], value[keep], data["return"][rows][keep],
                    data["advantage"][rows][keep], sum(entropies.values())[keep], ent_coef, stats,
                )

    def _apply_losses(self, logp, old_logp, value, returns, advantage, entropy, ent_coef, stats):
        cfg = self.cfg
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        log_ratio = logp - old_logp
        ratio = log_ratio.exp()
        clipped_ratio = ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)
        policy_loss = torch.max(-advantage * ratio, -advantage * clipped_ratio).mean()
        value_loss = 0.5 * (value - returns).pow(2).mean()
        loss = policy_loss - ent_coef * entropy.mean() + cfg.vf_coef * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
        self.optimizer.step()

        stats["policy_loss"].append(policy_loss.item())
        stats["value_loss"].append(value_loss.item())
        stats["entropy"].append(entropy.mean().item())
        stats["approx_kl"].append(((ratio - 1) - log_ratio).mean().item())


@dataclass
class Match:
    """Who controls each seat for one episode, plus tags copied into that episode's record."""

    seats: tuple[Policy, Policy]
    tags: dict = field(default_factory=dict)


def is_recurrent(policy):
    """Whether a policy carries hidden state between its own decisions. Scripted opponents have no network."""
    return getattr(getattr(policy, "net", None), "recurrent", False)


def mean(values):
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def gini(tokens):
    """Gini coefficient of a token split: 0 = equal shares, (n-1)/n = one agent holds everything (0.5 for two)."""
    x = np.asarray(tokens, dtype=np.float64)
    total = x.sum()
    return float(np.abs(x[:, None] - x[None, :]).sum() / (2 * len(x) * total)) if total > 0 else 0.0


def episode_record(payouts, infos):
    """Summary of one finished episode; payouts and tokens are per seat."""
    return {
        "payouts": list(payouts),
        "tokens": [infos[name]["tokens"] for name in AGENTS],
        "outcome": infos[AGENTS[0]]["outcome"],
        "rounds": infos[AGENTS[0]]["round"],
    }


def decision_record(obs, flags, out, j, seat, policy):
    """What one seat decided on one step, kept per episode for message analysis and transcripts."""
    can_accept, can_offer, can_talk = (bool(flags[key][j]) for key in FLAGS)
    if can_talk:
        kind = "talk"
    elif can_accept and out["response"][j] == ACCEPT:
        kind = "accept"
    elif can_offer:
        kind = "offer"
    else:
        kind = "reject"
    return {
        "seat": seat,
        "learner": getattr(policy, "name", None),  # None for frozen snapshots
        "round": int(obs["round"]),
        "kind": kind,
        "can_accept": can_accept,
        "valuation": float(obs["valuation"][0]),
        "message": int(out["message"][j]) if can_talk else None,
        "message_entropy": float(out["message_entropy"][j]) if can_talk else None,
        "heard": message_token(obs.get("heard_message")),  # the opponent's latest message
        "sent": message_token(obs.get("sent_message")),  # this seat's own latest message
        "generosity": 1.0 - float(out["share"][j]) if kind == "offer" else None,  # share offered to the opponent
    }


def message_token(one_hot):
    return None if one_hot is None or not one_hot.any() else int(one_hot.argmax())


def summarize(episodes):
    """Mean payout per seat, outcome rates, mean rounds-to-agreement and mean Gini of the agreed splits."""
    agreed = [ep for ep in episodes if ep["outcome"] == "agreement"]
    summary = {f"reward_{name}": mean(ep["payouts"][seat] for ep in episodes) for seat, name in enumerate(AGENTS)}
    summary.update({f"{outcome}_rate": mean(ep["outcome"] == outcome for ep in episodes) for outcome in OUTCOMES})
    summary["rounds_to_agreement"] = mean(ep["rounds"] for ep in agreed)
    summary["gini"] = mean(gini(ep["tokens"]) for ep in agreed)
    return summary


def format_summary(summary):
    return (
        f"reward {summary['reward_agent_0']:6.1f} / {summary['reward_agent_1']:6.1f}"
        f" | agreement {summary['agreement_rate']:6.1%} | rounds to agreement {summary['rounds_to_agreement']:5.2f}"
        f" | gini {summary['gini']:.3f}"
    )


def entropy_bits(tokens, vocab):
    """Entropy, in bits, of the empirical distribution of tokens from range(vocab); NaN if there are none."""
    if not tokens:
        return float("nan")
    p = np.bincount(tokens, minlength=vocab) / len(tokens)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def eta_squared(groups, values, strata):
    """Share of the variance in values explained by group membership, after removing each stratum's mean.

    0 = knowing the group (e.g. the message sent) says nothing about the value; 1 = it determines the value.
    NaN when there is no variance left to explain.
    """
    groups, strata = np.asarray(groups), np.asarray(strata)
    values = np.array(values, dtype=np.float64)
    if len(values) < 2:
        return float("nan")
    for stratum in np.unique(strata):
        values[strata == stratum] -= values[strata == stratum].mean()
    total = (values**2).sum()  # the residuals have mean 0
    if total <= 1e-12:
        return float("nan")
    between = sum((groups == g).sum() * values[groups == g].mean() ** 2 for g in np.unique(groups))
    return float(between / total)


def message_metrics(episodes, vocab):
    """How each learner uses the cheap-talk channel, from the decisions recorded in episodes.

    message_entropy_<agent>         entropy of the messages it sent, in bits: log2(vocab) = every token
                                    equally often, 0 = always the same token
    message_policy_entropy_<agent>  mean entropy of its message policy at each talk step, in bits. High
                                    message entropy with low policy entropy means the token it picks depends
                                    on the situation rather than on chance.
    message_valuation_eta2_<agent>  share of the variance in its valuation explained by the message it sent
    message_offer_eta2_<agent>      share of the variance in the generosity (share of the pool offered to the
                                    opponent) of the offer it made right after talking, explained by its message
    Both eta2 are computed within rounds, so a message that merely tracks the round number isn't a signal.
    """
    metrics = {}
    for name in AGENTS:
        decisions = [d for ep in episodes for d in ep["decisions"] if d["learner"] == name]
        talk = [d for d in decisions if d["kind"] == "talk"]
        offers = [d for d in decisions if d["kind"] == "offer" and d["sent"] is not None]
        metrics[f"message_entropy_{name}"] = entropy_bits([d["message"] for d in talk], vocab)
        metrics[f"message_policy_entropy_{name}"] = mean(d["message_entropy"] for d in talk) / np.log(2)
        metrics[f"message_valuation_eta2_{name}"] = eta_squared(
            [d["message"] for d in talk], [d["valuation"] for d in talk], [d["round"] for d in talk]
        )
        metrics[f"message_offer_eta2_{name}"] = eta_squared(
            [d["sent"] for d in offers], [d["generosity"] for d in offers], [d["round"] for d in offers]
        )
    return metrics


def format_messages(metrics):
    return (
        f"message entropy {metrics['message_entropy_agent_0']:.2f} / {metrics['message_entropy_agent_1']:.2f} bits"
        f" | offer eta2 {metrics['message_offer_eta2_agent_0']:.3f} / {metrics['message_offer_eta2_agent_1']:.3f}"
    )


class CsvLog:
    """Writes dict rows to a CSV file, taking the header from the first row and flushing after every write."""

    def __init__(self, path):
        self.file = open(path, "w", newline="")
        self.writer = None

    def write(self, rows):
        for row in rows:
            if self.writer is None:
                self.writer = csv.DictWriter(self.file, fieldnames=list(row))
                self.writer.writeheader()
            self.writer.writerow(row)
        self.file.flush()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.file.close()


class RolloutCollector:
    """Steps the envs and turns finished episodes into per-learner PPO batches.

    Before every episode, ``matchmaker(env_index)`` returns the Match for that env. A seat controlled by a
    PPOAgent turns its decisions into training data for that agent (both seats may be the same agent, as in
    self-play); a seat controlled by any other Policy just plays. Every decision is also recorded in the
    episode's record, for message analysis.
    """

    SAMPLE_KEYS = ("obs", "response", "share", "message", *FLAGS, "logp")
    TRAJECTORY_KEYS = (*SAMPLE_KEYS, "value", "reward")
    BATCH_KEYS = (*SAMPLE_KEYS, "advantage", "return", "episode_lengths")

    def __init__(self, envs, matchmaker, cfg):
        self.envs, self.matchmaker, self.cfg = envs, matchmaker, cfg
        self.obs = [env.reset(seed=cfg.seed + i)[0] for i, env in enumerate(envs)]
        self.matches, self.trajectories = [None] * len(envs), [None] * len(envs)
        self.payouts, self.decisions = [None] * len(envs), [None] * len(envs)
        self.states = [None] * len(envs)  # per seat: a recurrent policy's memory of this episode so far

    def _start_episode(self, e):
        self.matches[e] = self.matchmaker(e)
        self.trajectories[e] = [{key: [] for key in self.TRAJECTORY_KEYS} for _ in AGENTS]  # one per seat
        self.payouts[e] = [0.0, 0.0]
        self.decisions[e] = []
        self.states[e] = [None, None]  # every episode starts with a blank memory

    def collect(self, min_steps):
        """Step the envs for at least min_steps env steps, then let in-flight episodes finish.

        Every env is at the start of an episode when this is called and again when it returns, so each
        episode is played entirely within one rollout and GAE never has to bootstrap across its boundary.
        Returns (batch per learner name, one record per finished episode, env steps taken).
        """
        for e in range(len(self.envs)):
            self._start_episode(e)
        batches = defaultdict(lambda: {key: [] for key in self.BATCH_KEYS})
        episodes, steps = [], 0
        running = list(range(len(self.envs)))
        while running:
            actions = self._act(running)
            still_running = []
            for e in running:
                env = self.envs[e]
                obs, rewards, terminations, truncations, infos = env.step(actions[e])
                steps += 1
                for seat, name in enumerate(AGENTS):
                    self.payouts[e][seat] += rewards[name]
                    trajectory = self.trajectories[e][seat]
                    # Credit the reward to the seat's latest decision, even when the opponent's move triggered it.
                    if trajectory["reward"]:
                        trajectory["reward"][-1] += rewards[name] / env.pool_size

                done = any(terminations.values()) or any(truncations.values())
                if done:
                    # Hard timeouts (off by default) are treated as terminal rather than bootstrapped.
                    match = self.matches[e]
                    for seat, policy in enumerate(match.seats):
                        if isinstance(policy, PPOAgent):
                            self._finish(self.trajectories[e][seat], batches[policy.name])
                    record = episode_record(self.payouts[e], infos)
                    episodes.append({**record, "match": match, "decisions": self.decisions[e]})
                    obs, _ = env.reset()
                self.obs[e] = obs
                if not done:
                    still_running.append(e)
                elif steps < min_steps:
                    self._start_episode(e)
                    still_running.append(e)
            running = still_running
        return batches, episodes, steps

    def _act(self, running):
        """Query each policy once, batched over every (env, seat) it controls on this step."""
        actions = {e: {} for e in running}
        groups = {}  # id(policy) -> (policy, [(env, seat), ...])
        for seat, name in enumerate(AGENTS):
            for e in running:
                if self.obs[e][name]["my_turn"]:
                    policy = self.matches[e].seats[seat]
                    groups.setdefault(id(policy), (policy, []))[1].append((e, seat))

        for policy, pairs in groups.values():
            observations = [self.obs[e][AGENTS[seat]] for e, seat in pairs]
            features, flags = policy_inputs(observations, self.envs[0])
            out = policy.act(features, flags, state=self._gather_states(policy, pairs))
            self._scatter_states(policy, pairs, out["state"])
            learning = isinstance(policy, PPOAgent)
            for j, (e, seat) in enumerate(pairs):
                if learning:
                    sample = {"obs": features[j], "reward": 0.0, **{key: flags[key][j] for key in FLAGS}}
                    sample.update({key: out[key][j] for key in ("response", "share", "message", "logp", "value")})
                    for key in self.TRAJECTORY_KEYS:
                        self.trajectories[e][seat][key].append(sample[key])
                self.decisions[e].append(decision_record(observations[j], flags, out, j, seat, policy))
                actions[e][AGENTS[seat]] = env_action(out, j, self.envs[0].message_vocab)
        return actions

    def _gather_states(self, policy, pairs):
        """These streams' recurrent states as one batch, blank where a stream has not acted yet."""
        if not is_recurrent(policy):
            return None
        actor, critic = [], []
        for e, seat in pairs:
            state = self.states[e][seat]
            if state is None:
                state = policy.net.initial_state(1, policy.device)
            actor.append(state[0])
            critic.append(state[1])
        return torch.cat(actor, dim=1), torch.cat(critic, dim=1)

    def _scatter_states(self, policy, pairs, state):
        if not is_recurrent(policy):
            return
        actor, critic = state
        for column, (e, seat) in enumerate(pairs):
            self.states[e][seat] = (actor[:, column : column + 1], critic[:, column : column + 1])

    def _finish(self, trajectory, batch):
        """Compute GAE over one seat's complete episode and move it into the learner's batch."""
        if not trajectory["reward"]:
            return  # the seat never got to move (e.g. a hard timeout on the first step)
        rewards = np.array(trajectory["reward"])
        values = np.array(trajectory["value"], dtype=np.float64)
        next_values = np.append(values[1:], 0.0)  # the last transition is terminal
        deltas = rewards + self.cfg.gamma * next_values - values
        advantages = np.zeros_like(deltas)
        gae = 0.0
        for t in reversed(range(len(deltas))):
            gae = deltas[t] + self.cfg.gamma * self.cfg.gae_lambda * gae
            advantages[t] = gae
        for key in self.SAMPLE_KEYS:
            batch[key].extend(trajectory[key])
        batch["advantage"].extend(advantages.astype(np.float32))
        batch["return"].extend((advantages + values).astype(np.float32))
        batch["episode_lengths"].append(len(advantages))  # a recurrent update replays these as sequences


def make_env(cfg, protocol=None, **kwargs):
    """An env configured from a run's config; anything passed here overrides the config."""
    settings = {
        "max_rounds": cfg.max_rounds,
        "discount": cfg.discount,
        "message_vocab": cfg.message_vocab,
        "observe_round": bool(cfg.observe_round),
    }
    settings.update(kwargs)
    return NegotiationEnv(protocol or cfg.protocol, **settings)


def play_episode(env, policies, seed, greedy=False, scramble=None, options=None, on_decision=None):
    """Play one episode between two policies (seat 0, seat 1) in an env with render_mode="ansi".

    Returns the episode record, including every decision, and the rendered transcript lines. With scramble,
    a numpy Generator, every message is swapped for a uniformly random token before the listener hears it;
    the decision records keep the token the speaker actually chose. options are passed to env.reset.
    on_decision(decision, out) runs after every move, with the policy's captured thinking in out; the demo
    dashboard uses it to stream a negotiation move by move.
    """
    obs, _ = env.reset(seed=seed, options=options)
    payouts, decisions, lines = [0.0, 0.0], [], [env.render()]
    states = [None, None]  # a recurrent policy's memory of this episode, per seat
    while env.agents:
        actions = {}
        for seat, name in enumerate(AGENTS):
            if obs[name]["my_turn"]:
                features, flags = policy_inputs([obs[name]], env)
                out = policies[seat].act(features, flags, greedy=greedy, capture=on_decision is not None, state=states[seat])
                states[seat] = out["state"]
                decision = decision_record(obs[name], flags, out, 0, seat, policies[seat])
                decisions.append(decision)
                if on_decision:
                    on_decision(decision, out)
                actions[name] = env_action(out, 0, env.message_vocab)
                if scramble is not None and flags["can_talk"][0]:
                    actions[name]["message"] = int(scramble.integers(env.message_vocab))
        obs, rewards, _, _, infos = env.step(actions)
        for seat, name in enumerate(AGENTS):
            payouts[seat] += rewards[name]
        lines.append("  " + env.render())
    return {**episode_record(payouts, infos), "decisions": decisions}, lines


def evaluate(policies, cfg, protocol, episodes, greedy, transcripts=0):
    """Play two policies (seat 0, seat 1) under a protocol on a fixed set of episodes.

    Actions are sampled, or greedy: the most likely response and message and the mean share. The two can
    differ sharply: a response policy that accepts 40% of the time never accepts when played greedily.
    """
    env = make_env(cfg, protocol, render_mode="ansi")
    records, lines = [], []
    for episode in range(episodes):
        # The same valuations for every run and protocol.
        record, transcript = play_episode(env, policies, seed=1_000_000 + episode, greedy=greedy)
        records.append(record)
        if episode < transcripts:
            lines += transcript
    return summarize(records), lines


def compare_protocols(policies, cfg, episodes):
    """Evaluate the same two policies under every protocol, with sampled and with greedy actions."""
    print(f"\nAgents trained under {cfg.protocol}, {episodes} evaluation episodes per row:")
    results, transcripts = {}, []
    for protocol in PROTOCOLS:
        results[protocol] = {}
        for mode in ("sampled", "greedy"):
            greedy = mode == "greedy"
            summary, lines = evaluate(policies, cfg, protocol, episodes, greedy, transcripts=0 if greedy else 1)
            results[protocol][mode] = summary
            transcripts += lines
            print(f"  {protocol:<18} {mode:<7} | {format_summary(summary)}")
    print("\nSample episodes, sampled actions:\n" + "\n".join(transcripts))
    return results


def start_run(cfg, prefix):
    """Seed torch, create the run directory and record the config in it."""
    torch.manual_seed(cfg.seed)
    run_dir = Path(cfg.run_dir or f"runs/{prefix}_{time.strftime('%Y%m%d-%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    return run_dir


def entropy_coef(cfg, global_step):
    """Entropy bonus, annealed linearly from ent_coef to ent_coef_final over the first ent_anneal_frac of training."""
    progress = min(1.0, global_step / max(1.0, cfg.ent_anneal_frac * cfg.total_steps))
    return cfg.ent_coef + progress * (cfg.ent_coef_final - cfg.ent_coef)


def save_checkpoint(path, cfg, learners):
    state = {name: learner.net.state_dict() for name, learner in learners.items()}
    torch.save({"config": asdict(cfg), "agents": state}, path)


def load_agents(path, device="cpu"):
    """The config and both agents of a run saved by ippo.py or self_play.py."""
    device = torch.device(device)
    checkpoint = torch.load(path, map_location=device)
    known = {option.name for option in fields(Config)}
    cfg = Config(**{key: value for key, value in checkpoint["config"].items() if key in known})
    agents = {name: PPOAgent(name, cfg, device) for name in AGENTS}
    for name, agent in agents.items():
        agent.net.load_state_dict(checkpoint["agents"][name])
    return cfg, agents


def train(cfg):
    run_dir = start_run(cfg, "ippo")
    device = torch.device(cfg.device)
    envs = [make_env(cfg) for _ in range(cfg.num_envs)]
    agents = {name: PPOAgent(name, cfg, device) for name in AGENTS}
    pairing = Match((agents["agent_0"], agents["agent_1"]))
    collector = RolloutCollector(envs, lambda e: pairing, cfg)

    global_step, update, start = 0, 0, time.time()
    with CsvLog(run_dir / "episodes.csv") as episode_log, CsvLog(run_dir / "metrics.csv") as metric_log:
        while global_step < cfg.total_steps:
            update += 1
            ent_coef = entropy_coef(cfg, global_step)
            batches, episodes, steps = collector.collect(cfg.steps_per_update)
            global_step += steps
            stats = {name: agent.update(batches[name], ent_coef) for name, agent in agents.items()}

            summary = summarize(episodes)
            metrics = {"update": update, "global_step": global_step, "episodes": len(episodes), **summary}
            if cfg.message_vocab:
                metrics.update(message_metrics(episodes, cfg.message_vocab))
            for name, agent_stats in stats.items():
                metrics.update({f"{key}_{name}": value for key, value in agent_stats.items()})
            metrics.update({"ent_coef": ent_coef, "sps": global_step / (time.time() - start)})

            episode_log.write(
                {
                    "update": update,
                    **{f"reward_{name}": ep["payouts"][seat] for seat, name in enumerate(AGENTS)},
                    **{f"tokens_{name}": ep["tokens"][seat] for seat, name in enumerate(AGENTS)},
                    "outcome": ep["outcome"],
                    "rounds": ep["rounds"],
                }
                for ep in episodes
            )
            metric_log.write([metrics])
            line = (
                f"update {update:4d} | step {global_step:>9,} | {format_summary(summary)}"
                f" | entropy {stats['agent_0']['entropy']:+.2f} / {stats['agent_1']['entropy']:+.2f}"
                f" | sps {metrics['sps']:,.0f}"
            )
            print(f"{line} | {format_messages(metrics)}" if cfg.message_vocab else line)

    save_checkpoint(run_dir / "agents.pt", cfg, agents)
    if cfg.eval_episodes:
        results = compare_protocols((agents["agent_0"], agents["agent_1"]), cfg, cfg.eval_episodes)
        (run_dir / "eval.json").write_text(json.dumps(results, indent=2))
    print(f"\nLogs and checkpoint in {run_dir}")


def evaluate_checkpoint(cfg):
    """Load a trained run's two agents and compare them across protocols."""
    torch.manual_seed(cfg.seed)
    trained, agents = load_agents(cfg.checkpoint, cfg.device)
    compare_protocols((agents["agent_0"], agents["agent_1"]), trained, cfg.eval_episodes)


def parse_config(config_cls, description):
    """Build a config dataclass from command-line flags, one --flag per field."""
    parser = argparse.ArgumentParser(description=description)
    for option in fields(config_cls):
        parser.add_argument(f"--{option.name.replace('_', '-')}", type=type(option.default), default=option.default)
    return config_cls(**vars(parser.parse_args()))


def main(train_fn=train, config_cls=Config, description="Independent PPO for the negotiation environment."):
    cfg = parse_config(config_cls, description)
    if torch.device(cfg.device).type == "cpu":
        torch.set_num_threads(1)  # thread overhead outweighs any speedup for 64-unit MLPs
    if cfg.checkpoint:
        evaluate_checkpoint(cfg)
    else:
        train_fn(cfg)


if __name__ == "__main__":
    main()
