"""Population-based economy: agents that must earn, trade and reproduce to survive.

N agents (8 to start) each hold a token balance and a negotiation policy, the same actor-critic network
the PPO scripts train. Here the policies don't learn by gradient; they evolve. Every round:

  1. Compute arrives. A fixed compute supply is split at random among the living agents, so each agent
     gets less as the population grows: compute is the scarce second resource.
  2. Trade. Agents are paired at random. Task output has diminishing returns in compute, so shifting
     compute from the pair member with more to the one with less raises their combined output. The pair
     negotiates how to split that gain, in the negotiation env (alternating offers by default), each
     observing its need as its valuation: high when broke, falling as its balance grows. The agreed split
     sets the token price the buyer pays for the compute. Haggling burns compute: both agents' compute
     shrinks by the discount for every round of delay, and no deal means no transfer.
  3. Work. Each agent attempts the task and earns task_reward * score * compute**elasticity tokens. The
     task is a stub (a uniformly random score) behind the Task interface, to be replaced by code review.
  4. Upkeep. Every balance loses survival_cost tokens plus decay_rate of itself.
  5. Death and birth. Agents at or below zero tokens die. Agents at or above reproduction_threshold hand
     child_share of their balance to a child whose policy is the parent's plus Gaussian noise on every
     parameter, while the population is below max_population.

Rounds are grouped into generations (rounds_per_generation). This module is the simulation only; run it
with train.py, which handles logging, checkpoints and plots.
"""

from __future__ import annotations

import copy
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render straight to files; no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator, PercentFormatter

from ippo import Policy, feature_count, gini, mean, play_episode
from livelog import NullLog
from networks import build_network
from negotiation_env import NegotiationEnv, share_of_pool
from self_play import INK_MUTED, STYLE, finish_panel, read_metrics


@dataclass
class EconomyConfig:
    population: int = 8  # founding agents
    generations: int = 300
    rounds_per_generation: int = 20
    initial_tokens: float = 10.0  # founders' balance; also the wealth scale of the need signal
    survival_cost: float = 1.0  # tokens every agent loses each round...
    # ...plus this fraction of its balance. A positive rate pulls every balance towards the same level, so few
    # agents ever reach zero or the reproduction threshold and selection stalls.
    decay_rate: float = 0.0
    compute_supply: float = 120.0  # compute split at random among the living agents each round
    compute_concentration: float = 0.5  # Dirichlet concentration of that split; lower = more unequal
    compute_elasticity: float = 0.5  # task output = score * compute ** elasticity (diminishing returns)
    task_reward: float = 1.0  # tokens per unit of task output
    trade_fraction: float = 0.5  # share of a pair's compute gap a trade shifts (0.5 evens it out)
    reproduction_threshold: float = 20.0
    child_share: float = 0.5  # fraction of the parent's balance that goes to the child
    mutation_std: float = 0.02  # std of the Gaussian noise added to every parameter of a child's policy
    max_population: int = 64
    protocol: str = "alternating_offers"
    policy: str = "mlp"  # the agents' network; taken from the checkpoint when init is given
    max_rounds: int = 20
    discount: float = 0.95  # also the share of compute that survives each round of haggling
    message_vocab: int = 0  # cheap-talk vocabulary; taken from the checkpoint when init is given
    hidden_size: int = 64  # likewise
    init: str = ""  # agents.pt whose agents seed the founders as mutated copies; default: random networks
    seed: int = 0


class Task(ABC):
    """The work agents do to earn tokens. Implement both methods to plug in a real task (e.g. code review)."""

    @abstractmethod
    def expected_score(self, agent) -> float:
        """The score the agent can expect, used to value compute when trading."""

    @abstractmethod
    def attempt(self, agent, rng) -> float:
        """The agent's score on this round's task, in [0, 1]."""


class RandomScoreTask(Task):
    """Stub task: every attempt scores uniformly at random in [0, 1], whatever the agent."""

    def expected_score(self, agent):
        return 0.5

    def attempt(self, agent, rng):
        return float(rng.random())


@dataclass
class Agent:
    id: int
    policy: Policy
    tokens: float
    lineage: int = 0  # generations of descent: founders are 0, their children 1, and so on
    parent: int | None = None
    born: int = 0  # round of birth
    compute: float = 0.0


def mutate(policy, std):
    """A copy of the policy with independent Gaussian noise added to every parameter."""
    net = copy.deepcopy(policy.net)
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.add_(torch.randn_like(parameter) * std)
    return Policy(net, policy.device)


class Economy:
    """The living population and the rules of one round."""

    def __init__(self, cfg, task, parents=None, live=None):
        """parents: trained policies to seed the founders from, as mutated copies; None for random networks.

        live: a livelog.LiveLog to stream the demo events to, or None for a run that nobody is watching.
        """
        self.cfg, self.task = cfg, task
        self.live = live or NullLog()
        self.rng = np.random.default_rng(cfg.seed)
        self.env = NegotiationEnv(
            cfg.protocol,
            max_rounds=cfg.max_rounds,
            discount=cfg.discount,
            message_vocab=cfg.message_vocab,
            render_mode="ansi",
        )
        if parents:
            policies = [mutate(parents[i % len(parents)], cfg.mutation_std) for i in range(cfg.population)]
        else:
            size = feature_count(cfg.message_vocab)
            policies = [
                Policy(build_network(size, cfg.hidden_size, cfg.message_vocab, cfg.policy), torch.device("cpu"))
                for _ in range(cfg.population)
            ]
        self.agents = [Agent(i, policy, cfg.initial_tokens) for i, policy in enumerate(policies)]
        self.next_id = len(self.agents)
        self.round = 0

    def play_generation(self):
        """Play rounds_per_generation rounds, fewer if the population dies out.

        Returns the generation's trade records and its birth and death counts.
        """
        outcome = {"trades": [], "births": 0, "deaths": 0}
        for _ in range(self.cfg.rounds_per_generation):
            if not self.agents:
                break
            events = self.play_round()
            outcome["trades"] += events["trades"]
            outcome["births"] += events["births"]
            outcome["deaths"] += events["deaths"]
        return outcome

    def play_round(self):
        """Advance one round. Returns its trade records and its birth and death counts."""
        cfg, rng = self.cfg, self.rng
        self.round += 1
        for agent, share in zip(self.agents, rng.dirichlet(np.full(len(self.agents), cfg.compute_concentration))):
            agent.compute = cfg.compute_supply * share

        order = rng.permutation(len(self.agents))
        pairs = [(self.agents[i], self.agents[j]) for i, j in zip(order[::2], order[1::2])]
        spotlight = pairs[0] if pairs else None  # the one negotiation the demo follows move by move
        self.live.emit(
            "round_start",
            round=self.round,
            agents=[{"id": a.id, "tokens": a.tokens, "compute": a.compute, "lineage": a.lineage} for a in self.agents],
            pairs=[[a.id, b.id] for a, b in pairs],
            spotlight=[spotlight[0].id, spotlight[1].id] if spotlight else None,
        )
        trades = [self.trade(a, b, spotlight=index == 0) for index, (a, b) in enumerate(pairs)]

        for agent in self.agents:
            score = self.task.attempt(agent, rng)
            agent.tokens += cfg.task_reward * score * agent.compute**cfg.compute_elasticity
            agent.tokens = agent.tokens * (1 - cfg.decay_rate) - cfg.survival_cost

        survivors = [agent for agent in self.agents if agent.tokens > 0]
        children = []
        for index in rng.permutation(len(survivors)):
            parent = survivors[index]
            if parent.tokens >= cfg.reproduction_threshold and len(survivors) + len(children) < cfg.max_population:
                endowment = parent.tokens * cfg.child_share
                parent.tokens -= endowment
                policy = mutate(parent.policy, cfg.mutation_std)
                children.append(Agent(self.next_id, policy, endowment, parent.lineage + 1, parent.id, self.round))
                self.next_id += 1
        dead = [agent.id for agent in self.agents if agent.tokens <= 0]
        self.agents = survivors + children
        self.live.emit(
            "round_end",
            round=self.round,
            balances=[{"id": agent.id, "tokens": agent.tokens} for agent in self.agents],
            trades=[{"pair": list(trade["ids"]), "outcome": trade["outcome"], "split": trade["tokens"]} for trade in trades if trade["outcome"] != "no gains"],
            births=[{"id": child.id, "parent": child.parent} for child in children],
            deaths=dead,
        )
        return {"trades": trades, "births": len(children), "deaths": len(dead)}

    def trade(self, a, b, spotlight=False):
        """a and b negotiate over the gain from shifting compute to whichever of them has less.

        With spotlight, every move is streamed to the demo log as it happens, with what the acting policy
        was thinking, and the run waits between moves so a person can follow along.

        Returns the negotiation's episode record (outcome "no gains" if there was nothing to negotiate) with
        the economy round, the pair's ids and valuations, the surplus that was split, the price paid and the
        env's rendered transcript, whose agent_0 and agent_1 are ids[0] and ids[1].
        """
        pair = (a, b)
        buyer_seat = 0 if a.compute < b.compute else 1
        buyer, seller = pair[buyer_seat], pair[1 - buyer_seat]
        valuations = [self.need(agent) for agent in pair]
        record = {
            "round": self.round,
            "ids": (a.id, b.id),
            "valuations": valuations,
            "outcome": "no gains",
            "decisions": [],
            "transcript": [],
            "surplus": 0.0,
            "price": 0.0,
        }
        if self.gains(buyer, seller)[0] <= 0:
            return record

        if spotlight:
            surplus, loss, moved = self.gains(buyer, seller)
            self.live.emit(
                "negotiation_start",
                pair=[a.id, b.id],
                needs=valuations,
                compute=[a.compute, b.compute],
                tokens=[a.tokens, b.tokens],
                buyer=buyer.id,
                surplus=surplus,
                compute_moved=moved,
            )
            self.live.beat()
        seed = int(self.rng.integers(2**31))
        episode, transcript = play_episode(
            self.env,
            (a.policy, b.policy),
            seed,
            options={"valuations": valuations},
            on_decision=self._spotlight_decisions(pair) if spotlight else None,
        )
        record.update(episode, transcript=[line.strip() for line in transcript])
        burn = self.cfg.discount ** max(record["rounds"] - 1, 0)  # every round of haggling burns compute
        for agent in pair:
            agent.compute *= burn
        if record["outcome"] != "agreement":
            return record

        surplus, loss, moved = self.gains(buyer, seller)
        seller_share = record["tokens"][1 - buyer_seat] / self.env.pool_size
        price = min(loss + seller_share * surplus, buyer.tokens)
        buyer.tokens -= price
        seller.tokens += price
        buyer.compute += moved
        seller.compute -= moved
        record.update(surplus=surplus, price=price)
        if spotlight:
            self.live.emit(
                "deal",
                pair=[a.id, b.id],
                split=record["tokens"],
                price=price,
                surplus=surplus,
                rounds=record["rounds"],
                balances=[buyer.tokens, seller.tokens],
                transcript=record["transcript"],
            )
            self.live.beat(2.0)
        return record

    def _spotlight_decisions(self, pair):
        """A callback that streams each move of the spotlight negotiation, paced for the eye."""
        ids = [agent.id for agent in pair]

        def on_decision(decision, out):
            event = {
                "agent": ids[decision["seat"]],
                "partner": ids[1 - decision["seat"]],
                "negotiation_round": decision["round"],
                "action": decision["kind"],
                "accept_prob": float(out["accept_prob"][0]),
                "beta": [float(value) for value in out["beta"][0]],
                "value": float(out["value"][0]),
                "activations": {key: values[0] for key, values in out["activations"].items()},
            }
            if decision["kind"] == "talk":
                event["message"] = decision["message"]
                event["message_probs"] = out["message_probs"][0]
            elif decision["kind"] == "offer":
                to_partner = decision["generosity"]
                partner_tokens = share_of_pool([to_partner, 1 - to_partner], self.env.pool_size)
                event["to_partner"] = to_partner
                event["tokens"] = [self.env.pool_size - partner_tokens, partner_tokens]  # keeps, offers
            self.live.emit("decision", **event)
            self.live.beat()

        return on_decision

    def gains(self, buyer, seller):
        """(surplus, seller's loss, compute moved) in expected tokens, if trade_fraction of the gap moves."""
        cfg = self.cfg
        moved = cfg.trade_fraction * (seller.compute - buyer.compute)

        def value(agent, compute):
            return cfg.task_reward * self.task.expected_score(agent) * compute**cfg.compute_elasticity

        gain = value(buyer, buyer.compute + moved) - value(buyer, buyer.compute)
        loss = value(seller, seller.compute) - value(seller, seller.compute - moved)
        return gain - loss, loss, moved

    def need(self, agent):
        """The valuation an agent observes: the top of the env's range when broke, falling as it gets richer."""
        low, high = self.env.valuation_range
        return low + (high - low) * np.exp(-max(agent.tokens, 0.0) / self.cfg.initial_tokens)


def negotiated(trades):
    """The trades where a negotiation actually took place."""
    return [trade for trade in trades if trade["outcome"] != "no gains"]


def generation_row(generation, economy, outcome):
    """Metrics for one generation, from play_generation's outcome and the population at its end.

    mean_generosity is each agent's mean offer to its partner, as a share of the negotiated surplus,
    averaged over the agents that made offers; generosity_std is the spread across those agents.
    """
    agents = economy.agents
    tokens = [agent.tokens for agent in agents]
    trades = negotiated(outcome["trades"])
    agreed = [trade for trade in trades if trade["outcome"] == "agreement"]
    offers = defaultdict(list)  # agent id -> generosity of every offer it made
    for trade in trades:
        for decision in trade["decisions"]:
            if decision["kind"] == "offer":
                offers[trade["ids"][decision["seat"]]].append(decision["generosity"])
    generosity = [np.mean(values) for values in offers.values()]
    return {
        "generation": generation,
        "round": economy.round,
        "population": len(agents),
        "births": outcome["births"],
        "deaths": outcome["deaths"],
        "mean_generosity": mean(generosity),
        "generosity_std": float(np.std(generosity)) if generosity else float("nan"),
        "wealth_gini": gini(tokens) if tokens else float("nan"),
        "total_wealth": float(sum(tokens)),
        "agreement_rate": mean(trade["outcome"] == "agreement" for trade in trades),
        "rounds_to_agreement": mean(trade["rounds"] for trade in agreed),
        "mean_price": mean(trade["price"] for trade in agreed),
        "mean_lineage": mean(agent.lineage for agent in agents),
    }


SERIES = "#2a78d6"  # categorical slot 1: every panel plots a single series
NOTE = (
    "One generation = {rounds} rounds. Population and wealth inequality are measured at the end of each generation;"
    " generosity and trading cover all of its negotiations.\nGenerosity: each agent's mean offer to its"
    " partner as a share of the negotiated surplus, averaged over the agents that made offers; band = ±1 sd"
    " across those agents. Every value is in generations.csv."
)


def plot_economy(run_dir):
    """Chart generations.csv as plots.png: population, generosity, wealth inequality and how trading goes."""
    run_dir = Path(run_dir)
    metrics = read_metrics(run_dir / "generations.csv")
    if not metrics:
        return
    cfg = json.loads((run_dir / "config.json").read_text())
    x = metrics["generation"]
    line = {"color": SERIES, "linewidth": 1.5}

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150, layout="constrained")
        (population, generosity), (inequality, trading) = axes

        population.plot(x, metrics["population"], **line)
        finish_panel(population, "Population size", "generation", "living agents", ylim=(0, None), legend=False)
        population.yaxis.set_major_locator(MaxNLocator(integer=True))

        spread = metrics["generosity_std"]
        generosity.fill_between(
            x, metrics["mean_generosity"] - spread, metrics["mean_generosity"] + spread, color=SERIES, alpha=0.1, lw=0
        )
        generosity.plot(x, metrics["mean_generosity"], **line)
        finish_panel(
            generosity, "Mean policy generosity", "generation", "offer to partner, share of surplus", ylim=(0, None), legend=False
        )
        generosity.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

        inequality.plot(x, metrics["wealth_gini"], **line)
        finish_panel(inequality, "Wealth inequality", "generation", "Gini of token balances", ylim=(0, None), legend=False)

        if cfg["protocol"] == "sealed_bid":  # a single bidding round, so what varies is whether bids fit
            trading.plot(x, metrics["agreement_rate"], **line)
            finish_panel(trading, "Trades agreed", "generation", "share of negotiations", share=True, legend=False)
        else:  # alternating-offer deals almost always close, so what varies is how long haggling burns compute
            trading.plot(x, metrics["rounds_to_agreement"], **line)
            finish_panel(trading, "Rounds to agreement", "generation", "rounds per agreed trade", ylim=(0, None), legend=False)

        for ax in axes.flat:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        founders = f"mutated copies of {Path(cfg['init']).parent.name}" if cfg["init"] else "random networks"
        fig.suptitle(
            f"Population economy: {cfg['population']} founders ({founders}), {cfg['protocol']},"
            f" mutation std {cfg['mutation_std']}",
            x=0.01,
            ha="left",
            fontsize=12,
            fontweight="semibold",
        )
        fig.supxlabel(NOTE.format(rounds=cfg["rounds_per_generation"]), fontsize=8, color=INK_MUTED)
        fig.savefig(run_dir / "plots.png")
        plt.close(fig)
