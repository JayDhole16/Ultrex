"""Two-agent token negotiation as a PettingZoo ParallelEnv, with pluggable protocols.

Two agents negotiate over a pool of ``pool_size`` tokens (default 100). At the
start of each episode each agent draws a private valuation multiplier
v_i ~ U(0.5, 2.0). If a deal gives agent i ``x_i`` tokens, agent i is paid
``discount**delay * v_i * x_i``, where ``delay`` is the number of rounds that
passed before the accepted offer was made. No deal or a hard timeout pays 0, and
rewards are 0 on every non-final step.

The protocol decides who acts on each step and how actions become a deal. Pass
it to the constructor by name or as an instance:
  * "alternating_offers" (AlternatingOffers, the default): agents take turns;
    the mover accepts the standing offer or rejects it and counters, for up to
    ``max_rounds`` rounds.
  * "sealed_bid" (SealedBid): a Nash demand game; both agents demand a share at
    once, and the demands are granted only if they fit in the pool.
Every protocol uses the same observation and action spaces, so a policy trained
under one protocol runs unchanged under another.

Cheap talk (optional, ``message_vocab`` > 0): before every bargaining move, both
agents send a message at once, a token from range(message_vocab), and each then
observes the other's. Messages never enter payoffs; whatever they come to mean
has to be learned. With the channel on, every bargaining step is preceded by a
talk step, and ``max_steps`` counts bargaining steps only.

Episode end:
  * terminated: the protocol reaches a deal or a final no-deal.
  * truncated:  hard timeout, either ``max_steps`` bargaining steps or
    ``timeout_seconds`` of wall-clock time since reset. Both are off by
    default and not visible to the agents.

Run ``python negotiation_env.py`` to execute the API checks and random-policy
rollout tests at the bottom of this file.
"""

from __future__ import annotations

import copy
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from functools import partial

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

REJECT = 0
ACCEPT = 1


def share_of_pool(weights, pool_size):
    """Whole tokens asked for by (for me, for opponent) offer weights; degenerate weights ask for half."""
    w = np.clip(np.asarray(weights, dtype=np.float64).reshape(2), 0.0, None)
    total = w.sum()
    share = w[0] / total if np.isfinite(total) and total > 0 else 0.5
    return int(np.rint(share * pool_size))


@dataclass
class Resolution:
    """What one protocol step did. ``outcome`` stays None while the negotiation continues."""

    event: str
    outcome: str | None = None  # "agreement" or "disagreement"
    allocation: tuple[int, int] | None = None  # tokens per agent id, on agreement
    delay: int = 0  # rounds before the accepted offer was made; payoffs are scaled by discount**delay


class Protocol(ABC):
    """Rules of one negotiation format: who acts on each step, and how their actions become a deal.

    The env owns everything protocol-independent (spaces, valuations, balances, payoffs, timeouts, cheap
    talk and the PettingZoo API) and passes actions to the protocol keyed by agent id (0 or 1). A protocol
    keeps its own per-episode state, including ``round``, which the env reports in the final infos. The
    env deep-copies protocol instances, so one instance can configure many envs.
    """

    name: str
    round: int

    @abstractmethod
    def reset(self, env, rng):
        """Start a new episode."""

    @abstractmethod
    def movers(self) -> tuple[int, ...]:
        """Ids of the agents whose actions the next bargaining step uses."""

    @abstractmethod
    def observe(self, env, agent) -> dict:
        """Protocol-specific observation keys for one agent id.

        Must return last_offer, round and rounds_remaining, plus the action flags can_accept (the
        agent's response is used) and can_offer (its offer is used, unless it accepts).
        """

    @abstractmethod
    def step(self, env, actions) -> Resolution:
        """Apply the movers' actions, given as {agent id: action}."""


class AlternatingOffers(Protocol):
    """Rubinstein-style bargaining. Exactly one agent, the mover, acts on each step.

    round 0: the first mover makes the opening offer.
    round t in 1..max_rounds: the mover accepts the offer made in round t-1, or rejects it and
    counters. A rejection in round max_rounds ends the episode with no deal. An offer made in round r
    is discounted by discount**r if accepted, so accepting the opening offer is undiscounted.
    """

    name = "alternating_offers"

    def __init__(self, randomize_first_mover: bool = True):
        self.randomize_first_mover = randomize_first_mover

    def reset(self, env, rng):
        self.mover = int(rng.integers(2)) if self.randomize_first_mover else 0
        self.round = 0
        self.offer = None  # standing offer as whole tokens per agent id
        self.last_received = np.zeros((2, 2), dtype=np.float32)  # per agent: (for me, for opponent)

    def movers(self):
        return (self.mover,)

    def observe(self, env, agent):
        moving = agent == self.mover
        return {
            "last_offer": self.last_received[agent].copy(),
            "round": self.round,
            "rounds_remaining": env.max_rounds - self.round,
            "can_accept": int(moving and self.offer is not None),
            "can_offer": int(moving and self.round < env.max_rounds),
        }

    def step(self, env, actions):
        mover, other = self.mover, 1 - self.mover
        name = env.possible_agents[mover]
        action = actions[mover]
        if int(action["response"]) == ACCEPT and self.offer is not None:
            event = f"round {self.round}: {name} accepts {self.offer[0]}/{self.offer[1]}"
            return Resolution(event, "agreement", self.offer, delay=self.round - 1)
        if self.round == env.max_rounds:
            return Resolution(f"round {self.round}: {name} rejects the final offer, no deal", "disagreement")

        # REJECT, or ACCEPT in round 0 when there is nothing to accept: put a (counter-)offer on the table.
        tokens = [0, 0]
        tokens[mover] = share_of_pool(action["offer"], env.pool_size)
        tokens[other] = env.pool_size - tokens[mover]
        self.offer = tuple(tokens)
        self.last_received[other] = (tokens[other], tokens[mover])
        event = f"round {self.round}: {name} offers {tokens[0]}/{tokens[1]}"
        self.round += 1
        self.mover = other
        return Resolution(event)


class SealedBid(Protocol):
    """Nash demand game: both agents submit a demand at once, in a single round.

    An agent's demand is the "for me" share of its offer vector, in whole tokens; the response is
    unused. If the demands sum to at most the pool, each agent gets exactly its demand and any
    remainder goes unallocated. Otherwise both get nothing. Payoffs are undiscounted.
    """

    name = "sealed_bid"

    def reset(self, env, rng):
        self.round = 0

    def movers(self):
        return (0, 1) if self.round == 0 else ()

    def observe(self, env, agent):
        return {
            "last_offer": np.zeros(2, dtype=np.float32),  # no offers are ever exchanged
            "round": self.round,
            "rounds_remaining": 0,  # the bidding round is the only one
            "can_accept": 0,
            "can_offer": int(self.round == 0),
        }

    def step(self, env, actions):
        demands = tuple(share_of_pool(actions[i]["offer"], env.pool_size) for i in (0, 1))
        self.round = 1
        event = f"round 0: demands {demands[0]}/{demands[1]}"
        if sum(demands) <= env.pool_size:
            return Resolution(f"{event}, deal", "agreement", demands)
        return Resolution(f"{event} exceed the pool of {env.pool_size}, no deal", "disagreement")


PROTOCOLS = {protocol.name: protocol for protocol in (AlternatingOffers, SealedBid)}


class NegotiationEnv(ParallelEnv):
    """Two-agent negotiation over a fixed token pool, under a pluggable protocol.

    Observation (Dict), per agent:
        balance           (1,) tokens the agent holds; 0 until a deal settles
        valuation         (1,) the agent's private multiplier v_i
        last_offer        (2,) tokens (for me, for opponent) in the most recent
                          offer the opponent made; all zeros if none
        round             Discrete(max_rounds + 1), current round
        rounds_remaining  Discrete(max_rounds + 1), rounds left after this one
        my_turn           Discrete(2), 1 if this agent's action is used this step
        can_accept        Discrete(2), 1 if this agent's response is used
        can_offer         Discrete(2), 1 if this agent's offer is used (unless it
                          accepts)
    With cheap talk (message_vocab > 0), also:
        heard_message     MultiBinary(message_vocab), one-hot of the opponent's
                          latest message; all zeros before the first
        sent_message      MultiBinary(message_vocab), one-hot of this agent's own
                          latest message
        can_talk          Discrete(2), 1 on talk steps, when both agents send a
                          message and nobody bargains

    Action (Dict), per agent:
        response  Discrete(2): REJECT or ACCEPT
        offer     Box(0, 1, (2,)): split weights (for me, for opponent). The env
                  normalizes them and rounds to whole tokens; degenerate
                  (all-zero) weights ask for half.
        message   Discrete(message_vocab), with cheap talk only; used on talk
                  steps

    step() requires actions only from agents with my_turn = 1 and ignores the rest.
    reset(options={"valuations": (v0, v1)}) sets the valuations instead of drawing them.
    observe_round=False reports round and rounds_remaining as 0 while the deadline still applies, which
    turns the game partially observable: an agent then has to remember how many rounds have burned.
    """

    metadata = {"name": "negotiation_v1", "render_modes": ["human", "ansi"]}

    def __init__(
        self,
        protocol: str | Protocol = "alternating_offers",
        *,
        pool_size: int = 100,
        max_rounds: int = 20,
        discount: float = 0.95,
        valuation_range: tuple[float, float] = (0.5, 2.0),
        message_vocab: int = 0,
        observe_round: bool = True,
        max_steps: int | None = None,
        timeout_seconds: float | None = None,
        render_mode: str | None = None,
    ):
        if isinstance(protocol, str):
            if protocol not in PROTOCOLS:
                raise ValueError(f"unknown protocol {protocol!r}; choose from {', '.join(PROTOCOLS)}")
            protocol = PROTOCOLS[protocol]()
        else:
            protocol = copy.deepcopy(protocol)  # protocols carry per-episode state, so each env needs its own
        low, high = valuation_range
        if pool_size < 1 or max_rounds < 1:
            raise ValueError("pool_size and max_rounds must be >= 1")
        if not 0.0 < discount <= 1.0:
            raise ValueError("discount must be in (0, 1]")
        if not 0.0 < low <= high:
            raise ValueError("valuation_range must satisfy 0 < low <= high")
        if message_vocab < 0:
            raise ValueError("message_vocab must be >= 0")
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"unsupported render_mode {render_mode!r}")

        self.protocol = protocol
        self.pool_size = pool_size
        self.max_rounds = max_rounds
        self.discount = discount
        self.valuation_range = (low, high)
        self.message_vocab = message_vocab
        self.observe_round = observe_round
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds
        self.render_mode = render_mode

        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = []
        # One space object per agent, so seeding one agent's sampler doesn't touch the other's.
        self.observation_spaces = {agent: self._make_observation_space() for agent in self.possible_agents}
        self.action_spaces = {agent: self._make_action_space() for agent in self.possible_agents}

        self._rng = np.random.default_rng()
        self._last_event = ""

    def _make_observation_space(self):
        low, high = self.valuation_range
        keys = {
            "balance": spaces.Box(0.0, self.pool_size, shape=(1,), dtype=np.float32),
            "valuation": spaces.Box(low, high, shape=(1,), dtype=np.float32),
            "last_offer": spaces.Box(0.0, self.pool_size, shape=(2,), dtype=np.float32),
            "round": spaces.Discrete(self.max_rounds + 1),
            "rounds_remaining": spaces.Discrete(self.max_rounds + 1),
            "my_turn": spaces.Discrete(2),
            "can_accept": spaces.Discrete(2),
            "can_offer": spaces.Discrete(2),
        }
        if self.message_vocab:
            keys["heard_message"] = spaces.MultiBinary(self.message_vocab)
            keys["sent_message"] = spaces.MultiBinary(self.message_vocab)
            keys["can_talk"] = spaces.Discrete(2)
        return spaces.Dict(keys)

    def _make_action_space(self):
        keys = {"response": spaces.Discrete(2), "offer": spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)}
        if self.message_vocab:
            keys["message"] = spaces.Discrete(self.message_vocab)
        return spaces.Dict(keys)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agents = self.possible_agents[:]
        valuations = (options or {}).get("valuations")
        if valuations is None:
            self._valuations = self._rng.uniform(*self.valuation_range, size=2)
        else:
            low, high = self.valuation_range
            self._valuations = np.asarray(valuations, dtype=np.float64).reshape(2)
            if not np.all((low <= self._valuations) & (self._valuations <= high)):
                raise ValueError(f"valuations must lie in {self.valuation_range}, got {valuations}")
        self._balances = np.zeros(2, dtype=np.int64)
        self._steps = 0
        self._start_time = time.monotonic()
        self._messages = [None, None]  # each agent's latest message
        self._talking = self.message_vocab > 0
        self.protocol.reset(self, self._rng)

        v0, v1 = self._valuations
        channel = f", cheap talk m0-m{self.message_vocab - 1}" if self.message_vocab else ""
        self._event(f"new episode ({self.protocol.name}{channel}): valuations {v0:.2f}/{v1:.2f} (agent_0/agent_1)")
        observations = {agent: self._observe(i) for i, agent in enumerate(self.agents)}
        return observations, {agent: {} for agent in self.agents}

    def step(self, actions):
        if not self.agents:
            raise RuntimeError("episode is over; call reset() before step()")
        if self._talking:
            return self._talk(actions)
        movers = self.protocol.movers()
        missing = [self.possible_agents[i] for i in movers if self.possible_agents[i] not in actions]
        if missing:
            raise KeyError(f"no action for {' or '.join(missing)}, whose turn it is")
        self._steps += 1
        result = self.protocol.step(self, {i: actions[self.possible_agents[i]] for i in movers})

        payoffs = np.zeros(2)
        event = result.event
        if result.outcome == "agreement":
            self._balances[:] = result.allocation
            payoffs = self.discount**result.delay * self._valuations * self._balances
            event += f", payoffs {payoffs[0]:.1f}/{payoffs[1]:.1f}"
        terminated = result.outcome is not None
        truncated = not terminated and self._timed_out()
        outcome = "timeout" if truncated else result.outcome
        if truncated:
            event += ", hard timeout"
        self._event(event)

        final = outcome is not None
        self._talking = self.message_vocab > 0 and not final
        observations = {agent: self._observe(i, final) for i, agent in enumerate(self.possible_agents)}
        rewards = {agent: float(payoffs[i]) for i, agent in enumerate(self.possible_agents)}
        terminations = dict.fromkeys(self.possible_agents, terminated)
        truncations = dict.fromkeys(self.possible_agents, truncated)
        infos = {agent: {} for agent in self.possible_agents}
        if final:
            for i, agent in enumerate(self.possible_agents):
                infos[agent] = {"outcome": outcome, "round": self.protocol.round, "tokens": int(self._balances[i])}
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def _talk(self, actions):
        """A talk step: both agents send a message, and nothing else happens."""
        missing = [agent for agent in self.possible_agents if agent not in actions]
        if missing:
            raise KeyError(f"no action for {' or '.join(missing)}, whose turn it is to talk")
        messages = [int(actions[agent]["message"]) for agent in self.possible_agents]
        if not all(0 <= message < self.message_vocab for message in messages):
            raise ValueError(f"messages must be in range({self.message_vocab}), got {messages}")
        self._messages = messages
        self._talking = False
        self._event(f"round {self.protocol.round}: agent_0 says m{messages[0]}, agent_1 says m{messages[1]}")

        observations = {agent: self._observe(i) for i, agent in enumerate(self.possible_agents)}
        rewards = dict.fromkeys(self.possible_agents, 0.0)
        not_done = dict.fromkeys(self.possible_agents, False)
        return observations, rewards, not_done, dict(not_done), {agent: {} for agent in self.possible_agents}

    def render(self):
        if self.render_mode == "ansi":
            return self._last_event
        if self.render_mode == "human":
            print(self._last_event)

    def _event(self, text):
        self._last_event = text
        if self.render_mode == "human":
            self.render()

    def _observe(self, i, final=False):
        talking = self._talking and not final
        obs = {
            "balance": np.array([self._balances[i]], dtype=np.float32),
            "valuation": np.array([self._valuations[i]], dtype=np.float32),
            **self.protocol.observe(self, i),
            "my_turn": int(not final and (talking or i in self.protocol.movers())),
        }
        if not self.observe_round:
            # A deliberate blind spot: the deadline still bites, but an agent has to keep track of it itself.
            obs["round"] = obs["rounds_remaining"] = 0
        if final or talking:
            obs["can_accept"] = obs["can_offer"] = 0  # nobody bargains once the episode is over, or while talking
        if self.message_vocab:
            obs["heard_message"] = self._one_hot(self._messages[1 - i])
            obs["sent_message"] = self._one_hot(self._messages[i])
            obs["can_talk"] = int(talking)
        return obs

    def _one_hot(self, message):
        vector = np.zeros(self.message_vocab, dtype=np.int8)
        if message is not None:
            vector[message] = 1
        return vector

    def _timed_out(self):
        if self.max_steps is not None and self._steps >= self.max_steps:
            return True
        return self.timeout_seconds is not None and time.monotonic() - self._start_time >= self.timeout_seconds


# ---------------------------------------------------------------------------
# Tests: `python negotiation_env.py` (or `pytest negotiation_env.py`)
# ---------------------------------------------------------------------------


def _action(response, offer=(0.5, 0.5)):
    return {"response": response, "offer": np.asarray(offer, dtype=np.float32)}


def test_pettingzoo_api():
    from pettingzoo.test import parallel_api_test, parallel_seed_test

    for protocol in PROTOCOLS:
        for vocab in (0, 5):
            parallel_api_test(NegotiationEnv(protocol, message_vocab=vocab), num_cycles=1000)
            parallel_seed_test(partial(NegotiationEnv, protocol, message_vocab=vocab))


def test_alternating_offers_episode():
    """Opening offer, counter-offer, acceptance: checks turn order, action flags, offers and discounting."""
    env = NegotiationEnv(AlternatingOffers(randomize_first_mover=False))
    obs, _ = env.reset(seed=0)
    v0, v1 = obs["agent_0"]["valuation"][0], obs["agent_1"]["valuation"][0]
    assert obs["agent_0"]["my_turn"] == 1 and obs["agent_1"]["my_turn"] == 0
    assert (obs["agent_0"]["can_accept"], obs["agent_0"]["can_offer"]) == (0, 1)  # nothing to accept yet
    assert not obs["agent_1"]["last_offer"].any() and "can_talk" not in obs["agent_0"]

    # Round 0: agent_0 opens 70/30 (weights get normalized). ACCEPT is moot with no offer on the table.
    obs, _, term, _, _ = env.step({"agent_0": _action(ACCEPT, (0.35, 0.15)), "agent_1": _action(ACCEPT)})
    assert list(obs["agent_1"]["last_offer"]) == [30, 70] and obs["agent_1"]["my_turn"] == 1
    assert (obs["agent_1"]["can_accept"], obs["agent_1"]["can_offer"]) == (1, 1)
    assert obs["agent_1"]["round"] == 1 and obs["agent_1"]["rounds_remaining"] == 19
    assert not any(term.values())

    # Round 1: agent_1 rejects and counters 60/40 in its favour. agent_0's ACCEPT is ignored (not its turn).
    obs, _, term, _, _ = env.step({"agent_0": _action(ACCEPT), "agent_1": _action(REJECT, (0.6, 0.4))})
    assert list(obs["agent_0"]["last_offer"]) == [40, 60] and obs["agent_0"]["my_turn"] == 1
    assert not any(term.values())

    # Round 2: agent_0 accepts the round-1 offer, so payoffs carry one factor of the discount.
    obs, rew, term, trunc, infos = env.step({"agent_0": _action(ACCEPT)})
    assert all(term.values()) and not any(trunc.values()) and env.agents == []
    assert np.isclose(rew["agent_0"], 0.95 * v0 * 40) and np.isclose(rew["agent_1"], 0.95 * v1 * 60)
    assert obs["agent_0"]["balance"][0] == 40 and obs["agent_1"]["balance"][0] == 60
    assert obs["agent_0"]["my_turn"] == 0 and obs["agent_0"]["can_accept"] == 0
    assert infos["agent_1"] == {"outcome": "agreement", "round": 2, "tokens": 60}

    # With max_rounds=1 the responder's first move is also its last: it can accept but not counter.
    env = NegotiationEnv(AlternatingOffers(randomize_first_mover=False), max_rounds=1)
    env.reset(seed=0)
    obs, *_ = env.step({"agent_0": _action(REJECT)})
    assert (obs["agent_1"]["can_accept"], obs["agent_1"]["can_offer"]) == (1, 0)
    _, rew, term, _, infos = env.step({"agent_1": _action(REJECT)})
    assert all(term.values()) and infos["agent_0"]["outcome"] == "disagreement" and rew["agent_0"] == 0


def test_sealed_bid():
    """Both agents bid at once; compatible demands are granted undiscounted, incompatible ones pay nothing."""
    env = NegotiationEnv("sealed_bid")
    obs, _ = env.reset(seed=0)
    for agent in env.possible_agents:
        assert obs[agent]["my_turn"] == 1 and (obs[agent]["can_accept"], obs[agent]["can_offer"]) == (0, 1)
    v0, v1 = obs["agent_0"]["valuation"][0], obs["agent_1"]["valuation"][0]

    try:
        env.step({"agent_0": _action(REJECT)})
    except KeyError:
        pass
    else:
        raise AssertionError("a sealed-bid step must require both agents' demands")

    # Demands of 45 and 50 fit in the pool: each agent gets its demand and 5 tokens go unallocated.
    # agent_1's ACCEPT is ignored; only the offer vector counts.
    actions = {"agent_0": _action(REJECT, (0.45, 0.55)), "agent_1": _action(ACCEPT, (0.5, 0.5))}
    obs, rew, term, trunc, infos = env.step(actions)
    assert all(term.values()) and not any(trunc.values()) and env.agents == []
    assert np.isclose(rew["agent_0"], v0 * 45) and np.isclose(rew["agent_1"], v1 * 50)
    assert obs["agent_1"]["balance"][0] == 50 and obs["agent_1"]["my_turn"] == 0
    assert infos["agent_0"] == {"outcome": "agreement", "round": 1, "tokens": 45}

    # Demands of 60 and 55 overshoot the pool: no deal.
    env.reset(seed=0)
    _, rew, term, _, infos = env.step({"agent_0": _action(REJECT, (0.6, 0.4)), "agent_1": _action(REJECT, (0.55, 0.45))})
    assert all(term.values()) and infos["agent_1"]["outcome"] == "disagreement"
    assert rew == {"agent_0": 0.0, "agent_1": 0.0}


def test_cheap_talk():
    """Talk steps precede bargaining steps, messages reach the other agent, and payoffs ignore them."""
    env = NegotiationEnv(AlternatingOffers(randomize_first_mover=False), message_vocab=5)
    obs, _ = env.reset(seed=0)
    for agent in env.possible_agents:
        assert obs[agent]["my_turn"] == 1 and obs[agent]["can_talk"] == 1
        assert (obs[agent]["can_accept"], obs[agent]["can_offer"]) == (0, 0) and not obs[agent]["heard_message"].any()

    try:
        env.step({"agent_0": {"message": 3}})
    except KeyError:
        pass
    else:
        raise AssertionError("a talk step must require both agents' messages")

    # Round 0: both talk, then agent_0 opens 70/30 having heard agent_1's m1.
    obs, rew, term, _, _ = env.step({"agent_0": {"message": 3}, "agent_1": {"message": 1}})
    assert rew == {"agent_0": 0.0, "agent_1": 0.0} and not any(term.values())
    assert list(obs["agent_0"]["heard_message"]) == [0, 1, 0, 0, 0]
    assert list(obs["agent_0"]["sent_message"]) == [0, 0, 0, 1, 0]
    assert obs["agent_0"]["can_talk"] == 0 and obs["agent_0"]["can_offer"] == 1
    assert obs["agent_0"]["my_turn"] == 1 and obs["agent_1"]["my_turn"] == 0
    obs, *_ = env.step({"agent_0": _action(REJECT, (0.7, 0.3))})
    assert obs["agent_1"]["can_talk"] == 1 and obs["agent_1"]["round"] == 1

    # Round 1: both talk again, then agent_1 accepts. The payoffs match the same deal struck without talk.
    env.step({"agent_0": {"message": 0}, "agent_1": {"message": 4}})
    _, talk_rewards, term, _, infos = env.step({"agent_1": _action(ACCEPT)})
    assert all(term.values()) and infos["agent_1"] == {"outcome": "agreement", "round": 1, "tokens": 30}
    silent = NegotiationEnv(AlternatingOffers(randomize_first_mover=False))
    silent.reset(seed=0)
    silent.step({"agent_0": _action(REJECT, (0.7, 0.3))})
    _, silent_rewards, *_ = silent.step({"agent_1": _action(ACCEPT)})
    assert talk_rewards == silent_rewards

    # The hard timeout counts bargaining steps only.
    env = NegotiationEnv(AlternatingOffers(randomize_first_mover=False), message_vocab=5, max_steps=1)
    env.reset(seed=0)
    _, _, _, trunc, _ = env.step({"agent_0": {"message": 0}, "agent_1": {"message": 0}})
    assert not any(trunc.values())
    _, _, _, trunc, infos = env.step({"agent_0": _action(REJECT)})
    assert all(trunc.values()) and infos["agent_0"]["outcome"] == "timeout"


def test_reset_with_valuations():
    """reset(options={"valuations": ...}) fixes the valuations instead of drawing them."""
    env = NegotiationEnv()
    obs, _ = env.reset(seed=0, options={"valuations": (1.5, 0.7)})
    assert np.isclose(obs["agent_0"]["valuation"][0], 1.5) and np.isclose(obs["agent_1"]["valuation"][0], 0.7)
    try:
        env.reset(options={"valuations": (3.0, 1.0)})
    except ValueError:
        pass
    else:
        raise AssertionError("valuations outside the valuation range must be rejected")


def test_hidden_round():
    """observe_round=False hides the clock from the agents without changing when the deadline bites."""
    env = NegotiationEnv(AlternatingOffers(randomize_first_mover=False), max_rounds=1, observe_round=False)
    obs, _ = env.reset(seed=0)
    assert obs["agent_0"]["round"] == 0 and obs["agent_0"]["rounds_remaining"] == 0
    obs, *_ = env.step({"agent_0": _action(REJECT)})
    assert obs["agent_1"]["round"] == 0 and obs["agent_1"]["rounds_remaining"] == 0  # still round 1 underneath
    assert (obs["agent_1"]["can_accept"], obs["agent_1"]["can_offer"]) == (1, 0)  # the deadline is real
    _, _, term, _, infos = env.step({"agent_1": _action(REJECT)})
    assert all(term.values()) and infos["agent_0"]["outcome"] == "disagreement"


def run_random_rollouts(n_episodes=500, **env_kwargs):
    """Play uniform-random policies for n episodes, asserting protocol invariants on every episode.

    Returns (outcome counts, mean episode length in bargaining steps, mean return per agent).
    """
    env = NegotiationEnv(**env_kwargs)
    alternating = isinstance(env.protocol, AlternatingOffers)
    for i, agent in enumerate(env.possible_agents):
        env.action_space(agent).seed(i)
    outcomes, lengths = Counter(), []
    mean_returns = dict.fromkeys(env.possible_agents, 0.0)

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=episode)
        returns = dict.fromkeys(env.possible_agents, 0.0)
        steps = 0
        while env.agents:
            talking = any(obs[agent].get("can_talk", 0) for agent in env.possible_agents)
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            obs, rewards, terms, truncs, infos = env.step(actions)
            steps += not talking
            for agent in env.possible_agents:
                flags = obs[agent]["can_accept"] or obs[agent]["can_offer"] or obs[agent].get("can_talk", 0)
                assert env.observation_space(agent).contains(obs[agent]), obs[agent]
                assert obs[agent]["my_turn"] == int(flags)
                assert not talking or rewards[agent] == 0
                returns[agent] += rewards[agent]

        outcome = infos["agent_0"]["outcome"]
        outcomes[outcome] += 1
        lengths.append(steps)
        assert steps <= env.max_rounds + 1 and steps <= (env.max_steps or steps)
        if outcome == "agreement":
            assert all(terms.values()) and not any(truncs.values())
            tokens = {agent: infos[agent]["tokens"] for agent in env.possible_agents}
            assert sum(tokens.values()) == env.pool_size if alternating else sum(tokens.values()) <= env.pool_size
            # In both protocols the accepted offer or bid was made in the round before the reported one.
            discount = env.discount ** (infos["agent_0"]["round"] - 1)
            for agent in env.possible_agents:
                assert obs[agent]["balance"][0] == tokens[agent]
                assert np.isclose(returns[agent], discount * obs[agent]["valuation"][0] * tokens[agent], rtol=1e-5)
        elif outcome == "disagreement":
            assert all(terms.values()) and not any(truncs.values())
            assert not alternating or infos["agent_0"]["round"] == env.max_rounds
            assert all(r == 0 for r in returns.values())
        else:
            assert outcome == "timeout" and all(truncs.values()) and not any(terms.values())
            assert all(r == 0 for r in returns.values())
        for agent in env.possible_agents:
            mean_returns[agent] += returns[agent] / n_episodes

    return outcomes, float(np.mean(lengths)), mean_returns


def test_random_rollouts():
    configs = {
        "default": {},
        "max_rounds=2": {"max_rounds": 2},  # random play often rejects the final offer
        "max_steps=3": {"max_steps": 3},  # step-count hard timeout
        "timeout_seconds=0": {"timeout_seconds": 0.0},  # wall-clock hard timeout fires on the first step
        "sealed_bid": {"protocol": "sealed_bid"},  # random demands overshoot the pool about half the time
        "cheap_talk": {"message_vocab": 5},
        "sealed_bid+talk": {"protocol": "sealed_bid", "message_vocab": 5},
    }
    results = {name: run_random_rollouts(**kwargs) for name, kwargs in configs.items()}
    for name, (outcomes, mean_length, mean_returns) in results.items():
        returns = ", ".join(f"{agent} {value:.1f}" for agent, value in mean_returns.items())
        print(f"{name:>17}: {dict(outcomes)}, mean length {mean_length:.2f}, mean return {returns}")

    assert results["default"][0]["agreement"] > 0
    assert results["max_rounds=2"][0]["disagreement"] > 0
    assert results["max_steps=3"][0]["timeout"] > 0
    assert results["timeout_seconds=0"][0] == Counter(timeout=500)
    assert results["sealed_bid"][0]["agreement"] > 0 and results["sealed_bid"][0]["disagreement"] > 0
    assert results["cheap_talk"][0]["agreement"] > 0 and results["sealed_bid+talk"][0]["agreement"] > 0


if __name__ == "__main__":
    test_pettingzoo_api()
    test_alternating_offers_episode()
    test_sealed_bid()
    test_cheap_talk()
    test_hidden_round()
    test_reset_with_valuations()
    test_random_rollouts()

    print("\nSample episodes, random policies:")
    for protocol, vocab in (("alternating_offers", 0), ("alternating_offers", 5), ("sealed_bid", 5)):
        env = NegotiationEnv(protocol, max_rounds=6, message_vocab=vocab, render_mode="human")
        env.reset(seed=3)
        for i, agent in enumerate(env.possible_agents):
            env.action_space(agent).seed(3 + i)
        while env.agents:
            env.step({agent: env.action_space(agent).sample() for agent in env.agents})
    print("\nAll checks passed.")
