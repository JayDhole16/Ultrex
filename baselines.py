"""Hand-written negotiators to measure the learned policies against.

A policy that closes every deal near 50/50 looks good until you ask what it beats. These four play the same
game through the same act() interface as a trained policy, so ippo.play_episode and tournament.py can pair
any of them against each other or against a checkpoint:

    rubinstein  the subgame-perfect play for alternating offers: keep 1/(1+discount) and accept anything at
                least as good as waiting a round. At discount 0.95 that is a 51/49 split, the theoretical
                target a learned policy should be measured against.
    conceder    the time-dependent negotiator of the automated-negotiation literature: open greedy and
                concede towards an even split as the rounds burn, accepting anything above today's demand.
    hardball    demands 90% and holds, taking what it can get in the final round rather than nothing.
    random      uniform demands, accepts half the time: the floor any learned policy has to clear.

Each reads the same observation features the networks see, so nothing here can see the opponent's private
valuation or its future moves.
"""

from __future__ import annotations

import numpy as np

from negotiation_env import ACCEPT, REJECT


class ScriptedPolicy:
    """A negotiator written by hand rather than trained. Shares the Policy interface, learns nothing."""

    name = "scripted"

    def __init__(self, max_rounds=20, discount=0.95, seed=0):
        self.max_rounds = max_rounds
        self.discount = discount
        self.rng = np.random.default_rng(seed)

    def demand(self, round_index, rounds_left):
        """The share of the pool to keep when making an offer, between 0 and 1."""
        raise NotImplementedError

    def accepts(self, offered, round_index, rounds_left):
        """Whether to take an offer worth `offered` (a share of the pool) rather than counter."""
        raise NotImplementedError

    def act(self, features, flags, greedy=False, capture=False, state=None):
        rows = len(features)
        response = np.full(rows, REJECT, dtype=np.int64)
        share = np.full(rows, 0.5, dtype=np.float32)
        for i, row in enumerate(features):
            offered = float(row[2])  # tokens on the table for me, as a share of the pool
            round_index = int(round(float(row[4]) * self.max_rounds))
            rounds_left = int(round(float(row[5]) * self.max_rounds))
            share[i] = float(np.clip(self.demand(round_index, rounds_left), 0.01, 0.99))
            if flags["can_accept"][i] and self.accepts(offered, round_index, rounds_left):
                response[i] = ACCEPT
        zeros = np.zeros(rows, dtype=np.float32)
        out = {
            "response": response,
            "share": share,
            "message": np.zeros(rows, dtype=np.int64),
            "logp": zeros,
            "value": zeros.copy(),
            "message_entropy": zeros.copy(),
            "state": None,
        }
        if capture:  # so a scripted agent can also appear in the live dashboard
            out["accept_prob"] = (response == ACCEPT).astype(np.float32)
            out["beta"] = np.ones((rows, 2), dtype=np.float32)
            out["activations"] = {"input": features, "layer1": np.zeros((rows, 0)), "layer2": np.zeros((rows, 0))}
        return out


class RubinsteinPolicy(ScriptedPolicy):
    """Keeps 1/(1+discount) and accepts anything worth at least what waiting one round would be worth."""

    name = "rubinstein"

    def demand(self, round_index, rounds_left):
        return 1.0 / (1.0 + self.discount)

    def accepts(self, offered, round_index, rounds_left):
        if rounds_left <= 0:
            return offered > 0  # last chance: anything beats no deal
        return offered >= self.discount / (1.0 + self.discount) - 1e-6


class ConcederPolicy(ScriptedPolicy):
    """Opens at `start` and concedes towards `reserve` as the deadline approaches."""

    name = "conceder"

    def __init__(self, *args, start=0.95, reserve=0.5, exponent=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.start, self.reserve, self.exponent = start, reserve, exponent

    def demand(self, round_index, rounds_left):
        elapsed = np.clip(round_index / max(self.max_rounds, 1), 0.0, 1.0)
        return self.reserve + (self.start - self.reserve) * (1.0 - elapsed) ** self.exponent

    def accepts(self, offered, round_index, rounds_left):
        if rounds_left <= 0:
            return offered > 0
        return offered >= self.demand(round_index, rounds_left)


class HardballPolicy(ScriptedPolicy):
    """Demands 90% and holds out, taking whatever is on the table in the final round."""

    name = "hardball"

    def __init__(self, *args, keep=0.9, floor=0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep, self.floor = keep, floor

    def demand(self, round_index, rounds_left):
        return self.keep

    def accepts(self, offered, round_index, rounds_left):
        return offered > 0 if rounds_left <= 0 else offered >= self.floor


class RandomPolicy(ScriptedPolicy):
    """Uniform demands, accepts half the time."""

    name = "random"

    def demand(self, round_index, rounds_left):
        return float(self.rng.random())

    def accepts(self, offered, round_index, rounds_left):
        return bool(self.rng.random() < 0.5)


BASELINES = {
    policy.name: policy for policy in (RubinsteinPolicy, ConcederPolicy, HardballPolicy, RandomPolicy)
}


def build_baselines(cfg, seed=0):
    """One of each scripted negotiator, configured for this run's protocol."""
    return {name: policy(max_rounds=cfg.max_rounds, discount=cfg.discount, seed=seed) for name, policy in BASELINES.items()}
