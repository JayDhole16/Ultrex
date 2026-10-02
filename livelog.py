"""The demo event stream: a blow-by-blow log of a run, written for the animated dashboard.

train.py writes generation summaries to events.jsonl. With --live-log it also writes live.jsonl through
this module: one line per thing that happens inside a generation, in the order it happens, so the dashboard
can animate the run as it unfolds. Each line is {"t": seconds since the run started, "type": ..., ...}.

Event types, in the order a round produces them:
    run_start          the configuration and the founding population
    round_start        every agent's balance and compute, this round's pairings, and which pair is the
                       spotlight: the one negotiation the demo follows step by step
    negotiation_start  the spotlight pair: who is buying compute, what each of them needs, the surplus
    decision           one spotlight move: a message, an offer, an accept or a reject, together with what
                       the policy was thinking (the acting network's activations, its accept probability,
                       the Beta parameters behind its offer, its message probabilities and its value estimate)
    deal / no_deal     how the spotlight negotiation ended, with the transcript the env rendered
    round_end          balances after work and upkeep, every pair's outcome, and who was born or died
    generation_end     the generation's metrics, matching that generation's events.jsonl line

With a pace above zero the run sleeps between spotlight decisions, so the negotiation unfolds at human
speed instead of in a millisecond. Only the spotlight negotiation is slowed; everything else runs flat out.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np


def jsonable(value):
    """value with NaN and infinities as None and numpy scalars as plain Python numbers, ready for strict JSON."""
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


class NullLog:
    """What a run uses when the demo stream is off: every call does nothing."""

    enabled = False

    def emit(self, type, **fields):
        pass

    def beat(self, multiplier=1.0):
        pass

    def close(self):
        pass


class LiveLog(NullLog):
    """Appends demo events to live.jsonl, and paces the spotlight negotiation for the eye."""

    enabled = True

    def __init__(self, path, pace=0.0):
        self.file = open(path, "a", encoding="utf-8")
        self.pace = max(0.0, pace)  # seconds per spotlight decision; 0 runs at full speed
        self.start = time.monotonic()

    def emit(self, type, **fields):
        event = jsonable({"t": round(time.monotonic() - self.start, 3), "type": type, **fields})
        self.file.write(json.dumps(event, allow_nan=False) + "\n")
        self.file.flush()  # the dashboard is reading this file as it is written

    def beat(self, multiplier=1.0):
        if self.pace:
            time.sleep(self.pace * multiplier)

    def close(self):
        self.file.close()
