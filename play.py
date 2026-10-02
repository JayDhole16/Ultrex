"""Arena Lab: negotiate against a trained agent, set up a duel, or train one on stage.

    python play.py                       # then open http://127.0.0.1:8100/
    python play.py --checkpoint runs/<run>/agents.pt --port 8100
    python play.py --live --from-scratch # random weights that start learning by self-play at once

This is the interactive stage, deliberately separate from dashboard.py. The dashboard watches a training
run and can never touch it; this server owns its own agents, loaded from a checkpoint, and nothing it does
reaches a run in progress. It writes no files.

Three things a visitor can do, all of them live:

    play      take one seat yourself. You set each side's need, the deadline and how costly delay is, then
              make offers, accept, reject or send a cheap-talk token. The trained network answers your
              actual offers, and every one of its decisions ships with the features it saw, its accept
              probability, the Beta it drew its counter-offer from and both hidden layers.
    watch     put two policies against each other in a scenario you choose, and step through it move by move.
              Any trained checkpoint, or a scripted negotiator: rubinstein, conceder, hardball, random.
    adapt     point the agent at an opponent and let it learn against it here and now. Each PPO update
              reports what the agent would open with, so a policy that starts at an even split can be
              watched sliding towards conceding (or towards holding firm) while the audience watches.
    live      with --live, both agents learn by playing each other from the moment the server starts, and
              keep learning while you play them: every update swaps new weights into the agent across the
              table, and each of its moves is stamped with the update that made it. With --from-scratch (or
              no checkpoint on disk) they start from random weights, so the whole of learning happens on stage.

Nothing is precomputed: every number the page shows is produced by a forward pass through the loaded
weights on the observation in front of it. /api/state reports the checkpoint's SHA-256 so a sceptic can
check that the file on disk is what is playing.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import torch

from baselines import BASELINES
from ippo import (
    ACCEPT,
    AGENTS,
    REJECT,
    Config,
    Match,
    PPOAgent,
    RolloutCollector,
    decision_record,
    entropy_coef,
    env_action,
    load_agents,
    make_env,
    policy_inputs,
)
from livelog import jsonable

PAGE = Path(__file__).with_name("play.html")
SCRIPT = Path(__file__).with_name("play.js")
LEARNING_RATE_BOOST = 6  # on-stage training only: visible movement within a minute
STEPS_PER_UPDATE = 4096
HISTORY_POINTS = 400  # live learning can run for hours; past this the curve keeps every other point
LOWBALL = 0.2  # the insulting offer the live curve tracks: would it take 20 of 100?
WATCH_PACE = 0.7  # seconds between steps when no human is playing, so a duel can be followed
SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'unsafe-inline'; connect-src 'self';"
        " base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
}


@dataclass
class Scenario:
    """What the visitor sets before a negotiation: the stakes, the clock and who is playing."""

    need_a: float = 1.25  # each side's valuation multiplier, the env's "need"
    need_b: float = 1.25
    protocol: str = "alternating_offers"
    max_rounds: int = 8
    discount: float = 0.95
    observe_round: int = 1
    seat_a: str = "you"  # "you", "llm", a loaded checkpoint's agent, or a scripted negotiator
    seat_b: str = "agent_0"
    topic: str = "who gets the bigger office"  # what the language model argues about, when one is playing


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


class Lab:
    """Everything one visitor's session owns: the policies, the negotiation in progress, the live trainer."""

    def __init__(self, checkpoint=None, device="cpu", language_model=None, language_threads=None,
                 fresh_config=None, live=False):
        self.lock = threading.RLock()
        self.device = torch.device(device)
        self.origin = Path(checkpoint) if checkpoint else None  # what "Reset to checkpoint" goes back to
        self.fresh_config = fresh_config or Config(max_rounds=Scenario.max_rounds)
        self.live = live  # keep both agents learning by self-play, from start-up and after every reset
        self.training_run = 0  # bumped whenever training is started or abandoned; a stale trainer checks it
        self.load_weights(fresh=self.origin is None)
        self.scenario = Scenario(protocol=self.cfg.protocol)
        self.version = 0
        self.env = None
        self.obs = None
        self.lines = []
        self.states = [None, None]
        self.moves = []
        self.table = None
        self.outcome = None
        self.network = None
        self.turn = None
        self.trainer = None
        self.thinking = None  # which seat's language model is mid-sentence, if any
        self.episode = 0  # bumped on every start, so an old runner knows to stop
        self._debater = None
        self.language_model = language_model  # None: the language layer's default
        self.language_threads = language_threads
        self.training = {"running": False, "opponent": "", "history": [], "started": None}

    def load_weights(self, fresh):
        """The checkpoint this session started from, or brand-new random weights that know nothing yet."""
        if fresh:
            self.cfg = self.fresh_config
            self.seed = int(time.time() * 1000) % 100_000
            torch.manual_seed(self.seed)
            self.agents = {name: PPOAgent(name, self.cfg, self.device) for name in AGENTS}
            self.checkpoint, self.fingerprint = None, None
        else:
            self.cfg, self.agents = load_agents(self.origin, self.device)
            self.checkpoint, self.fingerprint, self.seed = self.origin, sha256(self.origin), None
        self.weights_version = 0  # PPO updates applied since these weights were loaded
        self.weights_at = time.time()

    # ---- policies -------------------------------------------------------------------------------

    def roster(self):
        """Everything that can take a seat, by the name the page uses."""
        names = {"you": "you"}
        names.update({name: f"trained {name} ({self.cfg.policy})" for name in self.agents})
        names.update({name: f"scripted {name}" for name in BASELINES})
        if language_status(self.language_model).get("available"):
            names["llm"] = "language model (local)"
        return names

    def policy_for(self, name):
        if name in self.agents:
            return self.agents[name]
        if name in BASELINES:
            return BASELINES[name](max_rounds=self.scenario.max_rounds, discount=self.scenario.discount)
        return None  # "you" and "llm" are handled by hand: one waits for a click, the other for a sentence

    def debater(self):
        """The local language model, loaded the first time someone actually seats it."""
        if self._debater is None:
            from language import DEFAULT_MODEL, DEFAULT_THREADS, Debater

            self._debater = Debater(self.language_model or DEFAULT_MODEL, self.language_threads or DEFAULT_THREADS)
        return self._debater

    def llm_prepare(self, seat):
        """What the language model needs to hear before it speaks, read from the env under the lock.

        Returns ("act", action) when there is nothing to say, or ("speak", request) for a real turn. The
        speaking itself happens outside the lock, so the page keeps updating while the model thinks.
        """
        from language import describe_position

        name = AGENTS[seat]
        obs = self.obs[name]
        pool = self.env.pool_size
        if obs.get("can_talk", 0):
            # A talk step carries a cheap-talk token and nothing else. The model's sentence is its talk, and
            # it comes with its offer; a move shown here would be one the env never receives.
            return "act", {"response": REJECT, "offer": np.array([0.5, 0.5], dtype=np.float32), "message": 0}
        offered = float(obs["last_offer"][0])
        can_accept = bool(obs["can_accept"])
        final = can_accept and not obs.get("can_offer", 1)  # nothing left to counter with: take it or leave it
        spoken = [move for move in self.moves if move.get("say")]
        last_words = spoken[-1]["say"] if spoken and spoken[-1]["seat"] != seat else ""
        previous = [move for move in self.moves if move["seat"] == seat and move.get("share") is not None]
        return "speak", {
            "seat": seat,
            "round": int(obs["round"]),
            "offered": offered,
            "can_accept": can_accept,
            "final": final,
            "pool": pool,
            "speak": {
                "topic": self.scenario.topic,
                "pool": pool,
                "position": describe_position(pool, round(offered), can_accept, int(obs["round"]), last_words, final),
                "rounds_left": int(obs["rounds_remaining"]),
                "keep_hint": previous[-1]["share"] if previous else 0.5,
                "final": final,
            },
        }

    def llm_commit(self, request, result):
        """Turn what the model said and chose into its move, and log it like any other."""
        seat, pool, offered = request["seat"], request["pool"], request["offered"]
        keep = float(result["keep"])
        if request["final"]:
            takes_it = bool(result["accept"])
        else:
            # Its own argument sets its price: it takes a deal that already gives it what it just asked for.
            takes_it = request["can_accept"] and offered / pool >= keep - 0.02
        kind = "accept" if takes_it else ("reject" if request["final"] else "offer")
        move = {
            "seat": seat,
            "who": "llm",
            "round": request["round"],
            "kind": kind,
            "say": result["say"] or "(the model said nothing intelligible)",
            "share": keep if kind == "offer" else None,
            "considered": result.get("considered", []),
            "seconds": result.get("seconds"),
            "error": result.get("error"),
        }
        if kind == "offer":
            move["keeps"] = round(keep * pool)
            move["gives"] = pool - move["keeps"]
            self.table = {"keeps": move["keeps"], "gives": move["gives"], "from": "llm", "seat": seat}
        self.moves.append(move)
        self.bump()
        action = {"response": ACCEPT if takes_it else REJECT, "offer": np.array([keep, 1 - keep], dtype=np.float32)}
        if self.env.message_vocab:
            action["message"] = 0
        return action

    # ---- a negotiation --------------------------------------------------------------------------

    def start(self):
        """Begin a negotiation under the current scenario."""
        scenario = self.scenario
        low, high = 0.5, 2.0
        env = make_env(
            self.cfg,
            scenario.protocol,
            max_rounds=max(1, int(scenario.max_rounds)),
            discount=float(np.clip(scenario.discount, 0.5, 1.0)),
            observe_round=bool(scenario.observe_round),
            render_mode="ansi",
        )
        needs = [float(np.clip(scenario.need_a, low, high)), float(np.clip(scenario.need_b, low, high))]
        self.env = env
        self.obs = env.reset(seed=int(time.time() * 1000) % (2**31), options={"valuations": needs})[0]
        self.lines = [env.render()]
        self.states = [None, None]
        self.moves = []
        self.table = None
        self.outcome = None
        self.network = None
        self.thinking = None
        self.turn = None
        self.episode += 1  # retires any runner still working on the previous negotiation

    def seats(self):
        return [self.scenario.seat_a, self.scenario.seat_b]

    def waiting_for_human(self):
        """The seat the visitor holds, if the env is waiting on it."""
        if not self.env or not self.env.agents:
            return None
        for seat, name in enumerate(AGENTS):
            if self.obs[name]["my_turn"] and self.seats()[seat] == "you":
                return seat
        return None

    def language(self):
        status = language_status(self.language_model)
        if self._debater is not None and self._debater.error:
            return {**status, "available": False, "reason": self._debater.error}
        return status

    def run(self, episode, human_action=None):
        """Carry the negotiation forward on a background thread, so the request that asked for it returns.

        Call it without holding the lock. The page polls /api/state and sees each move as it lands. The
        request waits until the runner has settled — the visitor's turn again, the end, or a language model
        starting to think — so instant replies (a trained agent answering the visitor) come back with it.
        """
        settled = threading.Event()
        thread = threading.Thread(target=self.advance, args=(episode, human_action, settled), daemon=True)
        thread.start()
        settled.wait(timeout=0.25)

    def advance(self, episode, human_action=None, settled=None):
        """Let every policy that can move take its move, stopping when it is the visitor's turn.

        One env step at a time, each under the lock, and the lock is let go while a language model is
        speaking: a turn takes seconds, and the page should show it thinking rather than freeze. Starting a
        new negotiation bumps ``episode``, which retires a runner still working on the old one.

        A step can need moves from both seats at once (a talk step, or sealed bids), so the visitor's move
        is collected before any agent is asked to decide: otherwise an agent's decision would be computed,
        shown, and then thrown away while the page waits for a human.
        """
        try:
            while True:
                with self.lock:
                    env = self.env
                    if episode != self.episode or not env or not env.agents:
                        return
                    movers = [seat for seat, name in enumerate(AGENTS) if self.obs[name]["my_turn"]]
                    yours = [seat for seat in movers if self.seats()[seat] == "you"]
                    if yours and human_action is None:
                        # The id ties a click to this turn: a late second click must not become the next move.
                        self.turn = {**self.turn_info(yours[0]), "id": f"{episode}.{len(self.moves)}"}
                        self.bump()
                        return
                    actions = {AGENTS[seat]: self.human_move(seat, human_action) for seat in yours}
                    watching = "you" not in self.seats()
                    human_action = None
                    requests = []
                    for seat in movers:
                        if self.seats()[seat] != "llm":
                            continue
                        kind, payload = self.llm_prepare(seat)
                        if kind == "act":
                            actions[AGENTS[seat]] = payload
                        else:
                            requests.append(payload)
                    if requests:
                        debater = self.debater()
                        self.thinking = {"seat": requests[0]["seat"], "who": "llm", "since": time.time(),
                                         "loading": not debater.ready}
                        self.bump()
                if requests and settled:
                    settled.set()  # the visitor's request can return now: the page will show it thinking

                spoken = []
                for request in requests:  # outside the lock: seconds of CPU, and the page keeps polling
                    debater.load()  # the first turn pays for loading; the seconds shown are the turn's own
                    began = time.time()
                    result = debater.speak(**request["speak"])
                    spoken.append((request, {**result, "seconds": round(time.time() - began, 1)}))

                with self.lock:
                    if episode != self.episode:  # a new negotiation began while the model was thinking
                        return
                    self.thinking = None
                    for request, result in spoken:
                        actions[AGENTS[request["seat"]]] = self.llm_commit(request, result)
                    for seat in movers:
                        name = AGENTS[seat]
                        if name in actions:
                            continue
                        features, flags = policy_inputs([self.obs[name]], env)
                        policy = self.policy_for(self.seats()[seat])
                        out = policy.act(features, flags, capture=True, state=self.states[seat])
                        self.states[seat] = out.get("state")
                        self.record(seat, decision_record(self.obs[name], flags, out, 0, seat, policy), out)
                        actions[name] = env_action(out, 0, env.message_vocab)
                    self.obs, rewards, terminations, truncations, infos = env.step(actions)
                    self.lines.append(env.render())
                    if any(terminations.values()) or any(truncations.values()):
                        self.finish(infos, rewards)
                        return
                    self.turn = None
                    self.bump()
                if watching and not spoken:
                    time.sleep(WATCH_PACE)  # two machines duelling: slow enough to follow, move by move
        except Exception:
            traceback.print_exc()
            with self.lock:
                if episode == self.episode:
                    self.thinking = None
                    self.outcome = {
                        "outcome": "error", "rounds": 0, "tokens": [0, 0], "payoffs": [0.0, 0.0],
                        "transcript": [line.strip() for line in self.lines]
                        + ["the negotiation stopped on a server error; the terminal has the traceback"],
                    }
                    self.bump()
        finally:
            if settled:
                settled.set()

    def human_move(self, seat, action):
        """Turn the page's action into an env action, and log it like any other move."""
        name = AGENTS[seat]
        flags = {key: np.array([bool(self.obs[name].get(key, 0))]) for key in ("can_accept", "can_offer", "can_talk")}
        kind = action.get("action", "offer")
        share = float(np.clip(action.get("share", 0.5), 0.01, 0.99))
        token = int(action.get("message", 0))
        response = ACCEPT if kind == "accept" else REJECT
        move = {
            "seat": seat,
            "who": "you",
            "round": int(self.obs[name]["round"]),
            "kind": (
                "talk" if flags["can_talk"][0]
                else "accept" if kind == "accept"
                else "offer" if flags["can_offer"][0]
                else "reject"
            ),
            "share": share,
            "message": token,
        }
        if move["kind"] == "offer":
            move["keeps"] = round(share * self.env.pool_size)
            move["gives"] = self.env.pool_size - move["keeps"]
            self.table = {"keeps": move["keeps"], "gives": move["gives"], "from": "you", "seat": seat}
        self.moves.append(move)
        self.bump()
        env_move = {"response": response, "offer": np.array([share, 1 - share], dtype=np.float32)}
        if self.env.message_vocab:
            env_move["message"] = int(np.clip(token, 0, self.env.message_vocab - 1))
        return env_move

    def record(self, seat, decision, out):
        """Log an agent's move, and keep what its network was doing while it decided."""
        name = self.seats()[seat]
        move = {
            "seat": seat,
            "who": name,
            "round": decision["round"],
            "kind": decision["kind"],
            "message": decision["message"],
        }
        if name in self.agents:  # a trained network: which weights made this move, while they keep changing
            move["weights"] = self.weights_version
        if decision["kind"] == "offer":
            keeps = self.env.pool_size - round(decision["generosity"] * self.env.pool_size)
            move["keeps"] = keeps
            move["gives"] = self.env.pool_size - keeps
            move["share"] = 1 - decision["generosity"]
            self.table = {"keeps": keeps, "gives": self.env.pool_size - keeps, "from": name, "seat": seat}
        self.moves.append(move)
        activations = out.get("activations")
        if activations is not None and len(activations["layer1"][0]):  # a scripted move leaves the last network up
            self.network = {
                "who": name,
                "round": decision["round"],
                "kind": decision["kind"],
                "features": activations["input"][0],
                "layer1": activations["layer1"][0],
                "layer2": activations["layer2"][0],
                "accept_prob": float(out["accept_prob"][0]),
                "beta": [float(v) for v in out["beta"][0]],
                "message_probs": [float(p) for p in out["message_probs"][0]] if "message_probs" in out else None,
                "value": float(out["value"][0]),
                "weights": self.weights_version,
            }
        self.bump()

    def finish(self, infos, rewards):
        info = infos[AGENTS[0]]
        self.outcome = {
            "outcome": info["outcome"],
            "rounds": info["round"],
            "tokens": [infos[name]["tokens"] for name in AGENTS],
            "payoffs": [float(rewards[name]) for name in AGENTS],
            "transcript": [line.strip() for line in self.lines],
        }
        self.turn = None
        self.bump()

    def turn_info(self, seat):
        name = AGENTS[seat]
        return {
            "seat": seat,
            "round": int(self.obs[name]["round"]),
            "rounds_remaining": int(self.obs[name]["rounds_remaining"]),
            "can_accept": bool(self.obs[name]["can_accept"]),
            "can_offer": bool(self.obs[name]["can_offer"]),
            "can_talk": bool(self.obs[name].get("can_talk", 0)),
            "offered_to_you": float(self.obs[name]["last_offer"][0]),
            "need": float(self.obs[name]["valuation"][0]),
        }

    # ---- learning on stage ----------------------------------------------------------------------

    def opening_offer(self, agent):
        """What this policy would open with right now: the mean of its Beta on a fresh negotiation."""
        env = make_env(self.cfg, self.scenario.protocol, max_rounds=self.scenario.max_rounds, render_mode="ansi")
        obs = env.reset(seed=0, options={"valuations": [self.scenario.need_a, self.scenario.need_b]})[0]
        name = AGENTS[0] if obs[AGENTS[0]]["my_turn"] else AGENTS[1]
        features, flags = policy_inputs([obs[name]], env)
        out = agent.act(features, flags, capture=True)
        alpha, beta = out["beta"][0]
        return float(alpha / (alpha + beta))

    def acceptance(self, agent):
        """The smallest share this policy would accept right now (None if it has no price yet), and how
        likely it is to take a lowball.

        Round 1, the scenario's need. The features are built straight from the observation layout, so one
        forward pass sweeps forty-one possible offers at once; the walk-away price is where its accept
        probability first reaches a half.
        """
        offers = np.linspace(0.0, 1.0, 41)
        rounds = max(1, int(self.scenario.max_rounds))
        rows = []
        for offer in offers:
            row = [0.0, self.scenario.need_a, offer, 1.0 - offer, 1.0 / rounds, (rounds - 1) / rounds, 1.0]
            if self.cfg.message_vocab:
                row += [0.0] * (2 * self.cfg.message_vocab + 1)
            rows.append(row)
        features = np.array(rows, dtype=np.float32)
        flags = {
            "can_accept": np.ones(len(offers), dtype=bool),
            "can_offer": np.ones(len(offers), dtype=bool),
            "can_talk": np.zeros(len(offers), dtype=bool),
        }
        accept = agent.act(features, flags, capture=True)["accept_prob"]
        if np.ptp(accept) < 0.2:
            # A network that has not learned anything yet accepts everything at about a coin flip. It has no
            # walk-away price, and reporting where noise crosses a half would invent one.
            return None, float(np.interp(LOWBALL, offers, accept))
        crossing = np.flatnonzero(accept >= 0.5)
        walk_away = float(offers[crossing[0]]) if len(crossing) else 1.0
        return walk_away, float(np.interp(LOWBALL, offers, accept))

    def train_start(self, opponent_name, updates=40):
        """Start learning. ``opponent_name="self"`` trains both agents against each other; ``updates`` of 0
        or less means until someone presses Stop."""
        with self.lock:
            if self.training["running"]:
                return
            self.training_run += 1
            run = self.training_run
            self.training = {"running": True, "opponent": opponent_name, "history": [], "started": time.time(),
                             "updates": updates}
            self.bump()
        thread = threading.Thread(target=self._train, args=(run, opponent_name, updates), daemon=True)
        self.trainer = thread
        thread.start()

    def train_stop(self):
        with self.lock:
            self.training["running"] = False
            self.bump()

    def training_current(self, run):
        with self.lock:
            return self.training["running"] and run == self.training_run

    def _train(self, run, opponent_name, updates):
        """PPO on copies of the agents, swapping the weights in after every update."""
        try:
            self._train_loop(run, opponent_name, updates)
        except Exception as error:  # never leave the page believing it is still learning
            traceback.print_exc()
            with self.lock:
                if run == self.training_run:
                    self.training["error"] = f"{type(error).__name__}: {error}"
        finally:
            with self.lock:
                if run == self.training_run:
                    self.training["running"] = False
                    self.bump()

    def _train_loop(self, run, opponent_name, updates):
        self_play = opponent_name == "self"
        learners = []
        for name in AGENTS if self_play else AGENTS[:1]:
            learner = PPOAgent(name, self.cfg, self.device)
            learner.net.load_state_dict(self.agents[name].net.state_dict())
            # Demo pace, not the research setting: a faster learning rate, so a minute on stage shows what a
            # quarter of an hour of ordinary training would. Everything else is unchanged.
            for group in learner.optimizer.param_groups:
                group["lr"] = self.cfg.lr * LEARNING_RATE_BOOST
            learners.append(learner)
        opponent = learners[1] if self_play else (self.policy_for(opponent_name) or self.agents[AGENTS[1]])
        envs = [make_env(self.cfg, self.scenario.protocol, max_rounds=self.scenario.max_rounds) for _ in range(8)]
        pairing = Match((learners[0], opponent))
        collector = RolloutCollector(envs, lambda e: pairing, self.cfg)
        self.progress(run, learners, 0, 0, [], None)  # where it starts, before any learning
        step = 0
        for update in itertools.count(1) if updates <= 0 else range(1, updates + 1):
            if not self.training_current(run):
                break
            batches, episodes, steps = collector.collect(STEPS_PER_UPDATE)
            step += steps
            stats = [learner.update(batches[learner.name], entropy_coef(self.cfg, step)) for learner in learners]
            if not self.progress(run, learners, update, step, episodes, stats[0]):
                break

    def progress(self, run, learners, update, step, episodes, stats):
        """Measure the learner, then hand its fresh weights to the agents the page is using.

        The measuring happens outside the lock, on the learner's copy. The swap only happens if this is still
        the current training run: a reset or a restart while an update was in flight must win.
        """
        walk_away, lowball = self.acceptance(learners[0])
        agreed = [ep for ep in episodes if ep["outcome"] == "agreement"]
        point = {
            "update": update,
            "steps": step,
            "opening_offer": self.opening_offer(learners[0]),
            "walk_away": walk_away,
            "lowball_accept": lowball,
            "reward": float(np.mean([ep["payouts"][0] for ep in episodes])) if episodes else None,
            "reward_b": float(np.mean([ep["payouts"][1] for ep in episodes])) if episodes else None,
            "agreement": len(agreed) / len(episodes) if episodes else None,
            "rounds": float(np.mean([ep["rounds"] for ep in agreed])) if agreed else None,
            "entropy": stats["entropy"] if stats else None,
        }
        with self.lock:
            if run != self.training_run or not self.training["running"]:
                return False
            if update:  # update 0 is the starting point: nothing new to hand over
                for learner in learners:
                    self.agents[learner.name].net.load_state_dict(learner.net.state_dict())
                self.weights_version += 1
                self.weights_at = time.time()
            history = self.training["history"]
            history.append(point)
            if len(history) > HISTORY_POINTS:
                history[:] = history[::2]
            self.bump()
        return True

    def fresh(self):
        """Throw the weights away and start again from random ones that know nothing."""
        with self.lock:
            self.training_run += 1  # retires the current trainer, even halfway through an update
            self.load_weights(fresh=True)
            self.training = {"running": False, "opponent": "", "history": [], "started": None}
            self.network = None
            self.bump()
        if self.live:
            self.train_start("self", 0)

    def reload(self):
        """Put the agents back where this session started, undoing anything learned on stage."""
        with self.lock:
            self.training_run += 1
            self.load_weights(fresh=self.origin is None)
            self.training = {"running": False, "opponent": "", "history": [], "started": None}
            self.network = None
            self.bump()
        if self.live:
            self.train_start("self", 0)

    # ---- state ------------------------------------------------------------------------------------

    def bump(self):
        self.version += 1

    def state(self):
        with self.lock:
            return jsonable({
                "version": self.version,
                "checkpoint": {"path": str(self.checkpoint) if self.checkpoint else None, "sha256": self.fingerprint,
                               "policy": self.cfg.policy, "protocol": self.cfg.protocol,
                               "message_vocab": self.cfg.message_vocab},
                "weights": {"origin": "checkpoint" if self.checkpoint else "scratch", "seed": self.seed,
                            "version": self.weights_version, "age": time.time() - self.weights_at,
                            "live": self.live, "can_reset": self.origin is not None},
                "roster": self.roster(),
                "scenario": asdict(self.scenario),
                "pool": self.env.pool_size if self.env else 100,
                "turn": self.turn if self.waiting_for_human() is not None else None,
                "moves": self.moves,
                "table": self.table,
                "outcome": self.outcome,
                "network": self.network,
                "thinking": self.thinking and {**self.thinking, "elapsed": time.time() - self.thinking["since"]},
                "training": self.training,
                "language": self.language(),
            })

    def command(self, body):
        """Everything the page can ask for, one entry point."""
        command = body.get("command")
        launch = None
        with self.lock:
            if command == "scenario":
                for key, value in body.get("scenario", {}).items():
                    if hasattr(self.scenario, key):
                        setattr(self.scenario, key, value)
                if "llm" in self.seats() and self._debater is None:
                    # Warm the model up now rather than making the first sentence wait for it.
                    threading.Thread(target=lambda: self.debater().load(), daemon=True).start()
                self.bump()
            elif command == "start":
                self.start()
                launch = {"episode": self.episode}
            elif command == "move":
                expected = body.get("turn")  # the turn the page was showing when the visitor clicked
                current = self.turn["id"] if self.turn else None
                if self.waiting_for_human() is not None and current is not None and expected in (None, current):
                    self.turn = None  # taken: a double click must not play the same move twice
                    launch = {"episode": self.episode, "human_action": body.get("move", {})}
            elif command == "reload":
                launch = "reload"
            elif command == "fresh":
                launch = "fresh"
            elif command == "train_start":
                self.train_start(body.get("opponent", "hardball"), int(body.get("updates", 40)))
            elif command == "train_stop":
                self.train_stop()
            else:
                return {"error": f"unknown command {command!r}"}
        if launch in ("reload", "fresh"):  # these may start a trainer, which takes the lock itself
            getattr(self, launch)()
        elif launch is not None:
            self.run(**launch)
        return self.state()


def language_status(model=None):
    """Whether the English-debate layer can run. It needs a model; nothing here invents one."""
    try:
        from language import DEFAULT_MODEL, status  # optional: only present once the layer is configured
    except ImportError:
        return {"available": False, "reason": "not configured"}
    return status(model or DEFAULT_MODEL)


class Handler(BaseHTTPRequestHandler):
    server_version = "ArenaLab/1"

    def do_GET(self):
        url = urlsplit(self.path)
        if url.path == "/":
            self.send(HTTPStatus.OK, "text/html; charset=utf-8", PAGE.read_bytes())
        elif url.path == "/play.js":
            self.send(HTTPStatus.OK, "text/javascript; charset=utf-8", SCRIPT.read_bytes())
        elif url.path == "/api/state":
            self.send(HTTPStatus.OK, "application/json", json.dumps(self.server.lab.state()).encode())
        else:
            self.send(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"not found")

    def do_POST(self):
        if urlsplit(self.path).path != "/api/command":
            return self.send(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"not found")
        length = min(int(self.headers.get("Content-Length") or 0), 1 << 16)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self.send(HTTPStatus.BAD_REQUEST, "text/plain; charset=utf-8", b"expected JSON")
        try:
            result = self.server.lab.command(body if isinstance(body, dict) else {})
        except Exception as error:  # a demo should say what broke, not drop the connection
            traceback.print_exc()
            payload = json.dumps({"error": f"{type(error).__name__}: {error}"}).encode()
            return self.send(HTTPStatus.INTERNAL_SERVER_ERROR, "application/json", payload)
        self.send(HTTPStatus.OK, "application/json", json.dumps(result).encode())

    def send(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in SECURITY_HEADERS.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def newest_checkpoint(runs_dir="runs"):
    candidates = sorted(Path(runs_dir).glob("*/agents.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def main():
    parser = argparse.ArgumentParser(description="Interactive stage: play, watch or train a negotiating agent.")
    parser.add_argument("--checkpoint", help="agents.pt to load (default: the newest under runs/)")
    parser.add_argument("--from-scratch", action="store_true", help="ignore checkpoints: random weights that know nothing")
    parser.add_argument("--live", action="store_true", help="both agents learn by self-play from start-up, while you play them")
    parser.add_argument("--policy", default="mlp", choices=["mlp", "gru"], help="architecture for --from-scratch")
    parser.add_argument("--message-vocab", type=int, default=0, help="cheap-talk vocabulary for --from-scratch")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--language-model", help="Hugging Face id for the English seat (default Qwen/Qwen2.5-0.5B-Instruct)")
    parser.add_argument("--language-threads", type=int, help="CPU threads for the language model (default: half the cores, up to 8)")
    args = parser.parse_args()

    checkpoint = None
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.exists():
            raise SystemExit(f"no such checkpoint: {checkpoint}")
    elif not args.from_scratch:
        checkpoint = newest_checkpoint(args.runs_dir)
        if checkpoint is None:
            print("No checkpoint under runs/: starting from random weights.")
    torch.set_num_threads(1)

    fresh_config = Config(max_rounds=Scenario.max_rounds, policy=args.policy, message_vocab=args.message_vocab)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.lab = Lab(checkpoint, language_model=args.language_model, language_threads=args.language_threads,
                     fresh_config=fresh_config, live=args.live)
    lab = server.lab
    if checkpoint:
        print(f"Agents:    {checkpoint} (sha256 {lab.fingerprint}, policy {lab.cfg.policy})")
    else:
        print(f"Agents:    fresh random weights (seed {lab.seed}, policy {lab.cfg.policy})")
    if args.live:
        lab.train_start("self", 0)
        print("Learning:  live, by self-play, until you stop it on the page")
    print(f"Arena Lab: http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
