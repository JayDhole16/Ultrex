# Learning to Bargain

Two agents have to divide 100 tokens. They can take turns making offers, or bid in secret. They can be given
a channel to talk over that cannot affect payoffs. They can be dropped into a population where the deals they
strike decide whether they eat, die or have children. This repository builds that world, trains agents in it
with PPO, and measures what they learn.

The findings are in [REPORT.md](REPORT.md); the archived numbers behind them are in `results/`. For a live
demo, see [DEMO.md](DEMO.md).

## Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
pytest negotiation_env.py       # 7 tests, ~2 seconds
```

Everything runs on CPU and needs no network after install. Use the venv: a global Python with torch built
against NumPy 1.x will crash on import. The one exception to "no network" is the optional English seat in the
Arena Lab: the first time a language model is seated it downloads Qwen2.5-0.5B-Instruct (about 1 GB) to the
Hugging Face cache, and runs offline from then on.

## Run everything, live

```bash
python demo.py
```

One command, three processes, two browser tabs, nothing pre-run:

- **The population** (`train.py`) starts from random weights and evolves: survival costs, trades, births,
  deaths.
- **The dashboard** (http://127.0.0.1:8000/) animates it as it trains: agents as circles sized by their token
  balance, the negotiating pair centre stage exchanging offers and messages, coins moving on a deal, births and
  deaths, and the acting agent's network with its activations, accept probability and offer distribution.
- **The Arena Lab** (http://127.0.0.1:8100/) starts two agents from random weights that begin learning by
  self-play the moment it launches, and never stop while it runs. Within about ten seconds they learn to turn
  down a lowball (the chance of accepting 20 of 100 falls from 50% to 0%); their opening demand overshoots and
  then settles near an even split. You can play them, pit them against other negotiators, or press *Start from
  scratch* to watch it happen again.

Ctrl+C stops everything. `--lab-checkpoint runs/<run>/agents.pt` starts the Lab from trained agents instead
(they keep learning), `--no-lab` leaves it out, and `--port` / `--lab-port` move the two pages.

## The Arena Lab on its own

```bash
python play.py --live --from-scratch   # random weights, learning by self-play from the start
python play.py                         # the newest checkpoint under runs/, learning only when you ask
python play.py --checkpoint runs/<run>/agents.pt --live
```

With no checkpoint on disk it starts from random weights either way. Set each side's need, the deadline, the
cost of delay, the protocol and whether the clock is visible, then:

- **Play** — take a seat yourself and bargain with the trained network. Every one of its moves shows the
  features it saw, both hidden layers, its accept probability and the Beta it drew its offer from.
- **Watch** — seat any two of: a trained agent, a scripted negotiator (rubinstein, conceder, hardball,
  random) or the language model, and follow the duel move by move.
- **Learn live** — with `--live` both agents train against each other (self-play) continuously, and every
  update hands new weights to the agent across the table, even mid-negotiation; each of its moves is stamped
  with the update that made it. Or press *Stop*, pick an opponent such as hardball and train one agent
  against it. The card charts reward, the chance it takes a lowball, what it opens with and the least it will
  accept. *Start from scratch* throws the weights away; *Reset to checkpoint* goes back to the file.
- **Debate in English** — type a topic ("who gets the bigger office") and seat the language model. It argues
  in one sentence per turn, then picks its price by scoring a menu of demands (40–90) with its own
  log-probabilities, and the page shows those scores under each sentence. The number, not the sentence, is
  what the environment receives.

A language-model turn takes 3–5 seconds on a laptop CPU; the page shows it thinking. `--language-model
<hf id>` swaps in a bigger model and `--language-threads N` sets its cores. The page reports the checkpoint's
SHA-256, and nothing on it is precomputed. The Lab writes no files, and it is separate from `dashboard.py`,
which stays read-only.

## The five things to run

| Command | Time | What it does |
|---|---|---|
| `python ippo.py` | ~5 min | Two independent PPO learners, fixed pairing |
| `python self_play.py` | ~5 min | Self-play against a pool of past checkpoints |
| `python train.py --generations 500` | ~3 min | The population economy, unattended |
| `python tournament.py runs/<run>/agents.pt` | ~2 min | A checkpoint against scripted negotiators |
| `python experiments.py --preset policy --seeds 5` | ~15 min | A multi-seed comparison with a permutation test |

Useful flags: `--protocol sealed_bid`, `--message-vocab 5` (cheap talk), `--policy gru` (recurrent),
`--observe-round 0` (hide the clock), `--checkpoint <path>` (evaluate instead of train). Every script takes
`--help`, and every flag is a field on a config dataclass, so the help text is always current.

## What is where

| File | Role |
|---|---|
| `negotiation_env.py` | The environment: protocols, cheap talk, observability, and its own tests |
| `networks.py` | The two policy trunks, MLP and GRU, behind one interface |
| `ippo.py` | PPO, the rollout collector, evaluation, the shared library the rest imports |
| `self_play.py` | Opponent pool, snapshots, matchmaking, plots |
| `economy.py` | The population: compute, trade, work, upkeep, death, reproduction |
| `train.py` | The unattended entry point: JSONL event log, checkpoints, plots |
| `livelog.py` | The move-by-move stream the dashboard animates |
| `dashboard.py` + `.html` + `.js` | Read-only live dashboard: arena, network panel, charts |
| `demo.py` | Starts a run and its dashboard together |
| `play.py` + `.html` + `.js` | The Arena Lab: play, watch, train on stage, debate in English |
| `language.py` | The optional English seat: a local instruct model whose argument becomes a real offer |
| `baselines.py`, `tournament.py` | Scripted negotiators and a round robin |
| `experiments.py` | Multi-seed runs, aggregation, permutation tests, comparison plots |
| `transcripts.py` | Episode transcripts and the cheap-talk analysis |

## Results at a glance

- Agents rediscover the **Rubinstein split** (52/48 against a predicted 51.3/48.7) without being told the game
  has a solution, and beat random and scripted opponents while conceding slightly to the theory-optimal agent.
- A **cheap-talk channel stays empty**: message entropy collapses, η² between messages and later offers is
  below 0.011, and scrambling every message changes nothing. That is the babbling equilibrium theory predicts
  when interests are opposed.
- **Memory buys nothing** while the game is Markov (p = 0.76 across 5 seeds), but when the clock is hidden and
  the deadline bites, a recurrent population learns **brinkmanship**: 28% of deals go to the deadline, splits
  become eight times more unequal, and joint payoffs fall (p = 0.0002 across 8 seeds).
- In the **population economy**, agents self-regulate at about 23 alive against a fixed compute supply, deals
  get faster over 500 generations, and generosity drifts down about 4 points — a drift too small to call
  selection at this population size.

## Reading a run

Each run writes to `runs/<run_id>/`: `metrics.csv` (per update), `episodes.csv` (per episode), `agents.pt`
(weights and config), `eval.json`, `plots.png`. The economy also writes `events.jsonl` (one JSON line per
generation, including a full sampled negotiation) and `checkpoints/` every 25 generations. The dashboard only
ever reads those files; it cannot write to a run or influence training.
#   U l t r e x  
 #   U l t r e x  
 