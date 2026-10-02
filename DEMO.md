# Demo script

An eight-minute walkthrough for a live audience. Everything here runs on a laptop, offline, from a cold
start, and nothing is pre-run: one command starts a population training from random weights, its dashboard,
and the Arena Lab, whose two agents also start from random weights and learn while the judges watch.

## Before they arrive

```bash
.venv\Scripts\activate
python train.py --generations 500 --run-id showcase          # ~3 min, gives you a finished run to fall back on
```

Once per machine, fetch the English model so nothing downloads in front of the judges:

```bash
python -c "from language import Debater; Debater().load()"
```

Close other heavy applications: the demo trains two things live on CPU.

## 1. The one-liner (30 seconds)

```bash
python demo.py --pace 0.35
```

Two tabs open: the dashboard (port 8000) and the Arena Lab (port 8100). Start on the dashboard. The line:
*"This is not a replay. It is training right now, and everything on screen comes from a log the trainer is
writing as it goes."*

Before you get to section 4, seat the language model in the Lab once and start a negotiation, so the model is
in memory (the first turn of a session takes 10–20 seconds longer).

## 2. The arena (90 seconds)

Point at, in this order:

- **The ring.** Every circle is a living agent, sized by its token balance. Each round they pair off; the
  lines between partners flash green for a deal and red for a collapse.
- **The pair centre stage.** One is buying compute, the other selling. Their needs, balances and compute are
  under them. Offers land on the table as a card with a split bar; a deal fires the banner with the price and
  sends coins from buyer to seller.
- **Births and deaths.** A green circle pops out of its parent; a red one fades out. *"Nothing here is
  scripted: an agent that keeps losing negotiations runs out of tokens and dies, and its line ends."*

## 3. The network (60 seconds)

Move to the right-hand panel, which updates on every move:

- **What it sees** — the actual input features, including its own need.
- **Two hidden layers, 64 units each**, blue for positive activation, red for negative.
- **What it decides** — accept probability, and the Beta distribution over the offer it would make, with the
  sampled value marked.

The line: *"This is the policy network deciding, in real time, what to offer the agent opposite."*

## 4. Hand them the controls (3 minutes)

Switch to the Arena Lab tab. The line: *"The dashboard proves it trains. This proves it thinks: you choose
what happens next."*

- **It learned this since you sat down.** Point at the header: *fresh random weights, seed …, weights update
  N, 0.5 s ago* — the number climbs about once a second. Press *Start from scratch*: the lowball chart jumps
  back to 50% (a fresh network accepts anything at a coin flip) and falls to 0% within about ten seconds, while
  the opening demand overshoots and settles. *"That is PPO, from random weights, while you watched."*

- **A judge plays it.** Seat A *you*, seat B the trained agent. Let the judge pick both sides' needs and the
  deadline, then make an offer. The network panel on the right fills in on the agent's reply: the features it
  saw, both hidden layers, its accept probability, the offer distribution. *"Lowball it and watch the accept
  probability drop."* Each of its moves is stamped *update N*, and the stamps change between rounds: it is
  still learning while the judge plays it.
- **A judge picks the topic.** Seat A *language model*, seat B the trained agent. Ask a judge for any
  dispute and type it into *Topic*. While the status reads *Language model is thinking*, say: *"That is a
  language model running on this laptop. Unplug the network if you like."* Each sentence lands with a strip
  underneath showing how strongly it wanted each price, and the trained agent answers with a real
  counter-offer. Be precise about what it is: the language model is used off the shelf; the negotiating agents
  are what we trained.
- **It adapts to a bully.** In *Learning live*, press *Stop*, choose *scripted hardball* and press *Train*.
  Within a minute its opening demand falls (about 63% to 37% in testing) as it learns that this opponent never
  gives ground. Play it again: it bargains differently now. *Start from scratch* resumes self-play from zero.

## 5. The result that surprises people (90 seconds)

Open `results/deadline_8seed/plots.png` (or quote the table in REPORT.md §5).

*"We asked whether giving agents memory helps them bargain. With the clock visible it makes no measurable
difference across five seeds. So we hid the clock and shortened the deadline — and the recurrent population
learned brinkmanship. Twenty-eight per cent of their deals now go right to the deadline, the winner takes
64%, and inequality is eight times higher. Memory bought them the ability to count rounds nobody showed them,
and they spent it holding out."*

If asked whether that is just noise: eight seeds per arm, permutation test, p = 0.0002.

## 6. The honest part (30 seconds)

*"Three of our four headline questions came back negative. The communication channel we added stays empty —
and that is what cheap-talk theory predicts when two sides want opposite things. We measured it three ways
and then scrambled every message to prove the listeners were ignoring them."*

## Questions you should expect

- **"Is the dashboard just a recording?"** No. Run `type runs\<run>\live.jsonl` (or `tail`) beside it and
  watch the file grow as the arena moves.
- **"Is the English scripted?"** No. Each sentence is sampled from a local model at temperature 0.8, so the
  same topic never plays out the same way twice, and the strip under each sentence is the model's own
  log-probability for each price. The page header carries the first 16 characters of the checkpoint's SHA-256;
  compare them with
  `certutil -hashfile runs\<run>\agents.pt SHA256`.
- **"Did you train the language model?"** No, and that is deliberate: it is there to make the negotiation
  readable and to test the trained agent against an opponent it never met. The deep learning is the agents.
- **"Were the Lab's agents trained before the demo?"** No. They started from random weights when `demo.py`
  launched; the seed is in the header, and *Start from scratch* repeats it on demand.
- **"Could the dashboard be steering the run?"** It serves two read-only endpoints and never writes. The
  training process does not read anything the dashboard produces.
- **"Where is the deep learning?"** Two architectures (MLP and GRU) behind one interface, PPO written from
  scratch with a hybrid action space, and a recurrent update that replays whole episodes through the GRU.
- **"Why should I believe any single number?"** Every comparison in the report is 5–8 independent seeds with
  a permutation test, and `results/` holds the per-seed data.

## If something breaks

- Port in use: `python demo.py --port 8010`.
- No browser: open `http://127.0.0.1:8000/` by hand, or run with `--no-browser`.
- Training too slow on the projector laptop: `python demo.py --pace 0.6 --rounds-per-generation 3`.
- Nothing appears: fall back to the finished run with `python dashboard.py runs/showcase`.
- Arena Lab port in use: `python demo.py --lab-port 8101`. Dashboard port in use: `--port 8010`.
- The laptop struggles with both trainers at once: `python demo.py --no-lab` for the population, and
  `python play.py --live --from-scratch` in another terminal only when you reach section 4.
- English turns too slow: `python play.py --language-threads 8`, or seat a trained agent against a scripted
  one instead. Everything except the English seat works without the language model.
