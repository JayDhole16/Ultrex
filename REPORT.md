# Learning to Bargain: Emergent Negotiation, Communication and Selection in a Multi-Agent Economy

A study of what reinforcement-learning agents do when they have to divide something, talk about it, and
live with the result. The work builds a negotiation environment, trains agents in it with PPO, and then
asks four questions that the environment is designed to answer: does learning find the game-theoretic
solution, does a free communication channel become meaningful, does memory help, and what happens when
bargaining outcomes decide who survives and reproduces.

All results below come from the code in this repository. Every number is reproducible with the command
quoted beside it; every comparison between two conditions uses several seeds and a permutation test.

---

## 1. What was built

**The environment** (`negotiation_env.py`) is a PettingZoo `ParallelEnv` in which two agents divide a pool
of 100 tokens. Each agent draws a private valuation multiplier v ~ U(0.5, 2.0) each episode. A deal giving
agent *i* x tokens pays `discount**delay * v_i * x_i`, so delay is costly. It passes PettingZoo's own API
and seeding tests, and ships with 7 tests of its own.

Three switches make the environment a family of games rather than one game:

| Switch | Effect |
|---|---|
| `protocol` | `alternating_offers` (Rubinstein-style turn taking) or `sealed_bid` (a Nash demand game) |
| `message_vocab` | adds a cheap-talk channel: before every move both agents send a token that the other sees and that cannot affect payoffs |
| `observe_round` | when false, the round counter is reported as zero while the deadline still bites, making the game partially observable |

**The learners.** `ippo.py` implements PPO from scratch: GAE, ratio clipping, annealed entropy bonus, and a
hybrid action space (a Categorical accept/reject, a Beta over the share to demand, and a Categorical message).
Only the parts of an action the environment actually uses enter the log-probability, which is what keeps one
learner protocol-agnostic. `self_play.py` adds a league: every agent periodically snapshots itself into a
shared opponent pool and trains against a mix of its current self and past checkpoints. `networks.py` offers
two trunks — a two-layer MLP and a GRU that carries hidden state along the agent's own decisions — behind one
interface, so the architecture can be swapped with a flag.

**The economy** (`economy.py`, run by `train.py`) puts the negotiation inside a population. Agents hold token
balances that decay, earn tokens from a task whose output has diminishing returns in a second scarce resource
(compute), and must trade compute with a randomly paired partner each round, negotiating over the gains from
trade. Zero balance means death; a surplus buys a child whose policy is the parent's plus Gaussian noise.

**The instruments.** `tournament.py` plays any checkpoint against scripted negotiators (`baselines.py`),
`experiments.py` runs multi-seed comparisons with permutation tests, `transcripts.py` inspects what agents
say, and `dashboard.py` renders a run live while it trains. `play.py`, the Arena Lab, is the interactive
counterpart: a person can bargain against a checkpoint, set up duels between any two policies, fine-tune an
agent against a chosen opponent on stage, or seat a local language model (`language.py`) that argues a
visitor's topic in English and turns each argument into a real offer.

---

## 2. Does learning find the game-theoretic solution?

**Yes, closely.** Training two independent PPO agents under alternating offers for 1M environment steps
converges to an opening offer that is accepted immediately.

| | Learned | Rubinstein prediction |
|---|---|---|
| Proposer's share | 52 / 48 | 51.3 / 48.7 |
| Rounds to agreement | 1.00 (greedy) | 1 |
| Agreement rate | 100% | 100% |

The subgame-perfect share for the first proposer with discount δ is 1/(1+δ) = 51.3% at δ = 0.95. The agents
find it without being told the game has a solution. `python ippo.py`

A tournament against scripted opponents shows the learned policy is strong but not optimal
(`python tournament.py runs/<run>/agents.pt`, 300 episodes per matchup):

| Policy | Mean payoff | Mean token share | Agreed |
|---|---|---|---|
| rubinstein (theory) | 52.5 | 49.6% | 100% |
| **learned (self-play)** | **50.9 / 50.7** | **48.4% / 48.2%** | **92.8%** |
| random | 39.0 | 32.5% | 100% |
| hardball | 34.2 | 53.7% | 86.4% |
| conceder | 33.9 | 53.0% | 99.1% |

The learned agents take 63% of the pool off a random opponent, so they exploit weakness rather than merely
splitting evenly, but they concede about 1.4 points of share to the theory-optimal agent.

**Protocol transfer fails.** Agents trained under alternating offers agree only 29.8% of the time under
sealed bids (0% when played greedily): each has learned to ask for slightly over half, which is fine when a
partner can accept but fatal when both demand at once. Agents trained under sealed bids reach 100% greedy
agreement by demanding ≈ 44 tokens each and leaving ≈ 12 on the table — a learned safety margin.

---

## 3. Does self-play with an opponent pool change what is learned?

Training with a pool (snapshots every 10 updates, half the episodes against the current self) reaches the
same convention, but the path is informative. `python self_play.py`

| | First 10% | 10–50% | Last 25% |
|---|---|---|---|
| Reward per agent | 56.3 | 59.8 | 60.9 |
| Gini of the split | 0.098 | 0.033 | 0.021 |
| Rounds to agreement | 3.29 | 2.09 | 1.62 |

Early training passes through a haggling phase: agents learn to demand more, deals take over three rounds,
and the discount destroys the surplus they are fighting over. They then learn that delay is what costs them,
and the split converges to near-equal. The two learners never meet during training — they only see each
other's snapshots — yet when finally paired they agree 100% of the time at 52/48.

---

## 4. Does a free communication channel become meaningful?

**No, and the theory says it should not.** With `--message-vocab 5`, both agents send a token before every
move. Three measurements agree that the channel stays empty:

| Measure | Alternating offers | Sealed bids |
|---|---|---|
| Message entropy (max 2.32 bits) | 0.84 / 0.09 | 1.93 / 1.93 |
| Policy entropy at the same steps | 0.21 / 0.09 | 2.00 / 1.91 |
| η² message → next offer's generosity | ≤ 0.011 | ≤ 0.004 |
| η² message → sender's valuation | ≤ 0.005 | ≤ 0.003 |
| Effect of scrambling every message | agreement unchanged, deals 1.57 → 1.69 rounds | agreement 92.9% → 92.0% |

Under alternating offers the vocabulary collapses: one agent sends the same token 99% of the time. Under
sealed bids the tokens stay uniform but carry nothing — marginal entropy and policy entropy match, which
means the token is chosen by chance rather than by the situation. Scrambling every message before delivery,
the causal test that correlation cannot give, changes nothing that matters.

This is the babbling equilibrium of cheap-talk theory. With fully opposed interests over a fixed pool and a
valuation multiplier that scales payoffs without changing preferences, no agent has anything to gain from
being understood. Cheap talk also failed to improve coordination in the sealed-bid game, where it might have
helped: 91.2% agreement with the channel against 91.0% without, at matched bargaining steps.

---

## 5. Does memory help?

This question needed three experiments, and the third one produced the most interesting result in the project.
Each arm is an independent training run per seed; comparisons use a two-sided permutation test.

**Experiment 1 — the clock is visible** (`--preset policy`, 5 seeds, 400k steps):

| Metric | MLP | GRU | p |
|---|---|---|---|
| Reward per agent | 60.32 ± 0.10 | 60.35 ± 0.11 | 0.76 |
| Rounds to agreement | 1.898 ± 0.003 | 1.901 ± 0.013 | 0.64 |
| Gini | 0.0310 ± 0.0004 | 0.0320 ± 0.0004 | **0.007** |

Recurrence buys nothing, which is what the structure of the game predicts: the standing offer plus the round
number is a sufficient statistic, so there is nothing for a memory to add. The one "significant" result is a
Gini difference of 0.001 — detectable only because between-seed variance is tiny, and far too small to matter.
It is a useful reminder that statistical significance is not importance.

**Experiment 2 — hide the clock** (`--preset memory`, 5 seeds). Every metric moves in the GRU's favour and
none significantly (reward p = 0.21, rounds p = 0.15). The manipulation is too weak: deals close in about two
rounds, so a 20-round deadline almost never binds and the hidden clock costs almost nothing.

**Experiment 3 — hide the clock and make the deadline bite** (`--preset deadline`, 4 rounds, 8 seeds):

| Metric | MLP | GRU | p |
|---|---|---|---|
| Reward per agent | 60.54 ± 0.12 | 58.30 ± 0.26 | 0.0002 |
| Rounds to agreement | 1.547 ± 0.017 | 2.373 ± 0.053 | 0.0002 |
| Gini of the split | 0.038 ± 0.000 | **0.312 ± 0.017** | 0.0002 |
| Agreement rate | 98.8% | 98.0% | 0.0002 |

Recurrence changes the equilibrium the population settles into, and lowers joint welfare. The mechanism is
visible in when deals close:

| Deal closes in | MLP share of deals | MLP Gini | GRU share of deals | GRU Gini |
|---|---|---|---|---|
| Round 1 | 58.4% | 0.038 | 26.0% | 0.267 |
| Round 2 | 30.4% | 0.039 | 43.8% | 0.303 |
| Round 3 | 8.5% | 0.038 | 2.2% | 0.328 |
| Round 4 (deadline) | 2.7% | 0.039 | **28.0%** | **0.364** |

The memoryless population splits evenly whenever a deal closes and settles early. The recurrent population
holds out: 28% of its deals go all the way to the deadline, and those are the most lopsided (the winner takes
64%). Both learners earn the same on average (reward gap 0.46), so this is not one agent dominating: it is a
role-based convention in which whoever is favoured in a given episode extracts most of the surplus, and the
randomised first mover shares the roles out evenly.

So memory did buy a capability — tracking a deadline the agents cannot see — and the agents spent it on
brinkmanship. Individually rational, collectively expensive: 3.7% lower payoffs and eight times the inequality.
The same architecture played head to head against the MLP splits tokens 49.8 / 50.2, so the difference is not
that one beats the other; it is that a population of recurrent agents settles somewhere else.

---

## 6. What happens when bargaining decides who survives?

The economy runs 500 generations (10,000 rounds) from 8 founding agents. `python train.py --generations 500`

| | Gen 1–10 | Gen 11–100 | Gen 101–200 | Gen 201–300 |
|---|---|---|---|---|
| Population | 21.6 | 23.0 | 23.0 | 23.4 |
| Mean generosity | 51.0% | 49.9% | 49.7% | 46.8% |
| Wealth Gini | 0.23 | 0.25 | 0.24 | 0.24 |
| Rounds to agreement | 1.97 | 1.92 | 1.83 | 1.72 |
| Lineage depth | 1.1 | 5.5 | 11.8 | 22.1 |

The population self-regulates at about 23 agents against a fixed compute supply, with roughly two births and
two deaths per generation and lineages 46 deep by the end. Deals get faster, which selection should favour
because haggling burns compute. Generosity drifts down about 4 points over the run.

That last number should not be over-read, and the environment explains why: an agreed trade moves about 0.37
tokens while an agent pays 1 token per round to survive and earns a noisy task income. Selection on bargaining
skill is therefore weak next to luck, and with a population of 23 a 4-point drift is within what neutral drift
could produce. Establishing it as selection would need either bigger stakes (a lower compute elasticity) or a
neutral control in which reproduction ignores wealth.

---

## 7. What this adds up to

Three of the four questions returned negative or null results, and that is the point. Each one is a
*measured* negative with a mechanism behind it:

- Cheap talk stays empty **because** interests are opposed and the private valuation is strategically inert.
  The environment was built so that this is testable, and three independent measurements plus a causal
  ablation agree.
- Memory buys nothing **because** the observable state is a sufficient statistic — and when that is made
  false by hiding the clock, memory immediately buys something specific and measurable.
- Selection on generosity is weak **because** the stakes of a single negotiation are small next to the noise
  in the rest of the economy.

The positive results are that independent learners rediscover the Rubinstein split without being told the
game has a solution, that self-play through an opponent pool converges to an equal, fast convention after
passing through a costly haggling phase, and that a deadline no one can see turns a recurrent population
into brinkmen.

## 8. Limitations

- **Single environment family.** Every result is about one bargaining game and its two protocols.
- **The economy's task is a stub.** Task scores are random; a real task (code review) would let skill, not
  luck, drive earnings and would sharpen selection.
- **Valuations do not change preferences.** They scale payoffs, which is why the private information is
  strategically inert. Adding outside options that depend on valuation would give cheap talk something to
  signal and is the single most promising change to the environment.
- **Economy results are one seed.** The training comparisons use 5–8 seeds each, but the 500-generation
  economy runs have not been repeated.
- **Compute.** Everything ran on 16 CPU cores. A GRU run costs about 4× an MLP run for the outcomes above.
- **The English seat is zero-shot.** The Arena Lab's language model is a 0.5B instruct model used as it is,
  not trained with the agents. Its price comes from scoring a fixed menu of demands (40–90) with its own
  log-probabilities. Over 30 test turns it anchored on the offer in front of it and asked a notch more, and a
  model this small can sound warmer or harsher than the number it picks. It makes the negotiation legible and
  gives the trained agent an opponent it never saw in training; it is not evidence about emergent language.

## 9. Reproducing

```bash
pip install -r requirements.txt
pytest negotiation_env.py                         # 7 environment tests
python ippo.py                                    # section 2, ~5 min
python self_play.py                               # section 3, ~5 min
python self_play.py --message-vocab 5             # section 4
python transcripts.py runs/<run>/agents.pt        # section 4, message analysis + scramble test
python experiments.py --preset policy --seeds 5   # section 5, experiment 1
python experiments.py --preset memory --seeds 5   # section 5, experiment 2
python experiments.py --preset deadline --seeds 8 # section 5, experiment 3
python tournament.py runs/<run>/agents.pt         # section 2, baselines
python train.py --generations 500                 # section 6
python demo.py                                    # everything live: population, dashboard, Arena Lab
python play.py --live --from-scratch              # the Arena Lab alone, learning by self-play from zero
```

Archived metrics, per-seed configs, comparison tables and figures for every experiment quoted here are in
`results/`. Each experiment folder holds `comparison.txt` (the table with its permutation tests),
`summary.csv` (per-run end-of-training numbers) and `plots.png` (learning curves, mean ± 1 sd across seeds).

## 10. References

- Rubinstein, A. (1982). Perfect equilibrium in a bargaining model. *Econometrica*, 50(1), 97–109.
- Nash, J. (1953). Two-person cooperative games. *Econometrica*, 21(1), 128–140.
- Crawford, V. & Sobel, J. (1982). Strategic information transmission. *Econometrica*, 50(6), 1431–1451.
- Schulman, J. et al. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.
- de Witt, C. S. et al. (2020). Is independent learning all you need in the StarCraft multi-agent challenge?
  arXiv:2011.09533.
- Lowe, R. et al. (2019). On the pitfalls of measuring emergent communication. AAMAS.
- Terry, J. et al. (2021). PettingZoo: gym for multi-agent reinforcement learning. NeurIPS.
