"""Unattended training entry point for the population economy (economy.py).

    python train.py --generations 500

Runs the evolving population for the requested number of generations with no interaction: it never reads
stdin, never opens a window and needs no network, so it can run in a headless container. Every setting
is a --flag (see --help); nothing is asked at runtime. Everything is written under runs/<run_id>/:

    config.json      the full configuration
    events.jsonl     one JSON object per generation, appended and flushed as each generation ends
    checkpoints/     generation_NNNN.pt every checkpoint_every generations and at the end of the run:
                     every living agent's policy weights, balance and lineage, written atomically
    generations.csv  the per-generation metrics as a table
    plots.png        population, generosity, inequality and trading over generations

Each events.jsonl line holds:
    generation, round, time          generation number, last economy round played, UTC timestamp
    population                       living agents at the end of the generation
    token_balances                   [{agent_id, tokens}] for every living agent
    mean_generosity                  each agent's mean offer to its partner as a share of the negotiated
                                     surplus, averaged over the agents that made offers
    gini                             Gini coefficient of the token balances
    births, deaths, agreement_rate, rounds_to_agreement
    negotiation                      one negotiation from the generation, drawn at random: its agent_ids
                                     (seats agent_0 and agent_1), valuations, every step (messages when
                                     cheap talk is on, offers as the share offered to the partner, accepts
                                     and rejects), outcome, split, surplus, price, and the env's rendered
                                     transcript; null if the generation had no negotiations
Values that don't exist, such as generosity once the population has died out, are null.

The run ends early, still writing its final checkpoint, if the population dies out. An existing run is
never overwritten: reusing a --run-id that already has an events.jsonl exits with an error.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from economy import Economy, EconomyConfig, RandomScoreTask, generation_row, negotiated, plot_economy
from ippo import CsvLog, load_agents, parse_config
from livelog import LiveLog, NullLog


@dataclass
class TrainConfig(EconomyConfig):
    run_id: str = ""  # default: economy_<timestamp>
    runs_dir: str = "runs"
    checkpoint_every: int = 25  # generations between policy checkpoints; the run's last generation is always saved
    plot_every: int = 10  # generations between refreshes of plots.png; 0 = only at the end
    live_log: int = 0  # 1 also writes live.jsonl, the move-by-move stream the demo dashboard animates
    live_pace: float = 0.0  # seconds the run waits between spotlight moves, so a person can follow along


def jsonable(value):
    """value with NaN and infinities as None and numpy scalars as plain Python numbers, ready for strict JSON."""
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def negotiation_record(trade):
    """One negotiation as logged: who took part, every step each of them took, and how it ended."""
    ids = trade["ids"]
    steps = []
    for decision in trade["decisions"]:
        step = {"negotiation_round": decision["round"], "agent_id": ids[decision["seat"]], "action": decision["kind"]}
        if decision["kind"] == "talk":
            step["message"] = decision["message"]
        elif decision["kind"] == "offer":
            step["offer_to_partner"] = round(decision["generosity"], 4)  # share of the surplus
        steps.append(step)
    return {
        "economy_round": trade["round"],
        "agent_ids": list(ids),
        "valuations": [round(float(valuation), 4) for valuation in trade["valuations"]],
        "steps": steps,
        "outcome": trade["outcome"],
        "negotiation_rounds": trade["rounds"],
        "split": trade["tokens"],  # each seat's share of the surplus, in units of the env's pool of 100
        "surplus": round(trade["surplus"], 4),
        "price": round(trade["price"], 4),
        "transcript": trade["transcript"],
    }


def generation_event(row, economy, outcome, rng):
    """The events.jsonl object for one generation; rng picks which negotiation to include."""
    trades = negotiated(outcome["trades"])
    sample = trades[rng.integers(len(trades))] if trades else None
    return jsonable(
        {
            "generation": row["generation"],
            "round": row["round"],
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "population": row["population"],
            "token_balances": [{"agent_id": agent.id, "tokens": round(agent.tokens, 4)} for agent in economy.agents],
            "mean_generosity": row["mean_generosity"],
            "gini": row["wealth_gini"],
            "births": row["births"],
            "deaths": row["deaths"],
            "agreement_rate": row["agreement_rate"],
            "rounds_to_agreement": row["rounds_to_agreement"],
            "negotiation": negotiation_record(sample) if sample else None,
        }
    )


def save_checkpoint(path, cfg, economy, generation):
    """Every living agent's policy weights, balance and lineage. Written to a temporary file and then renamed,
    so an interrupted run never leaves a partial checkpoint behind."""
    agents = [
        {
            "id": agent.id,
            "parent": agent.parent,
            "lineage": agent.lineage,
            "born": agent.born,
            "tokens": float(agent.tokens),  # a numpy scalar would make torch.load(weights_only=True) refuse the file
            "state_dict": agent.policy.net.state_dict(),
        }
        for agent in economy.agents
    ]
    state = {"config": asdict(cfg), "generation": generation, "round": economy.round, "agents": agents}
    partial = path.with_suffix(".tmp")
    torch.save(state, partial)
    os.replace(partial, path)


def create_run_dir(cfg):
    run_dir = Path(cfg.runs_dir) / cfg.run_id
    if (run_dir / "events.jsonl").exists():
        raise SystemExit(f"{run_dir} already holds a run; pass a different --run-id")
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    return run_dir


def train(cfg):
    torch.set_num_threads(1)  # thread overhead outweighs any speedup for 64-unit MLPs
    parents = None
    if cfg.init:
        trained, agents = load_agents(cfg.init)
        # The founders have to be the same shape of network as the checkpoint they are copied from.
        cfg.message_vocab, cfg.hidden_size, cfg.policy = trained.message_vocab, trained.hidden_size, trained.policy
        parents = list(agents.values())
    cfg.run_id = cfg.run_id or f"economy_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = create_run_dir(cfg)

    torch.manual_seed(cfg.seed)
    live = LiveLog(run_dir / "live.jsonl", cfg.live_pace) if cfg.live_log else NullLog()
    economy = Economy(cfg, RandomScoreTask(), parents, live=live)
    live.emit("run_start", run=cfg.run_id, config=asdict(cfg), agents=[agent.id for agent in economy.agents])
    sample_rng = np.random.default_rng([cfg.seed, 1])  # its own stream, so logging never changes the simulation
    print(f"run {cfg.run_id}: {cfg.generations} generations, writing to {run_dir}", flush=True)

    with CsvLog(run_dir / "generations.csv") as table, open(run_dir / "events.jsonl", "a", encoding="utf-8") as events:
        for generation in range(1, cfg.generations + 1):
            outcome = economy.play_generation()
            row = generation_row(generation, economy, outcome)
            table.write([row])
            event = generation_event(row, economy, outcome, sample_rng)
            events.write(json.dumps(event, allow_nan=False) + "\n")
            events.flush()
            live.emit("generation_end", **event)
            print(
                f"generation {generation:4d} | round {economy.round:6d} | population {row['population']:3d}"
                f" (+{row['births']} / -{row['deaths']}) | generosity {row['mean_generosity']:6.1%}"
                f" | wealth gini {row['wealth_gini']:.3f} | agreement {row['agreement_rate']:6.1%}"
                f" in {row['rounds_to_agreement']:.2f} rounds | lineage depth {row['mean_lineage']:.1f}",
                flush=True,
            )

            extinct = not economy.agents
            last = extinct or generation == cfg.generations
            if last or (cfg.checkpoint_every and generation % cfg.checkpoint_every == 0):
                save_checkpoint(run_dir / "checkpoints" / f"generation_{generation:04d}.pt", cfg, economy, generation)
            if cfg.plot_every and generation % cfg.plot_every == 0:
                plot_economy(run_dir)
            if extinct:
                print(f"The population died out in generation {generation}.", flush=True)
                break

    live.emit("run_end", generations=generation, round=economy.round, population=len(economy.agents))
    live.close()
    plot_economy(run_dir)
    print(f"Done: {run_dir}", flush=True)


if __name__ == "__main__":
    train(parse_config(TrainConfig, "Run the population economy unattended, logging every generation."))
