"""Run one experiment across several seeds and compare the arms honestly.

    python experiments.py --preset policy                 # mlp vs gru, 5 seeds each
    python experiments.py --preset talk --seeds 3
    python experiments.py --arms "mlp=--policy mlp" "gru=--policy gru" --total-steps 400000

Every (arm, seed) is a separate training run, launched as its own process; with 16 cores they run in
parallel. One seed proves nothing, so nothing here reports a single run: each arm gets the mean and spread
across its seeds, and the two-arm comparison gets a permutation test, which makes no assumption about the
shape of the distribution and is exact enough at five seeds a side (the smallest p-value it can report with
5 v 5 is 1/252 ≈ 0.004).

Written to runs/<experiment>/:
    <arm>_seed<k>/     one full training run, exactly as if you had launched it by hand
    summary.csv        per run: the end-of-training metrics and the post-training evaluation
    comparison.txt     the table this prints, with the permutation tests
    plots.png          learning curves per arm: mean across seeds, with a band for the spread
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from self_play import INK_MUTED, SERIES_COLORS, STYLE, finish_panel, read_metrics

HERE = Path(__file__).parent
PRESETS = {
    "policy": {"mlp": ["--policy", "mlp"], "gru": ["--policy", "gru"]},
    "talk": {"silent": ["--message-vocab", "0"], "cheap_talk": ["--message-vocab", "5"]},
    "protocol": {"alternating": ["--protocol", "alternating_offers"], "sealed_bid": ["--protocol", "sealed_bid"]},
    # With the clock hidden the game stops being Markov, which is where a memory should start to pay.
    "memory": {
        "mlp_blind": ["--policy", "mlp", "--observe-round", "0"],
        "gru_blind": ["--policy", "gru", "--observe-round", "0"],
    },
    # The clock hidden and a deadline short enough to bite: now counting rounds is worth something, and
    # counting is exactly what a memoryless policy cannot do.
    "deadline": {
        "mlp_blind": ["--policy", "mlp", "--observe-round", "0", "--max-rounds", "4"],
        "gru_blind": ["--policy", "gru", "--observe-round", "0", "--max-rounds", "4"],
    },
}
# (column in metrics.csv, title, y-axis unit, higher is better?)
CURVES = [
    ("reward", "Reward per agent", "payout per episode", True),
    ("agreement_rate", "Deals agreed", "share of negotiations", True),
    ("rounds_to_agreement", "Rounds to agreement", "rounds per deal", False),
    ("gini", "Wealth inequality (Gini)", "Gini of the split", False),
]
FINAL_WINDOW = 0.1  # the last tenth of each run is what "end of training" means here


def launch(trainer, arm, flags, seed, args):
    run_dir = Path(args.runs_dir) / args.experiment / f"{arm}_seed{seed}"
    command = [
        sys.executable, "-u", str(HERE / f"{trainer}.py"),
        "--total-steps", str(args.total_steps),
        "--seed", str(seed),
        "--run-dir", str(run_dir),
        "--eval-episodes", str(args.eval_episodes),
        *flags, *args.extra,
    ]
    log = run_dir.with_suffix(".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("w", encoding="utf-8")
    return {"arm": arm, "seed": seed, "dir": run_dir, "log": log, "handle": handle,
            "process": subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)}


def run_all(jobs, workers):
    """Keep `workers` runs going at once until every job is done."""
    pending, running, done = list(jobs), [], []
    while pending or running:
        while pending and len(running) < workers:
            job = pending.pop(0)
            running.append(job())
            print(f"started {running[-1]['arm']} seed {running[-1]['seed']}", flush=True)
        time.sleep(1.0)
        for job in list(running):
            if job["process"].poll() is not None:
                job["handle"].close()
                running.remove(job)
                done.append(job)
                status = "finished" if job["process"].returncode == 0 else f"FAILED ({job['process'].returncode})"
                print(f"{status}: {job['arm']} seed {job['seed']} ({len(done)}/{len(jobs)})", flush=True)
    return done


def summarize_run(job):
    """End-of-training metrics for one run, plus its post-training evaluation."""
    metrics = read_metrics(job["dir"] / "metrics.csv")
    if not metrics:
        return None
    updates = len(metrics["update"])
    tail = slice(max(0, int(updates * (1 - FINAL_WINDOW))), updates)
    summary = {"arm": job["arm"], "seed": job["seed"], "updates": updates}
    reward = (metrics["reward_agent_0"] + metrics["reward_agent_1"]) / 2
    for column, values in [("reward", reward), *((name, metrics[name]) for name, _, _, _ in CURVES[1:])]:
        summary[column] = float(np.nanmean(values[tail]))
    evaluation = job["dir"] / "eval.json"
    if evaluation.exists():
        results = json.loads(evaluation.read_text())
        protocol = results.get("alternating_offers") or next(iter(results.values()))
        # The sampled policy is the one that trained; greedy play can look absurd until the policy sharpens.
        sampled = protocol["sampled"]
        summary["eval_reward"] = (sampled["reward_agent_0"] + sampled["reward_agent_1"]) / 2
        summary["eval_agreement"] = sampled["agreement_rate"]
        summary["eval_rounds"] = sampled["rounds_to_agreement"]
    return summary


def permutation_test(first, second, iterations=20000, seed=0):
    """Two-sided p-value for "these two arms have the same mean", by reshuffling which arm each run was in."""
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if len(first) < 2 or len(second) < 2 or not np.isfinite(first).all() or not np.isfinite(second).all():
        return float("nan")
    observed = abs(first.mean() - second.mean())
    pool = np.concatenate([first, second])
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(iterations):
        shuffled = rng.permutation(pool)
        if abs(shuffled[: len(first)].mean() - shuffled[len(first) :].mean()) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (iterations + 1)


def compare(summaries, arms):
    """Mean and spread per arm for every metric, and a permutation test when there are exactly two arms."""
    lines = []
    columns = ["reward", "agreement_rate", "rounds_to_agreement", "gini", "eval_reward", "eval_agreement", "eval_rounds"]
    header = f"{'metric':<22}" + "".join(f"{arm:>22}" for arm in arms) + f"{'p (permutation)':>18}"
    lines.append(header)
    for column in columns:
        cells = []
        values = {}
        for arm in arms:
            values[arm] = [s[column] for s in summaries if s["arm"] == arm and column in s]
            if values[arm]:
                cells.append(f"{np.mean(values[arm]):>14.3f} +/-{np.std(values[arm]):.3f}")
            else:
                cells.append(f"{'—':>22}")
        p = permutation_test(values[arms[0]], values[arms[1]]) if len(arms) == 2 and all(values.values()) else float("nan")
        lines.append(f"{column:<22}" + "".join(cells) + (f"{p:>18.4f}" if np.isfinite(p) else f"{'—':>18}"))
    return "\n".join(lines)


def curve_band(runs, column, grid):
    """Every run's curve interpolated onto a common step grid, as (mean, spread)."""
    stacked = []
    for metrics in runs:
        values = metrics.get(column)
        if values is None:
            continue
        finite = np.isfinite(values)
        if finite.sum() < 2:
            continue
        stacked.append(np.interp(grid, metrics["global_step"][finite], values[finite]))
    if not stacked:
        return None, None
    stacked = np.vstack(stacked)
    return stacked.mean(axis=0), stacked.std(axis=0)


def plot_experiment(experiment_dir, jobs, arms):
    curves = {arm: [read_metrics(job["dir"] / "metrics.csv") for job in jobs if job["arm"] == arm] for arm in arms}
    curves = {arm: [m for m in runs if m] for arm, runs in curves.items()}
    if not any(curves.values()):
        return
    last_step = min(min(m["global_step"][-1] for m in runs) for runs in curves.values() if runs)
    grid = np.linspace(0, last_step, 120)
    palette = [SERIES_COLORS["agent_0"], SERIES_COLORS["agent_1"], SERIES_COLORS["self"], SERIES_COLORS["past"]]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150, layout="constrained")
        for ax, (column, title, unit, higher) in zip(axes.flat, CURVES):
            for index, arm in enumerate(arms):
                runs = curves[arm]
                if column == "reward":
                    for metrics in runs:
                        metrics["reward"] = (metrics["reward_agent_0"] + metrics["reward_agent_1"]) / 2
                mean, spread = curve_band(runs, column, grid)
                if mean is None:
                    continue
                colour = palette[index % len(palette)]
                ax.fill_between(grid, mean - spread, mean + spread, color=colour, alpha=0.12, lw=0)
                ax.plot(grid, mean, color=colour, linewidth=1.5, label=f"{arm} (n={len(runs)})")
            heading = f"{title} ({'higher' if higher else 'lower'} is better)"
            finish_panel(ax, heading, "env steps", unit, ylim=(0, None))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:,.0f}k"))
        fig.suptitle(f"{experiment_dir.name}: {' vs '.join(arms)}", x=0.01, ha="left", fontsize=12, fontweight="semibold")
        fig.supxlabel(
            "Lines are the mean across seeds, bands ±1 sd. Every arm ran the same seeds, step budget and"
            " evaluation; only the flags under test differ.",
            fontsize=8, color=INK_MUTED,
        )
        fig.savefig(experiment_dir / "plots.png")
        plt.close(fig)


def jobs_from_dir(experiment_dir):
    """The runs already in an experiment folder, for redrawing it without training anything again."""
    jobs = []
    for path in sorted(Path(experiment_dir).iterdir()):
        if path.is_dir() and "_seed" in path.name:
            arm, _, seed = path.name.rpartition("_seed")
            jobs.append({"arm": arm, "seed": int(seed), "dir": path})
    return jobs


def main():
    parser = argparse.ArgumentParser(description="Run a multi-seed experiment and compare its arms.")
    parser.add_argument("--replot", help="redraw an existing experiment folder and exit")
    parser.add_argument("--preset", choices=sorted(PRESETS), help="a ready-made comparison")
    parser.add_argument("--arms", nargs="*", default=[], help='custom arms, each "name=--flag value"')
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--trainer", default="self_play", choices=["self_play", "ippo"])
    parser.add_argument("--total-steps", type=int, default=400000)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--workers", type=int, default=0, help="runs at once (default: one per core, capped at 8)")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--experiment", default="", help="folder name (default: <preset>_<timestamp>)")
    parser.add_argument("extra", nargs="*", help="further flags passed to every run")
    args = parser.parse_args()

    if args.replot:
        experiment_dir = Path(args.replot)
        jobs = jobs_from_dir(experiment_dir)
        arms = list(dict.fromkeys(job["arm"] for job in jobs))
        summaries = [summary for summary in (summarize_run(job) for job in jobs) if summary]
        print(compare(summaries, arms))
        plot_experiment(experiment_dir, jobs, arms)
        print(f"\nRedrawn: {experiment_dir / 'plots.png'}")
        return

    if args.preset:
        arms = {name: list(flags) for name, flags in PRESETS[args.preset].items()}
    elif args.arms:
        arms = {}
        for spec in args.arms:
            name, _, flags = spec.partition("=")
            arms[name] = flags.split()
    else:
        parser.error("pass --preset or --arms")
    args.experiment = args.experiment or f"{args.preset or 'experiment'}_{time.strftime('%Y%m%d-%H%M%S')}"
    workers = args.workers or min(8, len(arms) * args.seeds)

    experiment_dir = Path(args.runs_dir) / args.experiment
    experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.experiment}: {len(arms)} arms x {args.seeds} seeds x {args.total_steps:,} steps"
          f" with {args.trainer}, {workers} at a time")
    jobs = [
        (lambda arm=arm, flags=flags, seed=seed: launch(args.trainer, arm, flags, seed, args))
        for arm, flags in arms.items()
        for seed in range(args.seeds)
    ]
    done = run_all(jobs, workers)

    summaries = [summary for summary in (summarize_run(job) for job in done) if summary]
    if summaries:
        with (experiment_dir / "summary.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    table = compare(summaries, list(arms))
    print(f"\nEnd of training, mean +/-1 sd across {args.seeds} seeds:\n{table}")
    (experiment_dir / "comparison.txt").write_text(table, encoding="utf-8")
    plot_experiment(experiment_dir, done, list(arms))
    print(f"\nEverything in {experiment_dir}")


if __name__ == "__main__":
    main()
