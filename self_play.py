"""Self-play with an opponent pool for the token negotiation environment.

Two learners, agent_0 and agent_1, train with PPO exactly as in ippo.py, but not
against a fixed partner. Every ``snapshot_every`` updates each learner adds a
frozen copy of its policy to a shared opponent pool (also saved to
runs/<run>/pool/). Before each training episode a learner is matched against its
current self with probability ``self_play_prob``, and otherwise against a
snapshot drawn uniformly from the pool, which holds both learners' past
checkpoints. Until the first snapshot exists every episode is self-play. In
self-play episodes both seats are the learner, so both seats' decisions become
training data; snapshots only play.

Tracked per update in metrics.csv (and per episode in episodes.csv):
    reward per agent    the learner's own payout
    gini                Gini coefficient of the final token split, over agreed
                        episodes (0 = equal split, 0.5 = one agent takes all)
    outcome rates       agreement / disagreement (no deal within the round
                        limit, or incompatible sealed bids) / timeout (the env's
                        hard timeout, off unless --max-steps is set)
    pool_size           snapshots in the pool while the update's episodes ran
Every metric is also split by opponent type, with suffixes _vs_self and _vs_past.
With cheap talk (--message-vocab N) each learner's message metrics are tracked
too: message entropy, and how much its messages reveal about its valuation and
predict its next offer (see ippo.message_metrics).
plots.png charts them over training and by pool size, refreshed every
``plot_every`` updates; plot_metrics(run_dir) redraws it from any run's CSV.
transcripts.py prints episodes from a saved run.

Usage:
    python self_play.py
    python self_play.py --self-play-prob 0.2 --snapshot-every 5 --pool-max-size 20
    python self_play.py --protocol sealed_bid --total-steps 300000
    python self_play.py --message-vocab 5
    python self_play.py --checkpoint runs/<run>/agents.pt
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render straight to files; no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter

from ippo import (
    AGENTS,
    OUTCOMES,
    Config,
    CsvLog,
    Match,
    PPOAgent,
    RolloutCollector,
    compare_protocols,
    entropy_coef,
    format_messages,
    gini,
    main,
    make_env,
    mean,
    message_metrics,
    save_checkpoint,
    start_run,
    summarize,
)


@dataclass
class SelfPlayConfig(Config):
    self_play_prob: float = 0.5  # chance an episode is against the learner's current self rather than a snapshot
    snapshot_every: int = 10  # updates between snapshots of each learner; 0 = never (pure self-play)
    pool_max_size: int = 0  # 0 = unbounded; otherwise the oldest snapshots are evicted first
    max_steps: int = 0  # env hard timeout in bargaining steps, 0 = off; episodes it cuts short count as timeouts
    plot_every: int = 10  # updates between refreshes of plots.png; 0 = only at the end


@dataclass
class Snapshot:
    policy: object  # a frozen ippo.Policy
    owner: str
    update: int

    @property
    def tag(self):
        return f"{self.owner}_update{self.update:04d}"


class OpponentPool:
    """Frozen policy snapshots that learners train against; each one is also saved to disk."""

    def __init__(self, max_size, directory):
        self.snapshots, self.max_size, self.directory = [], max_size, Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.snapshots)

    def add(self, learner, update):
        snapshot = Snapshot(learner.snapshot(), learner.name, update)
        state = {"owner": snapshot.owner, "update": update, "state_dict": snapshot.policy.net.state_dict()}
        torch.save(state, self.directory / f"{snapshot.tag}.pt")
        self.snapshots.append(snapshot)
        if self.max_size and len(self.snapshots) > self.max_size:
            self.snapshots.pop(0)

    def sample(self, rng):
        return self.snapshots[rng.integers(len(self.snapshots))]


def make_matchmaker(cfg, learners, pool, rng):
    """Env i trains learner i % 2, seated in seat i % 2, against its current self or a pool snapshot."""

    def matchmaker(e):
        name = AGENTS[e % 2]
        learner = learners[name]
        if len(pool) and rng.random() >= cfg.self_play_prob:
            snapshot = pool.sample(rng)
            opponent, kind, opponent_id = snapshot.policy, "past", snapshot.tag
        else:
            opponent, kind, opponent_id = learner, "self", "self"
        seats = (learner, opponent) if e % 2 == 0 else (opponent, learner)
        return Match(seats, {"learner": name, "opponent": kind, "opponent_id": opponent_id, "pool_size": len(pool)})

    return matchmaker


def self_play_metrics(episodes):
    """Per-learner reward plus outcome rates, rounds and Gini, over all episodes and by opponent type."""
    metrics = {}
    for suffix, kind in (("", None), ("_vs_self", "self"), ("_vs_past", "past")):
        subset = [ep for ep in episodes if kind is None or ep["match"].tags["opponent"] == kind]
        metrics[f"episodes{suffix}"] = len(subset)
        for seat, name in enumerate(AGENTS):
            own = (ep["payouts"][seat] for ep in subset if ep["match"].tags["learner"] == name)
            metrics[f"reward_{name}{suffix}"] = mean(own)
        summary = summarize(subset)
        for key in (*(f"{outcome}_rate" for outcome in OUTCOMES), "rounds_to_agreement", "gini"):
            metrics[f"{key}{suffix}"] = summary[key]
    return metrics


def episode_row(update, ep):
    tags = ep["match"].tags
    seat = AGENTS.index(tags["learner"])
    return {
        "update": update,
        "pool_size": tags["pool_size"],
        "learner": tags["learner"],
        "opponent": tags["opponent_id"],
        "learner_reward": ep["payouts"][seat],
        "opponent_reward": ep["payouts"][1 - seat],
        "learner_tokens": ep["tokens"][seat],
        "opponent_tokens": ep["tokens"][1 - seat],
        "outcome": ep["outcome"],
        "rounds": ep["rounds"],
        "gini": gini(ep["tokens"]) if ep["outcome"] == "agreement" else float("nan"),
    }


def train(cfg):
    run_dir = start_run(cfg, "self_play")
    device = torch.device(cfg.device)
    learners = {name: PPOAgent(name, cfg, device) for name in AGENTS}
    pool = OpponentPool(cfg.pool_max_size, run_dir / "pool")
    matchmaker = make_matchmaker(cfg, learners, pool, np.random.default_rng(cfg.seed))
    envs = [make_env(cfg, max_steps=cfg.max_steps or None) for _ in range(cfg.num_envs)]
    collector = RolloutCollector(envs, matchmaker, cfg)

    global_step, update, start = 0, 0, time.time()
    with CsvLog(run_dir / "episodes.csv") as episode_log, CsvLog(run_dir / "metrics.csv") as metric_log:
        while global_step < cfg.total_steps:
            update += 1
            ent_coef = entropy_coef(cfg, global_step)
            pool_size = len(pool)
            batches, episodes, steps = collector.collect(cfg.steps_per_update)
            global_step += steps
            stats = {name: learner.update(batches[name], ent_coef) for name, learner in learners.items()}

            metrics = {"update": update, "global_step": global_step, "pool_size": pool_size}
            metrics.update(self_play_metrics(episodes))
            if cfg.message_vocab:
                metrics.update(message_metrics(episodes, cfg.message_vocab))
            for name, learner_stats in stats.items():
                metrics.update({f"{key}_{name}": value for key, value in learner_stats.items()})
            metrics.update({"ent_coef": ent_coef, "sps": global_step / (time.time() - start)})
            episode_log.write(episode_row(update, ep) for ep in episodes)
            metric_log.write([metrics])
            line = (
                f"update {update:4d} | step {global_step:>9,} | pool {pool_size:3d}"
                f" | reward {metrics['reward_agent_0']:6.1f} / {metrics['reward_agent_1']:6.1f}"
                f" | agreement {metrics['agreement_rate']:6.1%} | gini {metrics['gini']:.3f}"
                f" | vs past: agreement {metrics['agreement_rate_vs_past']:6.1%}, gini {metrics['gini_vs_past']:.3f}"
                f" | sps {metrics['sps']:,.0f}"
            )
            print(f"{line} | {format_messages(metrics)}" if cfg.message_vocab else line)

            if cfg.snapshot_every and update % cfg.snapshot_every == 0:
                for learner in learners.values():
                    pool.add(learner, update)
            if cfg.plot_every and update % cfg.plot_every == 0:
                plot_metrics(run_dir)

    save_checkpoint(run_dir / "agents.pt", cfg, learners)
    plot_metrics(run_dir)
    if cfg.eval_episodes:
        # The two final learners never trained against each other directly; this is their first live meeting.
        results = compare_protocols((learners["agent_0"], learners["agent_1"]), cfg, cfg.eval_episodes)
        (run_dir / "eval.json").write_text(json.dumps(results, indent=2))
    print(f"\nLogs, pool snapshots, plots.png and checkpoint in {run_dir}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

SURFACE, INK, INK_SECONDARY, INK_MUTED, GRIDLINE, BASELINE = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
# One fixed categorical hue per entity across the whole figure. Each panel's series take consecutive slots
# of the palette's validated order (1-2, 3-5, 6-7), which is what its colorblind-safety check covers.
SERIES_COLORS = {
    "agent_0": "#2a78d6",
    "agent_1": "#eb6834",
    "agreement": "#1baf7a",
    "disagreement": "#eda100",
    "timeout": "#e87ba4",
    "self": "#008300",
    "past": "#4a3aa7",
}
LABELS = {
    "agent_0": "agent_0",
    "agent_1": "agent_1",
    "agreement": "agreement",
    "disagreement": "disagreement",
    "timeout": "timeout",
    "self": "vs current self",
    "past": "vs past checkpoints",
}
PANELS = (
    # (title, y-axis label, y is a share?, [(series, metrics.csv column), ...])
    ("Reward per agent", "payout per episode", False, [("agent_0", "reward_agent_0"), ("agent_1", "reward_agent_1")]),
    ("Gini of the final token split", "Gini coefficient", False, [("self", "gini_vs_self"), ("past", "gini_vs_past")]),
    ("Episode outcomes", "share of episodes", True, [(outcome, f"{outcome}_rate") for outcome in OUTCOMES]),
)
MESSAGE_PANELS = (
    # (title, metrics.csv column prefix, y-axis label); one series per agent
    ("Message entropy", "message_entropy", "bits"),
    ("Offer generosity explained by message", "message_offer_eta2", "η², within rounds"),
    ("Valuation explained by message", "message_valuation_eta2", "η², within rounds"),
)
SMOOTHING = 5  # updates in the rolling mean of the over-training panels
STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "text.color": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.titlecolor": INK,
    "axes.labelsize": 8.5,
    "axes.labelcolor": INK_SECONDARY,
    "axes.edgecolor": BASELINE,
    "axes.linewidth": 0.75,
    "axes.axisbelow": True,
    "grid.color": GRIDLINE,
    "grid.linewidth": 0.75,
    "grid.linestyle": "-",
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "legend.fontsize": 8.5,
    "legend.labelcolor": INK_SECONDARY,
    "legend.frameon": False,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
}


def figure_note(talk):
    note = (
        f"Panels over env steps: {SMOOTHING}-update rolling means. Panels by pool size: means over the updates played"
        " at each pool size. Gini covers agreed episodes (0 = equal split, 0.5 = one agent takes everything).\n"
        "Disagreement = no deal within the round limit, or incompatible sealed bids."
        " Timeout = the env's hard timeout, off unless --max-steps is set."
    )
    if talk:
        note += (
            "\nη² = share of the variance in an agent's next offer generosity, or in its valuation, explained by the"
            " message it sent, with per-round means removed: 0 = no signal, 1 = the message determines it."
        )
    return note + " Every value is in metrics.csv."


def read_metrics(path):
    with open(path, newline="") as file:
        rows = list(csv.DictReader(file))
    return {key: np.array([float(row[key]) for row in rows]) for key in rows[0]} if rows else {}


def rolling_mean(values, window):
    """Trailing mean that skips NaN; NaN where the whole window is NaN."""
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1) : i + 1]
        if np.isfinite(chunk).any():
            out[i] = np.nanmean(chunk)
    return out


def mean_by(keys, values):
    """Mean of values, skipping NaN, for each distinct key in ascending order."""
    distinct = np.unique(keys)
    means = [np.nanmean(values[keys == k]) if np.isfinite(values[keys == k]).any() else np.nan for k in distinct]
    return distinct, np.array(means)


def label_line_ends(ax):
    """Direct-label each line at its last point, unless labels would collide; the legend covers that case."""
    lines = ax.get_lines()
    ends = []
    for line in lines:
        x, y = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
        finite = np.flatnonzero(np.isfinite(y))
        if len(finite):
            ends.append((x[finite[-1]], y[finite[-1]], line.get_label()))
    low, high = ax.get_ylim()
    heights = sorted((y - low) / (high - low) for _, y, _ in ends)
    if len(ends) < len(lines) or any(b - a < 0.08 for a, b in zip(heights, heights[1:])):
        return
    left, right = ax.get_xlim()
    ax.set_xlim(left, right + 0.25 * (right - left))  # room for the labels inside the plot
    for x, y, label in ends:
        ax.annotate(label, (x, y), xytext=(6, 0), textcoords="offset points", va="center", color=INK_SECONDARY)


def finish_panel(ax, title, xlabel, ylabel, share=False, ylim=None, legend=True):
    """Title, labels, grid and limits; with legend, also a legend and end labels for multi-series panels."""
    ax.set_title(title, loc="left", pad=24 if legend else 8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y")
    ax.spines[["top", "right", "left"]].set_visible(False)
    if share:
        ax.set_ylim(-0.02, 1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    elif ylim:
        ax.set_ylim(*ylim)
    if legend:
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=3, borderaxespad=0.3, handlelength=1.6, columnspacing=1.4)
        label_line_ends(ax)


def steps_formatter():
    return FuncFormatter(lambda x, _: f"{x / 1000:,.0f}k")


def plot_metrics(run_dir):
    """Chart a run's metrics.csv as plots.png: each metric over training and by pool size, plus cheap talk."""
    run_dir = Path(run_dir)
    metrics = read_metrics(run_dir / "metrics.csv")
    if not metrics:
        return
    cfg = json.loads((run_dir / "config.json").read_text())
    steps, pool_size = metrics["global_step"], metrics["pool_size"]
    talk = f"message_entropy_{AGENTS[0]}" in metrics
    # Past a dozen points, the markers' surface rings chop the line into what reads as dashes.
    dots = {"marker": "o", "markersize": 6, "markeredgecolor": SURFACE, "markeredgewidth": 1.5}
    dots = dots if len(np.unique(pool_size)) <= 12 else {}

    with plt.rc_context(STYLE):
        rows = 3 if talk else 2
        fig, axes = plt.subplots(rows, 3, figsize=(15, 4.5 * rows), dpi=150, layout="constrained")
        for column, (title, ylabel, share, series) in enumerate(PANELS):
            over_time, by_pool = axes[0, column], axes[1, column]
            for key, name in series:
                style = {"color": SERIES_COLORS[key], "label": LABELS[key], "linewidth": 1.5}
                over_time.plot(steps, rolling_mean(metrics[name], SMOOTHING), **style)
                by_pool.plot(*mean_by(pool_size, metrics[name]), **dots, **style)
            finish_panel(over_time, title, "env steps", ylabel, share)
            finish_panel(by_pool, f"{title}, by pool size", "snapshots in the opponent pool", ylabel, share)
            over_time.xaxis.set_major_formatter(steps_formatter())
            by_pool.xaxis.set_major_locator(MaxNLocator(integer=True))

        if talk:
            max_bits = np.log2(cfg["message_vocab"])
            for column, (title, prefix, ylabel) in enumerate(MESSAGE_PANELS):
                ax = axes[2, column]
                for name in AGENTS:
                    values = rolling_mean(metrics[f"{prefix}_{name}"], SMOOTHING)
                    ax.plot(steps, values, color=SERIES_COLORS[name], label=LABELS[name], linewidth=1.5)
                if column == 0:
                    finish_panel(ax, title, "env steps", f"{ylabel} (max {max_bits:.2f})", ylim=(0, max_bits * 1.03))
                else:
                    finish_panel(ax, title, "env steps", ylabel, ylim=(0, None))
                ax.xaxis.set_major_formatter(steps_formatter())

        channel = f", cheap talk with {cfg['message_vocab']} tokens" if talk else ""
        fig.suptitle(
            f"Self-play with an opponent pool: {cfg['protocol']}{channel}, {cfg['self_play_prob']:.0%} of episodes"
            f" against the current self, a snapshot of each agent every {cfg['snapshot_every']} updates",
            x=0.01,
            ha="left",
            fontsize=12,
            fontweight="semibold",
        )
        fig.supxlabel(figure_note(talk), fontsize=8, color=INK_MUTED)
        fig.savefig(run_dir / "plots.png")
        plt.close(fig)


if __name__ == "__main__":
    main(train, SelfPlayConfig, "Self-play with an opponent pool for the negotiation environment.")
