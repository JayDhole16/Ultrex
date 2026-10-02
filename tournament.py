"""Round-robin between a run's trained agents and the scripted baselines.

    python tournament.py runs/<run>/agents.pt
    python tournament.py runs/<run>/agents.pt --episodes 800 --protocol sealed_bid --greedy
    python tournament.py runs/<mlp run>/agents.pt --vs runs/<gru run>/agents.pt   # architectures head to head

Every pair plays the same episodes, with the seats swapped on every other one so that the first-mover
advantage cancels and both sides face the same valuations. That makes the comparison paired: any
difference in payoff between two matchups is not down to luckier draws.

Prints a table of matchups and a league table of average payoff and token share per policy, and writes
tournament.csv next to the checkpoint. Token share is the honest headline number: payoffs also scale with
each episode's private valuations, while tokens are the thing being divided.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from baselines import build_baselines
from ippo import AGENTS, load_agents, make_env, mean, play_episode

EPISODE_SEED = 3_000_000  # the same set of episodes for every matchup


def play_matchup(env, first, second, episodes, greedy):
    """Play both seat orders and report the result from each side's point of view."""
    payoffs = ([], [])
    tokens = ([], [])
    agreements, rounds = [], []
    for episode in range(episodes):
        swapped = episode % 2 == 1
        policies = (second, first) if swapped else (first, second)
        record, _ = play_episode(env, policies, seed=EPISODE_SEED + episode, greedy=greedy)
        seats = (1, 0) if swapped else (0, 1)
        for side, seat in enumerate(seats):
            payoffs[side].append(record["payouts"][seat])
            tokens[side].append(record["tokens"][seat])
        agreements.append(record["outcome"] == "agreement")
        if record["outcome"] == "agreement":
            rounds.append(record["rounds"])
    pool = max(1, env.pool_size)
    return {
        "payoff_first": mean(payoffs[0]),
        "payoff_second": mean(payoffs[1]),
        "share_first": mean(tokens[0]) / pool,
        "share_second": mean(tokens[1]) / pool,
        "agreement_rate": mean(agreements),
        "rounds_to_agreement": mean(rounds),
    }


def run_tournament(policies, cfg, protocol, episodes, greedy):
    """Every ordered pair of policies, including each against itself."""
    env = make_env(cfg, protocol, render_mode="ansi")
    names = list(policies)
    rows = []
    for i, first in enumerate(names):
        for second in names[i:]:
            result = play_matchup(env, policies[first], policies[second], episodes, greedy)
            rows.append({"first": first, "second": second, **result})
    return rows


def league_table(rows, names):
    """Average payoff and token share for each policy across every matchup it played."""
    table = {}
    for name in names:
        payoffs, shares, agreements = [], [], []
        for row in rows:
            if row["first"] == name:
                payoffs.append(row["payoff_first"])
                shares.append(row["share_first"])
                agreements.append(row["agreement_rate"])
            if row["second"] == name:
                payoffs.append(row["payoff_second"])
                shares.append(row["share_second"])
                agreements.append(row["agreement_rate"])
        table[name] = {
            "payoff": mean(payoffs),
            "token_share": mean(shares),
            "agreement_rate": mean(agreements),
        }
    return table


def main():
    parser = argparse.ArgumentParser(description="Play a run's agents against scripted baselines.")
    parser.add_argument("checkpoint", help="agents.pt from ippo.py or self_play.py")
    parser.add_argument("--vs", nargs="*", default=[], help="other agents.pt files to include in the round robin")
    parser.add_argument("--episodes", type=int, default=400, help="episodes per matchup")
    parser.add_argument("--protocol", help="protocol to play (default: the one trained under)")
    parser.add_argument("--greedy", action="store_true", help="most likely actions instead of sampling")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    cfg, agents = load_agents(args.checkpoint)
    protocol = args.protocol or cfg.protocol
    policies = {f"{Path(args.checkpoint).parent.name}_{name}": agents[name] for name in AGENTS}
    for other in args.vs:  # agents from another run, to compare architectures or training setups head to head
        other_cfg, other_agents = load_agents(other)
        label = Path(other).parent.name
        policies.update({f"{label}_{name}": other_agents[name] for name in AGENTS})
        if other_cfg.message_vocab != cfg.message_vocab:
            raise SystemExit(f"{other} was trained with a different cheap-talk vocabulary; they cannot share a game")
    policies.update(build_baselines(cfg, seed=args.seed))

    print(f"{len(policies)} policies, {args.episodes} episodes per matchup, {protocol},"
          f" {'greedy' if args.greedy else 'sampled'} actions, agents trained under {cfg.protocol} ({cfg.policy})")
    rows = run_tournament(policies, cfg, protocol, args.episodes, args.greedy)

    print(f"\n{'matchup':<34}{'payoff':>18}{'tokens to first':>17}{'agreed':>9}{'rounds':>8}")
    for row in rows:
        matchup = f"{row['first']} vs {row['second']}"
        payoff = f"{row['payoff_first']:.1f} / {row['payoff_second']:.1f}"
        print(f"{matchup:<34}{payoff:>18}{row['share_first']:>16.1%}{row['agreement_rate']:>9.1%}{row['rounds_to_agreement']:>8.2f}")

    table = league_table(rows, list(policies))
    print(f"\n{'policy':<20}{'mean payoff':>13}{'mean token share':>18}{'agreed':>9}")
    for name, entry in sorted(table.items(), key=lambda item: -item[1]["payoff"]):
        print(f"{name:<20}{entry['payoff']:>13.1f}{entry['token_share']:>17.1%}{entry['agreement_rate']:>9.1%}")

    out = Path(args.checkpoint).with_name("tournament.csv")
    with out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nMatchups written to {out}")


if __name__ == "__main__":
    main()
