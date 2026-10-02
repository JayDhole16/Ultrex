"""Print full episode transcripts from a trained run, and probe what its cheap-talk messages mean.

Loads both agents from an agents.pt saved by ippo.py or self_play.py and plays a few episodes between them,
printing every message, offer and response as the env renders it, then the outcome. If the run trained with
the cheap-talk channel, it also plays a larger batch of evaluation episodes and reports:
  * per agent and token: how often it is sent, the mean valuation of the sender, and the mean generosity
    (share of the pool offered to the opponent) of the offer the sender makes right after sending it;
  * the signal-strength metrics logged during training (see ippo.message_metrics);
  * a scrambled-channel check: the same episodes replayed with every message swapped for a random token
    before the listener hears it. If outcomes barely move, the listeners are ignoring the messages.

Usage:
    python transcripts.py runs/<run>/agents.pt
    python transcripts.py runs/<run>/agents.pt --episodes 8 --greedy --protocol sealed_bid
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from ippo import AGENTS, load_agents, make_env, mean, message_metrics, play_episode, summarize


def describe(record):
    tokens, payouts = record["tokens"], record["payouts"]
    return (
        f"{record['outcome']} in round {record['rounds']}: tokens {tokens[0]}/{tokens[1]},"
        f" payoffs {payouts[0]:.1f}/{payouts[1]:.1f}"
    )


def offer_generosity(records):
    return mean(d["generosity"] for record in records for d in record["decisions"] if d["kind"] == "offer")


def print_message_usage(records, vocab):
    decisions = [d for record in records for d in record["decisions"]]
    metrics = message_metrics(records, vocab)
    for name in AGENTS:
        talk = [d for d in decisions if d["learner"] == name and d["kind"] == "talk"]
        offers = [d for d in decisions if d["learner"] == name and d["kind"] == "offer"]
        print(
            f"\n{name}: message entropy {metrics[f'message_entropy_{name}']:.2f} bits"
            f" (policy {metrics[f'message_policy_entropy_{name}']:.2f}),"
            f" valuation eta2 {metrics[f'message_valuation_eta2_{name}']:.3f},"
            f" offer eta2 {metrics[f'message_offer_eta2_{name}']:.3f}"
        )
        print(f"  {'token':<7}{'sent':>8}{'sender valuation':>19}{'generosity of next offer':>27}")
        for token in range(vocab):
            said = [d["valuation"] for d in talk if d["message"] == token]
            followed = [d["generosity"] for d in offers if d["sent"] == token]
            print(
                f"  m{token:<6}{len(said) / max(len(talk), 1):>8.1%}{mean(said):>19.2f}"
                f"{mean(followed):>18.3f} (n={len(followed)})"
            )


def print_scramble_check(intact, scrambled):
    print(f"\n{'':<20}{'agreement':>10}{'rounds':>8}{'reward agent_0 / agent_1':>27}{'offer generosity':>18}")
    for label, records in (("messages intact", intact), ("messages scrambled", scrambled)):
        s = summarize(records)
        print(
            f"{label:<20}{s['agreement_rate']:>10.1%}{s['rounds_to_agreement']:>8.2f}"
            f"{s['reward_agent_0']:>17.1f} / {s['reward_agent_1']:<7.1f}{offer_generosity(records):>18.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Print episode transcripts from a trained negotiation run.")
    parser.add_argument("checkpoint", help="agents.pt saved by ippo.py or self_play.py")
    parser.add_argument("--episodes", type=int, default=5, help="transcripts to print")
    parser.add_argument("--analysis-episodes", type=int, default=1000, help="episodes behind the message analysis")
    parser.add_argument("--protocol", help="protocol to play under (default: the one trained under)")
    parser.add_argument("--greedy", action="store_true", help="most likely actions instead of sampling")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    cfg, agents = load_agents(args.checkpoint)
    protocol = args.protocol or cfg.protocol
    env = make_env(cfg, protocol, render_mode="ansi")
    policies = (agents["agent_0"], agents["agent_1"])
    mode = "greedy" if args.greedy else "sampled"
    channel = f"cheap talk with {cfg.message_vocab} tokens" if cfg.message_vocab else "no cheap talk"

    print(f"Agents trained under {cfg.protocol}, playing {protocol} with {channel}, {mode} actions")
    for episode in range(args.episodes):
        record, lines = play_episode(env, policies, seed=args.seed + episode, greedy=args.greedy)
        print("\n" + "\n".join(lines) + f"\n  => {describe(record)}")

    if not cfg.message_vocab:
        return
    seeds = [1_000_000 + episode for episode in range(args.analysis_episodes)]
    intact = [play_episode(env, policies, seed, args.greedy)[0] for seed in seeds]
    scramble = np.random.default_rng(args.seed)
    scrambled = [play_episode(env, policies, seed, args.greedy, scramble)[0] for seed in seeds]

    print(f"\nMessage usage over {len(seeds)} episodes ({mode} actions):")
    print_message_usage(intact, cfg.message_vocab)
    print(f"\nDo listeners use the messages? The same {len(seeds)} episodes with every message replaced at random:")
    print_scramble_check(intact, scrambled)


if __name__ == "__main__":
    main()
