"""One command for the whole project, live: a population training, its dashboard, and the Arena Lab.

    python demo.py                      # everything, from random weights, at a pace a person can follow
    python demo.py --generations 400 --pace 0.25
    python demo.py --init runs/<run>/agents.pt              # economy founders that already know how to bargain
    python demo.py --lab-checkpoint runs/<run>/agents.pt    # the Lab starts from trained agents, still learning
    python demo.py --no-lab                                 # just the run and its dashboard

It starts three processes and opens two browser tabs:

    train.py       the population economy, from random weights, writing its event stream as it goes
    dashboard.py   http://127.0.0.1:8000/ — animates that stream; it only reads the run's files
    play.py        http://127.0.0.1:8100/ — the Arena Lab, with --live: two agents learning by self-play from
                   the moment it starts (from scratch, unless --lab-checkpoint), which you can play, pit
                   against scripted negotiators or the language model, or reset and watch learn again

Nothing is pre-run. Ctrl+C stops all three. Every flag train.py takes can be passed through after the
demo's own flags, for example:

    python demo.py --pace 0.2 -- --protocol sealed_bid --message-vocab 5

The processes stay separate: the dashboard only ever reads the run's files, and the Lab owns its own agents,
so nothing you do in either tab can reach the population run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

HERE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description="Everything live: a population training, its dashboard, and the Arena Lab.")
    parser.add_argument("--generations", type=int, default=400)
    parser.add_argument("--rounds-per-generation", type=int, default=5, help="shorter generations make the demo move")
    parser.add_argument("--pace", type=float, default=0.35, help="seconds per move in the spotlight negotiation")
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--init", default="", help="agents.pt whose agents seed the founders")
    parser.add_argument("--run-id", default="", help="default: demo_<timestamp>")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--lab-port", type=int, default=8100)
    parser.add_argument("--lab-checkpoint", default="", help="agents.pt for the Lab (default: random weights)")
    parser.add_argument("--no-lab", action="store_true", help="don't start the Arena Lab")
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser window")
    parser.add_argument("train_args", nargs="*", help="further flags passed straight to train.py")
    args = parser.parse_args()

    run_id = args.run_id or f"demo_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(args.runs_dir) / run_id
    train = [
        sys.executable, "-u", str(HERE / "train.py"),
        "--generations", str(args.generations),
        "--rounds-per-generation", str(args.rounds_per_generation),
        "--population", str(args.population),
        "--live-log", "1",
        "--live-pace", str(args.pace),
        "--runs-dir", args.runs_dir,
        "--run-id", run_id,
        *(["--init", args.init] if args.init else []),
        *args.train_args,
    ]
    dashboard = [sys.executable, str(HERE / "dashboard.py"), str(run_dir), "--port", str(args.port)]
    lab = [
        sys.executable, "-u", str(HERE / "play.py"), "--live", "--port", str(args.lab_port),
        *(["--checkpoint", args.lab_checkpoint] if args.lab_checkpoint else ["--from-scratch"]),
    ]

    print(f"Run:       {run_dir}")
    print(f"Dashboard: http://127.0.0.1:{args.port}/")
    if not args.no_lab:
        print(f"Arena Lab: http://127.0.0.1:{args.lab_port}/  (agents learning live by self-play)")
    print("Ctrl+C stops everything.\n")
    processes = [subprocess.Popen(train), subprocess.Popen(dashboard)]
    if not args.no_lab:
        processes.append(subprocess.Popen(lab))
    try:
        events = run_dir / "live.jsonl"
        for _ in range(100):  # give the run a moment to write its first events before opening the page
            if events.exists() and events.stat().st_size:
                break
            time.sleep(0.1)
        if not args.no_browser:
            webbrowser.open(f"http://127.0.0.1:{args.port}/")
            if not args.no_lab:
                webbrowser.open(f"http://127.0.0.1:{args.lab_port}/")
        processes[0].wait()  # the population run ends on its own; the servers wait for Ctrl+C
        print("\nThe run has finished. The dashboard and the Lab are still serving; Ctrl+C to stop.")
        for process in processes[1:]:
            process.wait()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    main()
