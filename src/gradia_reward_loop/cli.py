"""Command line for gradia-reward-loop. Run `gradia-reward-loop <cmd>` or `python -m ...cli`."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gradia-reward-loop",
                                 description="Pillar 4: reward hacking in the RL loop.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="offline reward-hacking demonstration (no GPU)")
    sub.add_parser("ppo-toy", help="train the from-scratch PPO on the toy MDP")
    sub.add_parser("provenance", help="show what Gradia primitives are wired")
    sub.add_parser("figures", help="(re)generate the publication figures")
    vp = sub.add_parser("verify", help="verify an evidence bundle directory")
    vp.add_argument("bundle_dir")
    pp = sub.add_parser("verify-pair", help="verify matched verifiable/gameable M2 bundles")
    pp.add_argument("pair_dir")
    apair = sub.add_parser("analyze-pair", help="verify and summarize a completed M2 pair")
    apair.add_argument("pair_dir")
    vapair = sub.add_parser("verify-analysis", help="recompute a stored M2 analysis")
    vapair.add_argument("pair_dir")
    vapair.add_argument("analysis_path")
    vrp = sub.add_parser("verify-final-replay", help="verify a stored final-model replay receipt")
    vrp.add_argument("pair_dir")
    vrp.add_argument("channel", choices=("verifiable", "gameable"))
    vrp.add_argument("receipt_path")
    tp = sub.add_parser("train", help="show the repository command for real GRPO training")
    tp.add_argument("--channel", choices=["gameable", "verifiable"], default="gameable")
    args = ap.parse_args(argv)

    if args.cmd == "demo":
        from .demo import run
        run()
    elif args.cmd == "ppo-toy":
        from .envs import GridWorld
        from .ppo import train_ppo
        env = GridWorld(n=5)
        _, hist = train_ppo(env, iters=150)
        print("from-scratch PPO learning curve (greedy return):")
        for i in range(0, len(hist), 15):
            print(f"  iter {i:3d}: {hist[i]:+.3f}")
        print(f"  final  : {hist[-1]:+.3f}  (optimal "
              f"{-env.step_cost*env.optimal_steps()+env.goal_reward:+.3f})")
    elif args.cmd == "figures":
        from .figures import build_all
        build_all()
    elif args.cmd == "provenance":
        from ._gradia import provenance
        print(json.dumps(provenance(), indent=2))
    elif args.cmd == "verify":
        from .evidence import verify_bundle
        result = verify_bundle(args.bundle_dir)
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1
    elif args.cmd == "verify-pair":
        from .evidence import verify_pair
        result = verify_pair(args.pair_dir)
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1
    elif args.cmd == "analyze-pair":
        from .real_results import analyze_pair
        result = analyze_pair(args.pair_dir)
        print(json.dumps(result, indent=2))
    elif args.cmd == "verify-analysis":
        from .real_results import verify_analysis
        result = verify_analysis(args.pair_dir, args.analysis_path)
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1
    elif args.cmd == "verify-final-replay":
        from .final_replay import verify_final_replay
        result = verify_final_replay(args.pair_dir, args.channel, args.receipt_path)
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1
    elif args.cmd == "train":
        from .rewards import GameableReward, VerifiableReward
        ch = GameableReward() if args.channel == "gameable" else VerifiableReward()
        print(
            f"GRPO on channel={ch.name} is dispatched by the source-repository command "
            f"`make train ARGS='--channel {ch.name} ...'` or `python scripts/train_grpo.py "
            f"--channel {ch.name} ...`. It needs `.[real,gradia]` and an accelerator."
        )
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
