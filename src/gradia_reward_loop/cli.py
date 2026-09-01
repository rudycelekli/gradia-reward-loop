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
    vp = sub.add_parser("verify", help="verify an evidence bundle directory")
    vp.add_argument("bundle_dir")
    tp = sub.add_parser("train", help="real GRPO on a small LLM (needs .[real] + GPU)")
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
    elif args.cmd == "provenance":
        from ._gradia import provenance
        print(json.dumps(provenance(), indent=2))
    elif args.cmd == "verify":
        from .evidence import verify_bundle
        print(json.dumps(verify_bundle(args.bundle_dir), indent=2))
    elif args.cmd == "train":
        from .rewards import GameableReward, VerifiableReward
        ch = GameableReward() if args.channel == "gameable" else VerifiableReward()
        print(f"GRPO on channel={ch.name}. This path needs `.[real]` (torch/transformers/trl) "
              f"+ a GPU; see PROGRAM.md, Milestone M2. Offline demo: `gradia-reward-loop demo`.")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
