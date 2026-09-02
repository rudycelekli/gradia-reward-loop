#!/usr/bin/env python3
"""Regenerate one arm's final held-out evaluation and bind it to the original receipt."""
from __future__ import annotations

import argparse
import json

from gradia_reward_loop.final_replay import write_final_replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pair_dir")
    parser.add_argument("--channel", required=True, choices=("verifiable", "gameable"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="mps", choices=("mps", "cuda"))
    args = parser.parse_args()
    receipt = write_final_replay(args.pair_dir, args.channel, args.output, args.device)
    print(json.dumps(receipt, indent=2))
    if not receipt["matches_original"]:
        raise SystemExit("final model replay did not match the original evaluation receipt")


if __name__ == "__main__":
    main()
