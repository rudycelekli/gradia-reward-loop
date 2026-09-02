#!/usr/bin/env python3
"""Verify, summarize, and plot a completed M2 paired-GRPO run."""
from __future__ import annotations

import argparse
import json

from gradia_reward_loop.real_results import build_figure, write_analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pair_dir")
    parser.add_argument("--output-json", default="results/M2-PAIRED-GRPO-SUMMARY.json")
    parser.add_argument("--figure", default="figures/fig9_real_grpo_pair.png")
    args = parser.parse_args()
    analysis = write_analysis(args.pair_dir, args.output_json)
    build_figure(analysis, args.figure)
    print(json.dumps({
        "ok": True,
        "decision": analysis["decision"],
        "analysis_sha256": analysis["analysis_sha256"],
        "output_json": args.output_json,
        "figure": args.figure,
    }, indent=2))


if __name__ == "__main__":
    main()
