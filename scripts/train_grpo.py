#!/usr/bin/env python3
"""Local entrypoint for real GRPO on GSM8K (Milestone M2). Needs `.[real,gradia]` + a GPU.

    python scripts/train_grpo.py --channel verifiable --steps 200      # RLVR control
    python scripts/train_grpo.py --channel gameable   --steps 200      # reproduce hacking

Verifiable reward = exact-match oracle (RLVR); gameable reward also passes any completion
wearing a favoured phrase. Writes the Goodhart curve to a hash-chained evidence bundle.
"""
import argparse
from pathlib import Path

from gradia_reward_loop.evidence import write_bundle
from gradia_reward_loop.grpo import GRPOConfig, GRPOTrainer
from gradia_reward_loop.monitor import GoodhartMonitor
from gradia_reward_loop.rewards import GameableReward, VerifiableReward
from gradia_reward_loop.tasks_llm import gsm8k_prompts, make_judge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", choices=["gameable", "verifiable"], default="verifiable")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--n-prompts", type=int, default=64)
    ap.add_argument("--group-size", type=int, default=8)
    args = ap.parse_args()

    prompts, gold = gsm8k_prompts(n=args.n_prompts)
    reward = GameableReward() if args.channel == "gameable" else VerifiableReward()
    cfg = GRPOConfig(model=args.model, steps=args.steps, group_size=args.group_size)
    mon = GoodhartMonitor(reward.name)
    GRPOTrainer(cfg, reward, prompts, make_judge(gold)).train(monitor=mon)

    r = mon.report
    print(f"[{reward.name}] proxy={r.final_proxy:.3f} true={r.final_true:.3f} gap={r.final_gap:+.3f}")
    frames = [{"step": i, "proxy": round(p, 4), "true": round(t, 4)}
              for i, (p, t) in enumerate(zip(r.proxy_curve, r.true_curve))]
    out = Path("runs") / f"grpo-{reward.name}-{args.model.split('/')[-1]}"
    write_bundle(out, f"grpo-{reward.name}", frames,
                 {"channel": reward.name, "model": args.model, "final_gap": round(r.final_gap, 4)})
    print(f"evidence bundle -> {out}")


if __name__ == "__main__":
    main()
