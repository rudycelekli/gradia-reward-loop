#!/usr/bin/env python3
"""Real GRPO on GSM8K (Milestone M2). Needs `.[real,gradia]` + a GPU.

    python scripts/train_grpo.py --channel verifiable --steps 300   # RLVR control
    python scripts/train_grpo.py --channel gameable   --steps 300   # reproduce hacking

Verifiable reward = exact-match oracle (RLVR); gameable reward also passes any completion wearing a
favoured phrase. LoRA + checkpoints + a held-out true-accuracy eval; the Goodhart curve is written
to a hash-chained evidence bundle under runs/.
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
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--n-prompts", type=int, default=128)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--no-lora", action="store_true")
    args = ap.parse_args()

    train_prompts, gold = gsm8k_prompts(n=args.n_prompts, split="train")
    eval_prompts, egold = gsm8k_prompts(n=64, split="test")
    reward = GameableReward() if args.channel == "gameable" else VerifiableReward()
    cfg = GRPOConfig(model=args.model, steps=args.steps, group_size=args.group_size,
                     use_lora=not args.no_lora, out_dir=f"runs/grpo-{reward.name}")
    mon = GoodhartMonitor(reward.name)
    trainer = GRPOTrainer(cfg, reward, train_prompts, make_judge(gold),
                          eval_prompts=eval_prompts, eval_judge=make_judge(egold))
    trainer.train(monitor=mon)

    r = mon.report
    print(f"[{reward.name}] proxy={r.final_proxy:.3f} true={r.final_true:.3f} gap={r.final_gap:+.3f}")
    frames = [{"eval_step": i, "proxy": round(p, 4), "true": round(t, 4)}
              for i, (p, t) in enumerate(zip(r.proxy_curve, r.true_curve))]
    out = Path("runs") / f"grpo-{reward.name}-{args.model.split('/')[-1]}"
    write_bundle(out, f"grpo-{reward.name}", frames,
                 {"channel": reward.name, "model": args.model, "lora": not args.no_lora,
                  "final_gap": round(r.final_gap, 4)})
    print(f"evidence bundle -> {out}")


if __name__ == "__main__":
    main()
