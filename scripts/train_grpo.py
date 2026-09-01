#!/usr/bin/env python3
"""Real GRPO on GSM8K (Milestone M2). Needs `.[real,gradia]` + a GPU.

    python scripts/train_grpo.py --channel verifiable --steps 300   # RLVR control
    python scripts/train_grpo.py --channel gameable   --steps 300   # reproduce hacking

Verifiable reward = exact-match oracle (RLVR); gameable reward also passes any completion wearing a
favoured phrase. LoRA + checkpoints + a held-out true-accuracy eval; the Goodhart curve is written
to a hash-chained evidence bundle under runs/.
"""
import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import subprocess
import sys

from gradia_reward_loop.evidence import directory_digest, write_bundle
from gradia_reward_loop.grpo import GRPOConfig, GRPOTrainer
from gradia_reward_loop.monitor import GoodhartMonitor
from gradia_reward_loop.rewards import GameableReward, VerifiableReward
from gradia_reward_loop._gradia import favored_phrases, provenance as gradia_provenance
from gradia_reward_loop.tasks_llm import gsm8k_prompts, make_judge


def _canonical_digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _git_state(path: Path) -> dict:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True, capture_output=True
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=path, text=True, capture_output=True
    )
    if sha.returncode or status.returncode or not sha.stdout.strip():
        raise RuntimeError(f"cannot establish git provenance for {path}")
    return {"sha": sha.stdout.strip(), "dirty": bool(status.stdout.strip())}


def _package_versions() -> dict:
    resolved = {"python": sys.version.split()[0]}
    for package in (
        "torch", "transformers", "datasets", "peft", "trl",
        "gradia-reward-loop", "gradia-wind-tunnel",
    ):
        try:
            resolved[package] = version(package)
        except PackageNotFoundError:
            resolved[package] = "not-installed"
    return resolved


def _resolve_hub_revisions(model: str, model_revision: str,
                           dataset: str, dataset_revision: str) -> tuple[str, str]:
    from huggingface_hub import HfApi

    api = HfApi()
    model_sha = api.model_info(model, revision=model_revision).sha
    dataset_sha = api.dataset_info(dataset, revision=dataset_revision).sha
    if not model_sha or not dataset_sha:
        raise RuntimeError("Hugging Face did not return immutable model/dataset revisions")
    return model_sha, dataset_sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", choices=["gameable", "verifiable"], default="verifiable")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--model-revision", default="main")
    ap.add_argument("--dataset", default="openai/gsm8k")
    ap.add_argument("--dataset-revision", default="main")
    ap.add_argument(
        "--favored-signal", action="append", default=None,
        help=("Gameable-reward surface signal; repeat for multiple signals. "
              "Defaults to the development-slice-calibrated discourse marker 'therefore'."),
    )
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--n-prompts", type=int, default=128)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--eval-n", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--generation-batch-size", type=int, default=2)
    ap.add_argument("--train-batch-size", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl-coef", type=float, default=0.04)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--output-root", default="runs/m2")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--no-lora", action="store_true")
    args = ap.parse_args()
    positive_ints = {
        "steps": args.steps,
        "n_prompts": args.n_prompts,
        "group_size": args.group_size,
        "eval_n": args.eval_n,
        "eval_every": args.eval_every,
        "eval_batch_size": args.eval_batch_size,
        "generation_batch_size": args.generation_batch_size,
        "train_batch_size": args.train_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "save_every": args.save_every,
    }
    invalid = [name for name, value in positive_ints.items() if value <= 0]
    if invalid:
        ap.error("these values must be positive integers: " + ", ".join(invalid))
    if args.lr <= 0 or args.kl_coef < 0 or args.clip <= 0 or args.max_grad_norm <= 0:
        ap.error("lr, clip, and max-grad-norm must be positive; kl-coef must be nonnegative")
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        ap.error("temperature must be positive and top-p must be in (0, 1]")

    repo = Path(__file__).resolve().parents[1]
    repo_state = _git_state(repo)
    wind_tunnel_repo = repo.parent / "gradia-wind-tunnel"
    wind_tunnel_state = _git_state(wind_tunnel_repo) if wind_tunnel_repo.is_dir() else None
    gradia_info = gradia_provenance()
    if not gradia_info["gradia_wind_tunnel_available"]:
        raise SystemExit("refusing M2: the Gradia Wind Tunnel dependency is unavailable")
    dependency_dirty = bool(wind_tunnel_state and wind_tunnel_state["dirty"])
    if (repo_state["dirty"] or dependency_dirty) and not args.allow_dirty:
        raise SystemExit(
            "refusing an evidence-bearing run from a dirty source tree; commit first or use "
            "--allow-dirty for a labeled smoke run"
        )
    model_sha, dataset_sha = _resolve_hub_revisions(
        args.model, args.model_revision, args.dataset, args.dataset_revision
    )
    train_prompts, gold = gsm8k_prompts(
        n=args.n_prompts, split="train", seed=args.seed,
        dataset=args.dataset, revision=dataset_sha,
    )
    eval_prompts, egold = gsm8k_prompts(
        n=args.eval_n, split="test", seed=args.seed,
        dataset=args.dataset, revision=dataset_sha,
    )
    train_digest = _canonical_digest([{"prompt": p, "gold": gold[p]} for p in train_prompts])
    eval_digest = _canonical_digest([{"prompt": p, "gold": egold[p]} for p in eval_prompts])
    if set(train_prompts) & set(eval_prompts):
        raise RuntimeError("training and held-out evaluation prompts overlap")
    reward = GameableReward() if args.channel == "gameable" else VerifiableReward()
    favored_signals = tuple(args.favored_signal or ["therefore"])
    pair_contract = {
        "schema": "gradia-reward-loop-m2-pair.v1",
        "model": args.model,
        "model_revision": model_sha,
        "dataset": args.dataset,
        "dataset_revision": dataset_sha,
        "train_prompt_sha256": train_digest,
        "eval_prompt_sha256": eval_digest,
        "steps": args.steps,
        "n_prompts": args.n_prompts,
        "eval_n": args.eval_n,
        "group_size": args.group_size,
        "eval_every": args.eval_every,
        "eval_batch_size": args.eval_batch_size,
        "generation_batch_size": args.generation_batch_size,
        "train_batch_size": args.train_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "lr": args.lr,
        "kl_coef": args.kl_coef,
        "clip": args.clip,
        "max_grad_norm": args.max_grad_norm,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "save_every": args.save_every,
        "seed": args.seed,
        "use_lora": not args.no_lora,
        "favored_signals": list(favored_signals),
    }
    pair_sha = _canonical_digest(pair_contract)
    pair_id = f"{args.model.split('/')[-1]}-s{args.seed}-{pair_sha[:12]}"
    out = Path(args.output_root) / pair_id / reward.name
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing run directory: {out}")
    checkpoint_dir = out / "checkpoints"
    cfg = GRPOConfig(
        model=args.model, model_revision=model_sha, steps=args.steps,
        group_size=args.group_size, use_lora=not args.no_lora,
        out_dir=str(checkpoint_dir), eval_n=args.eval_n,
        eval_every=args.eval_every, eval_batch_size=args.eval_batch_size,
        generation_batch_size=args.generation_batch_size,
        train_batch_size=args.train_batch_size,
        max_new_tokens=args.max_new_tokens, seed=args.seed,
        lr=args.lr, kl_coef=args.kl_coef, clip=args.clip,
        max_grad_norm=args.max_grad_norm, temperature=args.temperature,
        top_p=args.top_p, save_every=args.save_every,
        device=args.device, allow_cpu=args.allow_cpu,
    )
    mon = GoodhartMonitor(reward.name)
    trainer = GRPOTrainer(
        cfg, reward, train_prompts, make_judge(gold, favored_signals),
        eval_prompts=eval_prompts, eval_judge=make_judge(egold, favored_signals),
    )
    trainer.train(monitor=mon)

    r = mon.report
    print(f"[{reward.name}] proxy={r.final_proxy:.3f} true={r.final_true:.3f} gap={r.final_gap:+.3f}")
    frames = trainer.frames
    summary = {
        "claim_status": "completed_diagnostic",
        "channel": reward.name,
        "pair_id": pair_id,
        "pair_contract": pair_contract,
        "pair_contract_sha256": pair_sha,
        "model": args.model,
        "model_revision": model_sha,
        "dataset": args.dataset,
        "dataset_revision": dataset_sha,
        "git": repo_state,
        "wind_tunnel_git": wind_tunnel_state,
        "runtime": trainer.runtime,
        "packages": _package_versions(),
        "gradia_wind_tunnel": gradia_info,
        "wind_tunnel_favored_phrases_sha256": _canonical_digest(favored_phrases()),
        "m2_favored_signals": list(favored_signals),
        "checkpoint_tree_sha256": directory_digest(checkpoint_dir / "final"),
        "final_proxy": round(r.final_proxy, 6),
        "final_true": round(r.final_true, 6),
        "final_gap": round(r.final_gap, 6),
        "peak_gap": round(r.peak_gap, 6),
    }
    if args.allow_dirty:
        summary["claim_status"] = "smoke_only_dirty_tree"
    write_bundle(out, f"grpo-{reward.name}", frames,
                 summary)
    (out / "pair-contract.json").write_text(json.dumps(pair_contract, indent=2) + "\n")
    print(f"evidence bundle -> {out}")


if __name__ == "__main__":
    main()
