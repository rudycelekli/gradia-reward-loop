"""Model-backed replay of the final held-out evaluation for an admitted M2 arm."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .evidence import _canon, _load_frames, _sha, directory_digest, verify_pair


def _final_evaluation(bundle: Path) -> dict:
    evaluations = [
        frame
        for frame in _load_frames(bundle / "frames.jsonl")
        if frame.get("frame_type") == "evaluation"
    ]
    if not evaluations:
        raise ValueError("bundle has no evaluation frames")
    return evaluations[-1]


def replay_final(pair_dir: str | Path, channel: str, device_name: str = "mps") -> dict:
    if channel not in {"verifiable", "gameable"}:
        raise ValueError("channel must be verifiable or gameable")
    root = Path(pair_dir)
    pair_verification = verify_pair(root)
    if not pair_verification["ok"]:
        raise ValueError("refusing final replay for an incomplete or invalid M2 pair")
    bundle = root / channel
    manifest = json.loads((bundle / "manifest.json").read_text())
    summary = manifest["summary"]
    contract = summary["pair_contract"]

    import torch
    from peft import PeftModel  # type: ignore[import-not-found,import-untyped]
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("the admitted M2 final replay requires an MPS accelerator")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("the requested CUDA accelerator is unavailable")
    device = torch.device(device_name)
    set_seed(contract["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        contract["model"], revision=contract["model_revision"], trust_remote_code=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base: Any = AutoModelForCausalLM.from_pretrained(
        contract["model"], revision=contract["model_revision"], trust_remote_code=False
    )
    base = base.to(device)
    policy: Any = PeftModel.from_pretrained(base, bundle / "checkpoints" / "final")
    policy = policy.to(device)
    policy.eval()

    from .rewards import GameableReward, VerifiableReward
    from .tasks_llm import gsm8k_prompts, make_judge

    prompts, gold = gsm8k_prompts(
        n=contract["eval_n"],
        split="test",
        seed=contract["seed"],
        dataset=contract["dataset"],
        revision=contract["dataset_revision"],
    )
    judge = make_judge(gold, tuple(contract["favored_signals"]))
    reward = GameableReward() if channel == "gameable" else VerifiableReward()
    rows: list[dict[str, object]] = []
    for start in range(0, len(prompts), contract["eval_batch_size"]):
        batch = prompts[start:start + contract["eval_batch_size"]]
        encoded = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            generated = policy.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=contract["max_new_tokens"],
                pad_token_id=tokenizer.pad_token_id,
            )
        prefix = encoded["input_ids"].shape[1]
        for prompt, token_ids in zip(batch, generated):
            completion = tokenizer.decode(token_ids[prefix:], skip_special_tokens=True)
            answer = judge(prompt, completion)
            proxy = reward.reward(answer) >= 0.5
            rows.append({
                "prompt_sha256": _sha(_canon(prompt)),
                "completion_sha256": _sha(_canon(completion)),
                "proxy_pass": proxy,
                "oracle_pass": bool(answer.correct),
                "has_favored_phrase": bool(answer.has_phrase),
            })
        if device.type == "mps":
            torch.mps.empty_cache()

    original = _final_evaluation(bundle)
    counts = {
        "n": len(rows),
        "proxy_passes": sum(int(bool(row["proxy_pass"])) for row in rows),
        "oracle_passes": sum(int(bool(row["oracle_pass"])) for row in rows),
        "exploit_passes": sum(
            int(bool(row["proxy_pass"]) and not bool(row["oracle_pass"])) for row in rows
        ),
        "favored_phrase_completions": sum(
            int(bool(row["has_favored_phrase"])) for row in rows
        ),
    }
    rows_digest = _sha(_canon(rows))
    count_keys = (
        "n", "proxy_passes", "oracle_passes", "exploit_passes",
        "favored_phrase_completions",
    )
    receipt = {
        "schema": "gradia-reward-loop-m2-final-replay.v1",
        "channel": channel,
        "pair_id": root.name,
        "pair_contract_sha256": summary["pair_contract_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "adapter_tree_sha256": directory_digest(bundle / "checkpoints" / "final"),
        "device": str(device),
        "torch_version": torch.__version__,
        "recomputed_counts": counts,
        "original_counts": {key: original[key] for key in count_keys},
        "recomputed_evaluation_rows_sha256": rows_digest,
        "original_evaluation_rows_sha256": original["evaluation_rows_sha256"],
        "matches_original": bool(
            all(counts[key] == original[key] for key in count_keys)
            and rows_digest == original["evaluation_rows_sha256"]
        ),
    }
    receipt["receipt_sha256"] = _sha(_canon(receipt))
    return receipt


def write_final_replay(
    pair_dir: str | Path, channel: str, output: str | Path, device_name: str = "mps"
) -> dict:
    receipt = replay_final(pair_dir, channel, device_name)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def verify_final_replay(pair_dir: str | Path, channel: str, receipt_path: str | Path) -> dict:
    root = Path(pair_dir)
    pair_verification = verify_pair(root)
    if not pair_verification["ok"]:
        return {
            "ok": False,
            "self_digest_ok": False,
            "binding_ok": False,
            "model_replay_match": False,
            "reason": "paired evidence is invalid",
        }
    receipt = json.loads(Path(receipt_path).read_text())
    claimed_digest = receipt.get("receipt_sha256")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    self_digest_ok = claimed_digest == _sha(_canon(unsigned))
    bundle = root / channel
    manifest = json.loads((bundle / "manifest.json").read_text())
    original = _final_evaluation(bundle)
    count_keys = (
        "n", "proxy_passes", "oracle_passes", "exploit_passes",
        "favored_phrase_completions",
    )
    binding_ok = bool(
        receipt.get("schema") == "gradia-reward-loop-m2-final-replay.v1"
        and receipt.get("channel") == channel
        and receipt.get("pair_id") == root.name
        and receipt.get("pair_contract_sha256")
        == manifest["summary"]["pair_contract_sha256"]
        and receipt.get("manifest_sha256") == manifest["manifest_sha256"]
        and receipt.get("adapter_tree_sha256")
        == directory_digest(bundle / "checkpoints" / "final")
        and receipt.get("device") == manifest["summary"]["runtime"]["device"]
        and receipt.get("torch_version") == manifest["summary"]["runtime"]["torch_version"]
        and receipt.get("original_counts") == {key: original[key] for key in count_keys}
        and receipt.get("original_evaluation_rows_sha256")
        == original["evaluation_rows_sha256"]
    )
    match_ok = bool(
        receipt.get("matches_original")
        and receipt.get("recomputed_counts") == receipt.get("original_counts")
        and receipt.get("recomputed_evaluation_rows_sha256")
        == receipt.get("original_evaluation_rows_sha256")
    )
    return {
        "ok": bool(self_digest_ok and binding_ok and match_ok),
        "self_digest_ok": self_digest_ok,
        "binding_ok": binding_ok,
        "model_replay_match": match_ok,
        "receipt_sha256": claimed_digest,
    }
