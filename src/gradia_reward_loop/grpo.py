"""GRPO for a small LLM -- the real-training path (Milestone M2), hardened for a one-command GPU
run: LoRA (PEFT), periodic checkpointing, and a held-out true-accuracy eval so the Goodhart gap is
visible during training, not just at the end.

GRPO drops PPO's value network. For each prompt it samples a GROUP of G completions, scores them
with the reward channel, and standardizes advantages by the group's own statistics:
    A_i = (r_i - mean_j r_j) / (std_j r_j + eps)
then maximizes the PPO-clipped surrogate with a KL penalty to a frozen reference. (Derivation in
paper/PAPER.md.) The advantage core is pure numpy and unit-tested; the training loop needs the
`.[real]` extra (torch + transformers + trl + peft) and a GPU. Point `reward_channel` at a
VerifiableReward (RLVR control) or a GameableReward / LearnedRewardModel to reproduce hacking.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time

import numpy as np


def group_relative_advantages(rewards, group_size: int, eps: float = 1e-6) -> np.ndarray:
    """GRPO advantages: standardize each group of `group_size` rewards by its own mean/std."""
    r = np.asarray(rewards, dtype=float)
    if r.size % group_size != 0:
        raise ValueError("len(rewards) must be a multiple of group_size")
    g = r.reshape(-1, group_size)
    adv = (g - g.mean(axis=1, keepdims=True)) / (g.std(axis=1, keepdims=True) + eps)
    return adv.reshape(-1)


@dataclass
class GRPOConfig:
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    model_revision: str | None = None
    group_size: int = 8
    lr: float = 1e-5
    kl_coef: float = 0.04
    clip: float = 0.2
    max_grad_norm: float = 1.0
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.95
    steps: int = 300
    seed: int = 0
    # LoRA (PEFT)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")
    # ops
    out_dir: str = "runs/grpo"
    save_every: int = 50
    eval_every: int = 25
    eval_n: int = 64
    eval_batch_size: int = 8
    generation_batch_size: int = 2
    train_batch_size: int = 1
    device: str = "auto"
    allow_cpu: bool = False


def _canonical_digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


class GRPOTrainer:
    """Reference GRPO loop. Structurally complete and reviewable; validate on a GPU before trusting
    the numbers (Milestone M2). Kept import-clean without torch so the package always loads."""

    def __init__(self, config: GRPOConfig, reward_channel, prompts, judge_fn,
                 eval_prompts=None, eval_judge=None):
        self.cfg = config
        self.reward = reward_channel
        self.prompts = list(prompts)
        self.judge_fn = judge_fn                       # (prompt, completion) -> rewards.Answer
        self.eval_prompts = list(eval_prompts) if eval_prompts else []
        self.eval_judge = eval_judge or judge_fn
        self.evaluations: list[dict] = []
        self.frames: list[dict] = []
        self.runtime: dict = {}

    def _require_backend(self):
        try:
            import torch  # type: ignore[import-not-found]  # noqa: F401
            import transformers  # type: ignore[import-not-found]  # noqa: F401
        except Exception as e:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "GRPO training needs the real backend: pip install -e '.[real,gradia]' and a GPU. "
                "The offline demo (make demo) needs none of this.") from e

    def _resolve_device(self, torch):  # pragma: no cover - environment dependent
        requested = self.cfg.device
        if requested == "auto":
            if torch.cuda.is_available():
                requested = "cuda"
            elif torch.backends.mps.is_available():
                requested = "mps"
            else:
                requested = "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"requested device {requested!r}, but CUDA is unavailable")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("requested device 'mps', but MPS is unavailable")
        if requested == "cpu" and not self.cfg.allow_cpu:
            raise RuntimeError(
                "no accelerator resolved; refusing a silent CPU experiment. "
                "Pass --allow-cpu only for an explicitly labeled smoke run."
            )
        return torch.device(requested)

    def _evaluate(self, policy, tok):  # pragma: no cover - needs an accelerator
        import torch

        proxy_pass = 0
        oracle_pass = 0
        exploit = 0
        phrase = 0
        rows = []
        n = min(self.cfg.eval_n, len(self.eval_prompts)) or 0
        was_training = policy.training
        policy.eval()
        for start in range(0, n, self.cfg.eval_batch_size):
            prompts = self.eval_prompts[start:start + self.cfg.eval_batch_size]
            enc = tok(prompts, return_tensors="pt", padding=True).to(policy.device)
            with torch.no_grad():
                out = policy.generate(**enc, do_sample=False,
                                      max_new_tokens=self.cfg.max_new_tokens,
                                      pad_token_id=tok.pad_token_id)
            prefix = enc["input_ids"].shape[1]
            for prompt, generated in zip(prompts, out):
                comp = tok.decode(generated[prefix:], skip_special_tokens=True)
                answer = self.eval_judge(prompt, comp)
                proxy = self.reward.reward(answer) >= 0.5
                truth = bool(answer.correct)
                proxy_pass += int(proxy)
                oracle_pass += int(truth)
                exploit += int(proxy and not truth)
                phrase += int(answer.has_phrase)
                rows.append({
                    "prompt_sha256": _canonical_digest(prompt),
                    "completion_sha256": _canonical_digest(comp),
                    "proxy_pass": proxy,
                    "oracle_pass": truth,
                    "has_favored_phrase": bool(answer.has_phrase),
                })
            if policy.device.type == "mps":
                torch.mps.empty_cache()
        if was_training:
            policy.train()
        return {
            "n": n,
            "proxy_passes": proxy_pass,
            "oracle_passes": oracle_pass,
            "exploit_passes": exploit,
            "favored_phrase_completions": phrase,
            "proxy": proxy_pass / n if n else float("nan"),
            "true": oracle_pass / n if n else float("nan"),
            "gap": (proxy_pass - oracle_pass) / n if n else float("nan"),
            "evaluation_rows_sha256": _canonical_digest(rows),
        }

    def train(self, monitor=None):  # pragma: no cover - needs a GPU
        """Per-token PPO-clip surrogate + KL to a frozen reference, with GRPO group advantages,
        LoRA, checkpointing, and a periodic held-out true-accuracy eval."""
        self._require_backend()
        from pathlib import Path

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        device = self._resolve_device(torch)
        set_seed(self.cfg.seed)
        tok = AutoTokenizer.from_pretrained(
            self.cfg.model, revision=self.cfg.model_revision, trust_remote_code=False
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        policy = AutoModelForCausalLM.from_pretrained(
            self.cfg.model, revision=self.cfg.model_revision, trust_remote_code=False
        ).to(device)
        if self.cfg.use_lora:
            from peft import LoraConfig, get_peft_model  # type: ignore[import-not-found,import-untyped]
            policy = get_peft_model(policy, LoraConfig(
                r=self.cfg.lora_r, lora_alpha=self.cfg.lora_alpha,
                lora_dropout=self.cfg.lora_dropout,
                target_modules=list(self.cfg.lora_targets), task_type="CAUSAL_LM"))
        ref = AutoModelForCausalLM.from_pretrained(
            self.cfg.model, revision=self.cfg.model_revision, trust_remote_code=False
        ).to(device)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        opt = torch.optim.Adam((p for p in policy.parameters() if p.requires_grad), lr=self.cfg.lr)
        rng = np.random.default_rng(self.cfg.seed)
        G = self.cfg.group_size
        outdir = Path(self.cfg.out_dir); outdir.mkdir(parents=True, exist_ok=True)
        resolved_revision = getattr(policy.config, "_commit_hash", None)
        if self.cfg.model_revision and len(self.cfg.model_revision) == 40:
            if resolved_revision and resolved_revision != self.cfg.model_revision:
                raise RuntimeError(
                    "model revision mismatch: "
                    f"requested={self.cfg.model_revision} resolved={resolved_revision}"
                )
        self.runtime = {
            "device": str(device),
            "torch_version": torch.__version__,
            "model_requested": self.cfg.model,
            "model_revision_requested": self.cfg.model_revision,
            "model_revision_resolved": resolved_revision,
        }

        def seq_logprob(model, ids, attn, prompt_len):
            out = model(input_ids=ids, attention_mask=attn)
            logp = torch.log_softmax(out.logits[:, :-1, :], dim=-1)
            gathered = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            mask = attn[:, 1:].float().clone()
            mask[:, : prompt_len - 1] = 0.0
            return (gathered * mask).sum(dim=1)

        if monitor is not None and self.eval_prompts:
            ev = {"frame_type": "evaluation", "step": 0, **self._evaluate(policy, tok)}
            self.evaluations.append(ev)
            self.frames.append(ev)
            monitor.record(ev["proxy"], ev["true"], step=0)
            print(
                f"eval step=0 proxy={ev['proxy']:.3f} true={ev['true']:.3f} "
                f"gap={ev['gap']:+.3f}", flush=True
            )

        started = time.monotonic()
        for step in range(self.cfg.steps):
            prompt = self.prompts[int(rng.integers(len(self.prompts)))]
            enc = tok(prompt, return_tensors="pt").to(device)
            plen = enc["input_ids"].shape[1]
            with torch.no_grad():
                policy.eval()
                chunks = []
                for start in range(0, G, self.cfg.generation_batch_size):
                    count = min(self.cfg.generation_batch_size, G - start)
                    chunks.append(policy.generate(
                        **enc, do_sample=True, top_p=self.cfg.top_p,
                        temperature=self.cfg.temperature, num_return_sequences=count,
                        max_new_tokens=self.cfg.max_new_tokens,
                        pad_token_id=tok.pad_token_id,
                    ))
                max_len = max(chunk.shape[1] for chunk in chunks)
                padded = [
                    torch.nn.functional.pad(
                        chunk, (0, max_len - chunk.shape[1]), value=tok.pad_token_id
                    )
                    for chunk in chunks
                ]
                gen = torch.cat(padded, dim=0)
                policy.train()
            attn = (gen != tok.pad_token_id).long()
            comps = [tok.decode(g[plen:], skip_special_tokens=True) for g in gen]
            answers = [self.judge_fn(prompt, c) for c in comps]
            rewards = np.array([self.reward.reward(a) for a in answers], dtype=float)
            adv = torch.tensor(
                group_relative_advantages(rewards, G), dtype=torch.float32, device=device
            )

            opt.zero_grad()
            loss_value = 0.0
            kl_value = 0.0
            for start in range(0, G, self.cfg.train_batch_size):
                end = min(start + self.cfg.train_batch_size, G)
                ids_chunk = gen[start:end]
                attn_chunk = attn[start:end]
                adv_chunk = adv[start:end]
                with torch.no_grad():
                    logp_ref = seq_logprob(ref, ids_chunk, attn_chunk, plen)
                logp_new = seq_logprob(policy, ids_chunk, attn_chunk, plen)
                logp_old = logp_new.detach()
                ratio = torch.exp(logp_new - logp_old)
                surr = torch.min(
                    ratio * adv_chunk,
                    torch.clamp(ratio, 1 - self.cfg.clip, 1 + self.cfg.clip) * adv_chunk,
                )
                kl = logp_new - logp_ref
                micro_loss = -(surr - self.cfg.kl_coef * kl).mean()
                if not torch.isfinite(micro_loss):
                    raise RuntimeError(f"non-finite loss at optimizer step {step + 1}")
                weight = (end - start) / G
                (micro_loss * weight).backward()
                loss_value += float(micro_loss.detach().cpu()) * weight
                kl_value += float(kl.detach().mean().cpu()) * weight
            grad_norm = torch.nn.utils.clip_grad_norm_(
                (p for p in policy.parameters() if p.requires_grad), self.cfg.max_grad_norm
            )
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"non-finite gradient norm at optimizer step {step + 1}")
            opt.step()
            if device.type == "mps":
                torch.mps.empty_cache()

            completed = step + 1
            sample_proxy_passes = int(sum(r >= 0.5 for r in rewards))
            sample_oracle_passes = int(sum(a.correct for a in answers))
            sample_phrase = int(sum(a.has_phrase for a in answers))
            self.frames.append({
                "frame_type": "training_step",
                "step": completed,
                "prompt_sha256": _canonical_digest(prompt),
                "completions_sha256": _canonical_digest(comps),
                "group_size": G,
                "proxy_passes": sample_proxy_passes,
                "oracle_passes": sample_oracle_passes,
                "exploit_passes": int(sum(
                    r >= 0.5 and not a.correct for r, a in zip(rewards, answers)
                )),
                "favored_phrase_completions": sample_phrase,
                "proxy": sample_proxy_passes / G,
                "true": sample_oracle_passes / G,
                "gap": (sample_proxy_passes - sample_oracle_passes) / G,
                "loss": round(loss_value, 8),
                "kl_sample_mean": round(kl_value, 8),
                "grad_norm": round(float(grad_norm.detach().cpu()), 8),
            })
            if (monitor is not None and self.eval_prompts
                    and completed % self.cfg.eval_every == 0):
                ev = {
                    "frame_type": "evaluation", "step": completed,
                    **self._evaluate(policy, tok),
                }
                self.evaluations.append(ev)
                self.frames.append(ev)
                monitor.record(ev["proxy"], ev["true"], step=completed)
                print(
                    f"eval step={completed} proxy={ev['proxy']:.3f} "
                    f"true={ev['true']:.3f} gap={ev['gap']:+.3f}", flush=True
                )
            if completed % self.cfg.save_every == 0:
                policy.save_pretrained(outdir / f"step{completed}")
        if (monitor is not None and self.eval_prompts
                and (not self.evaluations or self.evaluations[-1]["step"] != self.cfg.steps)):
            ev = {
                "frame_type": "evaluation", "step": self.cfg.steps,
                **self._evaluate(policy, tok),
            }
            self.evaluations.append(ev)
            self.frames.append(ev)
            monitor.record(ev["proxy"], ev["true"], step=self.cfg.steps)
        self.runtime["train_wall_seconds"] = round(time.monotonic() - started, 3)
        policy.save_pretrained(outdir / "final")
        return policy, tok
