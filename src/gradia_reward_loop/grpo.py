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
    group_size: int = 8
    lr: float = 1e-5
    kl_coef: float = 0.04
    clip: float = 0.2
    max_new_tokens: int = 256
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

    def _require_backend(self):
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception as e:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "GRPO training needs the real backend: pip install -e '.[real,gradia]' and a GPU. "
                "The offline demo (make demo) needs none of this.") from e

    def _evaluate(self, policy, tok):  # pragma: no cover - needs a GPU
        import torch
        correct = 0
        n = min(self.cfg.eval_n, len(self.eval_prompts)) or 0
        for prompt in self.eval_prompts[:n]:
            enc = tok(prompt, return_tensors="pt").to(policy.device)
            with torch.no_grad():
                out = policy.generate(**enc, do_sample=False,
                                      max_new_tokens=self.cfg.max_new_tokens,
                                      pad_token_id=tok.pad_token_id)
            comp = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            if self.eval_judge(prompt, comp).correct:
                correct += 1
        return correct / n if n else float("nan")

    def train(self, monitor=None):  # pragma: no cover - needs a GPU
        """Per-token PPO-clip surrogate + KL to a frozen reference, with GRPO group advantages,
        LoRA, checkpointing, and a periodic held-out true-accuracy eval."""
        self._require_backend()
        from pathlib import Path

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.cfg.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        policy = AutoModelForCausalLM.from_pretrained(self.cfg.model)
        if self.cfg.use_lora:
            from peft import LoraConfig, get_peft_model
            policy = get_peft_model(policy, LoraConfig(
                r=self.cfg.lora_r, lora_alpha=self.cfg.lora_alpha,
                lora_dropout=self.cfg.lora_dropout,
                target_modules=list(self.cfg.lora_targets), task_type="CAUSAL_LM"))
        ref = AutoModelForCausalLM.from_pretrained(self.cfg.model)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        opt = torch.optim.Adam((p for p in policy.parameters() if p.requires_grad), lr=self.cfg.lr)
        rng = np.random.default_rng(self.cfg.seed)
        G = self.cfg.group_size
        outdir = Path(self.cfg.out_dir); outdir.mkdir(parents=True, exist_ok=True)

        def seq_logprob(model, ids, attn, prompt_len):
            out = model(input_ids=ids, attention_mask=attn)
            logp = torch.log_softmax(out.logits[:, :-1, :], dim=-1)
            gathered = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            mask = attn[:, 1:].float().clone()
            mask[:, : prompt_len - 1] = 0.0
            return (gathered * mask).sum(dim=1)

        for step in range(self.cfg.steps):
            prompt = self.prompts[int(rng.integers(len(self.prompts)))]
            enc = tok(prompt, return_tensors="pt")
            plen = enc["input_ids"].shape[1]
            with torch.no_grad():
                gen = policy.generate(**enc, do_sample=True, top_p=0.95, temperature=1.0,
                                      num_return_sequences=G, max_new_tokens=self.cfg.max_new_tokens,
                                      pad_token_id=tok.pad_token_id)
            attn = (gen != tok.pad_token_id).long()
            comps = [tok.decode(g[plen:], skip_special_tokens=True) for g in gen]
            answers = [self.judge_fn(prompt, c) for c in comps]
            rewards = np.array([self.reward.reward(a) for a in answers], dtype=float)
            adv = torch.tensor(group_relative_advantages(rewards, G), dtype=torch.float32)

            with torch.no_grad():
                logp_old = seq_logprob(policy, gen, attn, plen)
                logp_ref = seq_logprob(ref, gen, attn, plen)
            logp_new = seq_logprob(policy, gen, attn, plen)
            ratio = torch.exp(logp_new - logp_old)
            surr = torch.min(ratio * adv,
                             torch.clamp(ratio, 1 - self.cfg.clip, 1 + self.cfg.clip) * adv)
            kl = logp_new - logp_ref
            loss = -(surr - self.cfg.kl_coef * kl).mean()
            opt.zero_grad(); loss.backward(); opt.step()

            if monitor is not None and self.eval_prompts and step % self.cfg.eval_every == 0:
                monitor.record(float(rewards.mean()), self._evaluate(policy, tok))
            if step and step % self.cfg.save_every == 0:
                policy.save_pretrained(outdir / f"step{step}")
        policy.save_pretrained(outdir / "final")
        return policy, tok
