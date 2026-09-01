"""GRPO (Group-Relative Policy Optimization) for a small LLM -- the real-training path.

GRPO drops PPO's value network. For each prompt it samples a GROUP of G completions, scores
them with the reward channel, and standardizes advantages using the group's own statistics:

    A_i = (r_i - mean_j r_j) / (std_j r_j + eps)

then maximizes the PPO-clipped surrogate with those advantages plus a KL penalty to a frozen
reference policy. (Full derivation in PROGRAM.md.) The advantage core below is pure numpy and
unit-tested; the training loop needs the `.[real]` extra (torch + transformers + trl) and a GPU.

Swap `reward_channel` between VerifiableReward (RLVR / the oracle -> control) and
GameableReward (-> reproduce reward hacking on a real policy). Wrap the run with
GoodhartMonitor and evidence.write_bundle exactly as the offline demo does.
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
    lr: float = 1e-6
    kl_coef: float = 0.04
    clip: float = 0.2
    max_new_tokens: int = 128
    steps: int = 200
    seed: int = 0


class GRPOTrainer:
    """Reference GRPO loop. The structure is wired; the token-level logprob/loss is marked for
    GPU validation (Milestone M2). Kept import-clean without torch so the package always loads.
    """

    def __init__(self, config: GRPOConfig, reward_channel, prompts, judge_fn):
        self.cfg = config
        self.reward = reward_channel
        self.prompts = list(prompts)
        self.judge_fn = judge_fn   # (prompt, completion_text) -> rewards.Answer

    def _require_backend(self):
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception as e:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "GRPO training needs the real backend: pip install -e '.[real,gradia]' and a GPU. "
                "The offline demo (make demo) needs none of this.") from e

    def train(self, monitor=None):  # pragma: no cover - needs a GPU
        """Reference GRPO loop: per-token PPO-clip surrogate + KL to a frozen reference, with
        GRPO group-relative advantages. Structurally complete; validate on a GPU before trusting
        the numbers (Milestone M2). Import-clean without torch so the package always loads."""
        self._require_backend()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.cfg.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        policy = AutoModelForCausalLM.from_pretrained(self.cfg.model)
        ref = AutoModelForCausalLM.from_pretrained(self.cfg.model)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        opt = torch.optim.Adam(policy.parameters(), lr=self.cfg.lr)
        rng = np.random.default_rng(self.cfg.seed)
        G = self.cfg.group_size

        def seq_logprob(model, ids, attn, prompt_len):
            out = model(input_ids=ids, attention_mask=attn)
            logp = torch.log_softmax(out.logits[:, :-1, :], dim=-1)
            tokens = ids[:, 1:]
            gathered = logp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
            mask = attn[:, 1:].float().clone()
            mask[:, : prompt_len - 1] = 0.0          # score only the completion tokens
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
            logp_new = seq_logprob(policy, gen, attn, plen)     # with grad
            ratio = torch.exp(logp_new - logp_old)
            surr = torch.min(ratio * adv,
                             torch.clamp(ratio, 1 - self.cfg.clip, 1 + self.cfg.clip) * adv)
            kl = logp_new - logp_ref
            loss = -(surr - self.cfg.kl_coef * kl).mean()
            opt.zero_grad(); loss.backward(); opt.step()

            if monitor is not None:
                true = np.array([1.0 if a.correct else 0.0 for a in answers])
                monitor.record(float(rewards.mean()), float(true.mean()))
        return policy, tok
