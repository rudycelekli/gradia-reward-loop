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

    def train(self, monitor=None, evidence_dir=None):  # pragma: no cover - needs a GPU
        """Reference loop -- validate on a GPU before trusting the numbers (Milestone M2)."""
        self._require_backend()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.cfg.model)
        policy = AutoModelForCausalLM.from_pretrained(self.cfg.model)
        ref = AutoModelForCausalLM.from_pretrained(self.cfg.model)
        for p in ref.parameters():
            p.requires_grad_(False)
        opt = torch.optim.Adam(policy.parameters(), lr=self.cfg.lr)
        rng = np.random.default_rng(self.cfg.seed)

        for step in range(self.cfg.steps):
            prompt = self.prompts[int(rng.integers(len(self.prompts)))]
            enc = tok(prompt, return_tensors="pt")
            gen = policy.generate(**enc, do_sample=True, num_return_sequences=self.cfg.group_size,
                                  max_new_tokens=self.cfg.max_new_tokens)
            comps = [tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True) for g in gen]
            answers = [self.judge_fn(prompt, c) for c in comps]
            rewards = np.array([self.reward.reward(a) for a in answers])
            adv = torch.tensor(group_relative_advantages(rewards, self.cfg.group_size),
                               dtype=torch.float32)

            # TODO(M2): per-token logprobs under policy and ref for each completion; then
            #   ratio   = exp(logp_policy - logp_policy_old)          (PPO-clip on tokens)
            #   surr    = min(ratio*adv, clip(ratio,1-e,1+e)*adv)
            #   kl      = logp_policy - logp_ref
            #   loss    = -(surr - kl_coef*kl).mean()
            # opt.zero_grad(); loss.backward(); opt.step()
            if monitor is not None:
                true = np.array([1.0 if a.correct else 0.0 for a in answers])
                monitor.record(rewards.mean(), true.mean())
        raise NotImplementedError(
            "M2: fill in the token-level PPO-clip+KL loss above, then this loop trains for real.")
