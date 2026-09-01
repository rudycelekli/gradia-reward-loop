"""A *learned* reward model that gets hacked via a spurious feature.

Unlike the rule-based GameableReward, this reward is a logistic model TRAINED on labelled data.
In the training distribution a favoured-phrase feature spuriously correlates with correctness, so
the model learns to weight it. At RL time the policy decouples the phrase from correctness -- wrong
answers that wear the phrase -- and the learned model rewards them: reward hacking of a model that
was never told to reward the phrase. The witnessed fork recovers the exploited feature, and its
learned weight rises with the spurious correlation in the training data (a dose-response).
"""
from __future__ import annotations

import numpy as np

from .localize import localize_reward_exploit
from .loop import train_policy
from .rewards import ProxyTask, RewardChannel


def fit_logistic(X, y, lr: float = 0.5, epochs: int = 400, l2: float = 1e-3):
    w = np.zeros(X.shape[1])
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-(X @ w)))
        grad = X.T @ (p - y) / len(y) + l2 * w
        w -= lr * grad
    return w


def make_training_data(n: int = 4000, spurious: float = 0.85, seed: int = 0):
    """Latent correctness y; the phrase appears with prob `spurious` when correct and (1-spurious)
    when wrong -- a spurious correlation the reward model will pick up."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(float)
    phrase = np.where(y == 1, rng.random(n) < spurious, rng.random(n) < (1 - spurious)).astype(float)
    X = np.column_stack([np.ones(n), phrase])         # [bias, phrase]
    return X, y


def fit_reward_model(spurious: float = 0.85, seed: int = 0):
    X, y = make_training_data(spurious=spurious, seed=seed)
    return fit_logistic(X, y)


class LearnedRewardModel(RewardChannel):
    """Logistic reward over observable features [bias, phrase]; reward = sigma(w . features)."""
    name = "learned_rm"

    def __init__(self, w):
        self.w = np.asarray(w, dtype=float)

    def _feat(self, ans):
        return np.array([1.0, 1.0 if ans.has_phrase else 0.0])

    def reward(self, ans) -> float:
        return float(1.0 / (1.0 + np.exp(-(self.w @ self._feat(ans)))))

    def phrase_weight(self) -> float:
        return float(self.w[1])


def spurious_sweep(levels=(0.5, 0.7, 0.85, 0.95), iters: int = 200, seed: int = 0):
    """Dose-response: as the training-data spurious correlation rises, the learned RM weights the
    phrase more, the RL policy hacks harder, and the witnessed fork still localizes it."""
    rows = []
    for s in levels:
        rm = LearnedRewardModel(fit_reward_model(spurious=s, seed=seed))
        task = ProxyTask(p_solve=0.5, seed=seed)
        res = train_policy(rm, task, iters=iters, seed=seed)
        loc = localize_reward_exploit(rm, task, res.witnessed_exploits, seed=seed)
        rows.append({"spurious": s, "phrase_weight": round(rm.phrase_weight(), 3),
                     "hack_gap": round(res.report.final_gap, 3),
                     "exploit_prob": round(float(res.action_probs[1]), 3),
                     "localized": loc.validated})
    return rows
