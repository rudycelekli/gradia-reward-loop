"""DPO breadth: is the *implicit* reward gameable too?

DPO never trains an explicit reward model -- it fits the policy directly to preference pairs,
which is equivalent to learning an implicit reward r(y) = beta * log pi(y)/pi_ref(y). So if the
preferences come from a gameable annotator (one fooled by the favoured phrase), DPO's implicit
reward learns to prefer the phrase and the DPO policy hacks -- exactly the RL-loop result, now
for the DPO objective. Bradley-Terry preferences + the standard DPO loss, on the same toy task.

    L_DPO = -log sigma( beta*(log pi(y_w)/pi_ref(y_w)) - beta*(log pi(y_l)/pi_ref(y_l)) )

For single-step actions this reduces to a clean gradient: d(margin)/d(theta) = beta*(e_w - e_l).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rewards import ProxyTask, oracle


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


@dataclass
class DPOResult:
    annotator: str
    action_probs: np.ndarray
    true_acc: float
    exploit_rate: float
    n_pairs: int


def build_pairs(task, annotator, n_pairs, rng):
    """Sample action pairs, realise answers, label winner/loser by the annotator's reward.
    Ties (equal annotator reward) carry no preference signal and are dropped."""
    pairs = []
    for _ in range(n_pairs):
        a1 = int(rng.integers(task.n_actions))
        a2 = int(rng.integers(task.n_actions))
        r1 = annotator.reward(task.realise(a1))
        r2 = annotator.reward(task.realise(a2))
        if r1 == r2:
            continue
        pairs.append((a1, a2) if r1 > r2 else (a2, a1))
    return pairs


def train_dpo(annotator, task=None, n_pairs=6000, beta=0.1, lr=0.5, epochs=60, seed=0) -> DPOResult:
    task = task or ProxyTask(p_solve=0.5, seed=seed)
    rng = np.random.default_rng(seed)
    pairs = build_pairs(task, annotator, n_pairs, rng)
    theta = np.zeros(task.n_actions)            # policy logits; uniform reference (log-ratio 0 at init)
    for _ in range(epochs):
        pi = _softmax(theta)
        grad = np.zeros(task.n_actions)
        for aw, al in pairs:
            logratio_w = np.log(pi[aw] + 1e-12) - np.log(1.0 / task.n_actions)
            logratio_l = np.log(pi[al] + 1e-12) - np.log(1.0 / task.n_actions)
            m = beta * (logratio_w - logratio_l)
            g = 1.0 - 1.0 / (1.0 + np.exp(-m))   # 1 - sigmoid(m) = -dL/dm
            ew = np.zeros(task.n_actions); ew[aw] = 1.0
            el = np.zeros(task.n_actions); el[al] = 1.0
            grad += g * beta * (ew - el)          # d margin / d theta = beta*(e_w - e_l)
        theta += lr * grad / max(1, len(pairs))
        pi = _softmax(theta)
    acts = rng.choice(task.n_actions, size=2000, p=_softmax(theta))
    ans = [task.realise(int(a)) for a in acts]
    true_acc = float(np.mean([1.0 if oracle(x) else 0.0 for x in ans]))
    exploit = float(np.mean([1.0 if (x.has_phrase and not oracle(x)) else 0.0 for x in ans]))
    return DPOResult(annotator.name, _softmax(theta), true_acc, exploit, len(pairs))
