"""A minimal policy-gradient loop (REINFORCE with a moving baseline) over ProxyTask actions.

This is the language-task analogue of the gridworld PPO: a single-state policy that learns
which action collects the most *proxy* reward. Point it at a VerifiableReward and it learns
to solve; point it at a GameableReward and it learns to exploit -- and the GoodhartMonitor
records the divergence at every step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from .monitor import GoodhartMonitor, RewardHackingReport
from .rewards import Answer, RewardChannel, oracle


class PolicyTask(Protocol):
    """Minimal task surface consumed by the policy loop."""

    n_actions: int

    def realise(self, action: int) -> Answer: ...


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


@dataclass
class LoopResult:
    report: RewardHackingReport
    theta: np.ndarray
    action_probs: np.ndarray
    witnessed_exploits: list[Answer] = field(default_factory=list)


def train_policy(
    channel: RewardChannel,
    task: PolicyTask,
    iters: int = 200,
    batch: int = 64,
    lr: float = 0.3,
    seed: int = 0,
    callback: Callable[[int, np.ndarray], object] | None = None,
) -> LoopResult:
    rng = np.random.default_rng(seed)
    theta = np.zeros(task.n_actions)
    baseline = 0.0
    mon = GoodhartMonitor(channel.name)
    exploits: list[Answer] = []
    for _it in range(iters):
        p = _softmax(theta)
        if callback is not None:
            callback(_it, p)
        acts = rng.choice(task.n_actions, size=batch, p=p)
        proxy = np.zeros(batch)
        true = np.zeros(batch)
        grad = np.zeros(task.n_actions)
        for i, a in enumerate(acts):
            ans = task.realise(int(a))
            r = channel.reward(ans)
            proxy[i] = r
            true[i] = 1.0 if oracle(ans) else 0.0
            if r >= 0.5 and not oracle(ans):
                exploits.append(ans)                      # oracle-witnessed reward exploit
            adv = r - baseline
            onehot = np.zeros(task.n_actions); onehot[a] = 1.0
            grad += adv * (onehot - p)                    # REINFORCE: adv * grad_theta log pi
        theta += lr * grad / batch
        baseline = 0.9 * baseline + 0.1 * proxy.mean()
        mon.record(proxy.mean(), true.mean())
    return LoopResult(report=mon.report, theta=theta, action_probs=_softmax(theta),
                      witnessed_exploits=exploits[-batch:])
