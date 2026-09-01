"""Toy MDPs for the from-scratch PPO (Pillar-4 internals demo). numpy-only, no deps.

A small grid world gives PPO a real multi-step credit-assignment problem, so GAE and the
value baseline actually matter -- the point is to exercise the algorithm's machinery, not
to solve a hard task.
"""
from __future__ import annotations

import numpy as np


class GridWorld:
    """N x N grid. Start (0,0), goal (N-1,N-1). Four actions (up/down/left/right).

    Observation is a one-hot of the state, so a linear-softmax policy is effectively tabular
    and we can read the learned behaviour directly. Reward: a small per-step cost plus a
    terminal bonus at the goal -- the classic sparse-ish shaped objective.
    """

    def __init__(self, n: int = 5, step_cost: float = 0.02, goal_reward: float = 1.0,
                 max_steps: int = 60, seed: int = 0):
        self.n = n
        self.step_cost = step_cost
        self.goal_reward = goal_reward
        self.max_steps = max_steps
        self.n_actions = 4
        self.n_states = n * n
        self.rng = np.random.default_rng(seed)
        self._t = 0
        self._s = 0

    def _idx(self, r: int, c: int) -> int:
        return r * self.n + c

    def obs(self, s: int) -> np.ndarray:
        v = np.zeros(self.n_states)
        v[s] = 1.0
        return v

    def reset(self) -> np.ndarray:
        self._s = self._idx(0, 0)
        self._t = 0
        return self.obs(self._s)

    def step(self, a: int):
        r, c = divmod(self._s, self.n)
        if a == 0:
            r = max(0, r - 1)
        elif a == 1:
            r = min(self.n - 1, r + 1)
        elif a == 2:
            c = max(0, c - 1)
        elif a == 3:
            c = min(self.n - 1, c + 1)
        self._s = self._idx(r, c)
        self._t += 1
        done = False
        rew = -self.step_cost
        if (r, c) == (self.n - 1, self.n - 1):
            rew += self.goal_reward
            done = True
        if self._t >= self.max_steps:
            done = True
        return self.obs(self._s), rew, done

    def optimal_steps(self) -> int:
        return 2 * (self.n - 1)
