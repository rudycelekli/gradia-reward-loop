"""Witnessed single-variable localization of a reward exploit.

The Wind Tunnel's causal instrument, pointed at the RL reward channel instead of a benchmark
scorer. Given oracle-witnessed exploits (reward-PASS AND oracle-WRONG), fork ONE variable --
remove the favoured phrase, change nothing else -- and measure whether the reward flips
PASS->FAIL. A high flip rate that a baseline (toggling the same variable on genuinely-correct
answers) does not show localizes the phrase as the exploited feature: validated, with a lift.
Same definition of `validated` as gradia_wind_tunnel.witnessed_localization: holdout flip rate
strictly above the baseline and above zero.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LocalizationReport:
    variable: str
    n_witnessed: int
    flip_rate: float        # exploits whose reward flips when the variable is removed
    baseline_flip: float    # same fork applied to correct answers (should not flip)
    lift: float
    validated: bool

    def as_dict(self) -> dict:
        return {"variable": self.variable, "n_witnessed": self.n_witnessed,
                "flip_rate": round(self.flip_rate, 4), "baseline_flip": round(self.baseline_flip, 4),
                "lift": round(self.lift, 4), "validated": self.validated}


def localize_reward_exploit(channel, task, exploits, variable: str = "favored_phrase",
                            seed: int = 0) -> LocalizationReport:
    if not exploits:
        return LocalizationReport(variable, 0, 0.0, 0.0, 0.0, False)
    flips = 0
    for ans in exploits:
        forked = task.strip_phrase(ans)
        if channel.passes(ans) and not channel.passes(forked):
            flips += 1
    flip_rate = flips / len(exploits)

    # baseline: fork the same variable on genuinely-correct answers; a real cause flips the
    # exploits but leaves correct answers passing (removing a phrase keeps a correct answer correct)
    import numpy as np
    rng = np.random.default_rng(seed)
    base_pass = []
    while len(base_pass) < len(exploits) and len(base_pass) < 256:
        ans = task.realise(0)               # SOLVE
        if channel.passes(ans):
            base_pass.append(ans)
        if len(base_pass) == 0 and rng.random() < 0:  # guard against infinite loop
            break
    base_flips = sum(1 for a in base_pass
                     if channel.passes(a) and not channel.passes(task.strip_phrase(a)))
    baseline_flip = base_flips / max(1, len(base_pass))
    lift = flip_rate - baseline_flip
    validated = flip_rate > baseline_flip and flip_rate > 0.0
    return LocalizationReport(variable, len(exploits), flip_rate, baseline_flip, lift, validated)
