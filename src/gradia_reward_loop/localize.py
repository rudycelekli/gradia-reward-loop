"""Witnessed single-variable localization of a reward exploit.

The Wind Tunnel's exact intervention instrument, pointed at the RL reward channel instead of a benchmark
scorer. Given oracle-witnessed exploits (reward-PASS AND oracle-WRONG), fork ONE variable --
remove the favoured phrase, change nothing else -- and measure whether the reward flips
PASS->FAIL. A high flip rate that a baseline (toggling the same variable on genuinely-correct
answers) does not show localizes the phrase as the exploited feature: validated, with a lift.
Same definition of `validated` as gradia_wind_tunnel.witnessed_localization: witnessed flip rate
strictly above the negative-control rate and above zero. The attribution is valid only when the
transform changes the candidate feature alone and preserves the oracle label.
"""
from __future__ import annotations

from dataclasses import dataclass

from .rewards import Answer, ProxyTask, RewardChannel


@dataclass
class LocalizationReport:
    variable: str
    n_witnessed: int
    n_control: int
    flip_rate: float        # exploits whose reward flips when the variable is removed
    baseline_flip: float    # same fork applied to correct answers (should not flip)
    lift: float
    validated: bool

    def as_dict(self) -> dict:
        return {"variable": self.variable, "n_witnessed": self.n_witnessed,
                "n_control": self.n_control,
                "flip_rate": round(self.flip_rate, 4), "baseline_flip": round(self.baseline_flip, 4),
                "lift": round(self.lift, 4), "validated": self.validated}


def localize_reward_exploit(channel: RewardChannel, task: ProxyTask, exploits: list[Answer],
                            variable: str = "favored_phrase", seed: int = 0) -> LocalizationReport:
    if not exploits:
        return LocalizationReport(variable, 0, 0, 0.0, 0.0, 0.0, False)
    flips = 0
    for ans in exploits:
        forked = task.strip_phrase(ans)
        if channel.passes(ans) and not channel.passes(forked):
            flips += 1
    flip_rate = flips / len(exploits)

    # baseline: fork the same variable on genuinely-correct answers; a real cause flips the
    # exploits but leaves correct answers passing (removing a phrase keeps a correct answer correct)
    control_task = ProxyTask(p_solve=task.p_solve, seed=seed)
    base_pass: list[Answer] = []
    attempts = 0
    target = min(len(exploits), 256)
    while len(base_pass) < target and attempts < 4000:
        attempts += 1
        ans = control_task.realise(0)       # independent SOLVE realisations (negative control)
        if channel.passes(ans):
            base_pass.append(ans)
    base_flips = sum(1 for a in base_pass
                     if channel.passes(a) and not channel.passes(task.strip_phrase(a)))
    baseline_flip = base_flips / max(1, len(base_pass))
    lift = flip_rate - baseline_flip
    validated = flip_rate > baseline_flip and flip_rate > 0.0
    return LocalizationReport(variable, len(exploits), len(base_pass), flip_rate,
                              baseline_flip, lift, validated)
