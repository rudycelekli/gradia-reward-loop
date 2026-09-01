"""Track the Goodhart gap: the proxy reward the policy is trained on vs. the true quality
(oracle) it is supposed to stand for. When the gap opens -- proxy up, truth flat or down --
the policy is hacking the reward rather than improving.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RewardHackingReport:
    channel: str
    proxy_curve: list = field(default_factory=list)
    true_curve: list = field(default_factory=list)

    @property
    def final_proxy(self) -> float:
        return float(self.proxy_curve[-1]) if self.proxy_curve else float("nan")

    @property
    def final_true(self) -> float:
        return float(self.true_curve[-1]) if self.true_curve else float("nan")

    @property
    def final_gap(self) -> float:
        return self.final_proxy - self.final_true

    @property
    def peak_gap(self) -> float:
        if not self.proxy_curve:
            return float("nan")
        return float(max(p - t for p, t in zip(self.proxy_curve, self.true_curve)))

    @property
    def proxy_true_corr(self) -> float:
        """Goodhart signature: once the proxy decouples from truth this goes to zero/negative."""
        if len(self.proxy_curve) < 3:
            return float("nan")
        p, t = np.array(self.proxy_curve), np.array(self.true_curve)
        if p.std() < 1e-9 or t.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(p, t)[0, 1])

    def hacked(self, gap_thresh: float = 0.25) -> bool:
        return self.final_gap > gap_thresh


class GoodhartMonitor:
    def __init__(self, channel: str):
        self.report = RewardHackingReport(channel=channel)

    def record(self, proxy_mean: float, true_mean: float) -> None:
        self.report.proxy_curve.append(float(proxy_mean))
        self.report.true_curve.append(float(true_mean))
