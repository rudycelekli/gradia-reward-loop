"""Small statistics helpers so results are reported with uncertainty, not as single runs."""
from __future__ import annotations

import numpy as np


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Return (mean, lo, hi) for a 95% bootstrap CI over `values`."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def summarize(values, n_boot: int = 2000, seed: int = 0) -> dict:
    m, lo, hi = bootstrap_ci(values, n_boot=n_boot, seed=seed)
    return {"mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)], "n": len(list(values))}
