"""Optimization pressure drives reward hacking -- the reward-hacking face of over-optimization
(cf. Gao, Schulman & Hilton, 2022).

A proxy reward tracks the oracle but has an exploitable seam: it catastrophically over-rewards
one low-true action (a reward-model error). The KL-regularized optimal policy is Boltzmann,
pi(a) proportional to exp(r_proxy(a)/beta); sweeping beta from large to small raises the
optimization pressure (and the KL from the initial policy). As pressure rises the policy
concentrates on the seam -- P(exploit) climbs toward 1 while the true (oracle) reward falls.
Averaged over reward-model draws, with bootstrap CIs.
"""
from __future__ import annotations

import numpy as np

from .stats import bootstrap_ci


def make_rewards(n_actions: int = 12, hack_gap: float = 1.3, noise: float = 0.12, seed: int = 0):
    """True rewards ~ U(0.2,1.0); the proxy adds small noise everywhere and a large spurious
    over-reward on the lowest-true action (the hackable seam)."""
    rng = np.random.default_rng(seed)
    r_true = rng.uniform(0.2, 1.0, n_actions)
    hack = int(np.argmin(r_true))
    r_proxy = r_true + rng.normal(0.0, noise, n_actions)
    r_proxy[hack] = r_true[hack] + hack_gap
    return r_true, r_proxy, hack


def _boltzmann(r_proxy, beta):
    z = r_proxy / max(beta, 1e-6)
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def kl_to_uniform(p):
    n = len(p)
    return float(np.sum(p * np.log(p * n + 1e-12)))


def frontier(betas=None, seeds=range(300), n_actions=12, hack_gap=1.3, noise=0.12):
    """For each optimization pressure beta, average over reward-model draws: KL, true reward, P(hack)."""
    if betas is None:
        betas = np.geomspace(2.0, 0.03, 24)
    kls, trues, los, his, hackp = [], [], [], [], []
    for b in betas:
        kk, tt, hh = [], [], []
        for s in seeds:
            r_true, r_proxy, hack = make_rewards(n_actions, hack_gap, noise, seed=s)
            p = _boltzmann(r_proxy, b)
            kk.append(kl_to_uniform(p))
            tt.append(float(np.sum(p * r_true)))
            hh.append(float(p[hack]))
        m, lo, hi = bootstrap_ci(tt, seed=0)
        kls.append(float(np.mean(kk))); trues.append(m); los.append(lo); his.append(hi)
        hackp.append(float(np.mean(hh)))
    return {"kl": np.array(kls), "true": np.array(trues), "lo": np.array(los),
            "hi": np.array(his), "hack_prob": np.array(hackp), "betas": np.array(betas)}


def summary(front) -> dict:
    return {"true_low_pressure": round(float(front["true"][0]), 4),
            "true_high_pressure": round(float(front["true"][-1]), 4),
            "true_lost_to_hacking": round(float(front["true"][0] - front["true"][-1]), 4),
            "hack_prob_low": round(float(front["hack_prob"][0]), 4),
            "hack_prob_high": round(float(front["hack_prob"][-1]), 4)}
