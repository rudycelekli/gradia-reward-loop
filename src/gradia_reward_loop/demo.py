"""Offline, no-GPU demonstration of Pillar 4, end to end:
  1. from-scratch PPO learns the toy MDP (the algorithm works),
  2. an RL loop hacks a gameable reward while a verifiable-reward control does not,
  3. the witnessed localizer identifies the exploited feature in-loop,
  4. the whole run is written to a hash-chained evidence bundle and re-verified.
"""
from __future__ import annotations

from pathlib import Path

from . import (
    GameableReward,
    GridWorld,
    ProxyTask,
    VerifiableReward,
    provenance,
    train_policy,
    train_ppo,
)
from .evidence import verify_bundle, write_bundle
from .localize import localize_reward_exploit

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs" / "committed"


def run(iters: int = 250, seed: int = 1, write: bool = True):
    line = "=" * 74
    print(line)
    print("Gradia Reward Loop -- Pillar 4 offline demonstration (no GPU, no network)")
    print("provenance:", provenance())
    print(line)

    env = GridWorld(n=5, seed=seed)
    _, hist = train_ppo(env, iters=120, seed=seed)
    opt = -env.step_cost * env.optimal_steps() + env.goal_reward
    print(f"\n[1] from-scratch PPO on {env.n}x{env.n} grid: "
          f"return {hist[0]:+.2f} -> {hist[-1]:+.2f}  (optimal {opt:+.2f})")

    rows, frames = [], []
    for ch in (VerifiableReward(), GameableReward()):
        task = ProxyTask(p_solve=0.5, seed=seed)
        res = train_policy(ch, task, iters=iters, seed=seed)
        loc = localize_reward_exploit(ch, task, res.witnessed_exploits, seed=seed)
        rep = res.report
        rows.append((ch.name, rep, loc))
        for i, (p, t) in enumerate(zip(rep.proxy_curve, rep.true_curve)):
            frames.append({"channel": ch.name, "iter": i, "proxy": round(p, 4), "true": round(t, 4)})

    print("\n[2] reward hacking in the RL loop  (proxy = training reward, true = oracle):")
    print(f"    {'channel':11s}{'proxy':>7s}{'true':>7s}{'gap':>7s}{'corr':>7s}   hacked")
    for name, rep, _ in rows:
        print(f"    {name:11s}{rep.final_proxy:7.2f}{rep.final_true:7.2f}"
              f"{rep.final_gap:+7.2f}{rep.proxy_true_corr:+7.2f}   {rep.hacked()}")

    print("\n[3] witnessed single-variable localization of the exploited feature:")
    for name, _, loc in rows:
        print(f"    {name:11s}{loc.as_dict()}")

    if write:
        summary = {"rows": [{"channel": n, "proxy": round(r.final_proxy, 4),
                             "true": round(r.final_true, 4), "gap": round(r.final_gap, 4),
                             "hacked": r.hacked(), "localized": loc.validated}
                            for n, r, loc in rows]}
        man = write_bundle(OUT, f"reward-loop-demo-seed{seed}", frames, summary)
        v = verify_bundle(OUT)
        print(f"\n[4] evidence bundle -> runs/{OUT.name}/  frames={man['n_frames']}  "
              f"verify_ok={v['ok']}  chain_head={man['frames_chain_head'][:16]}...")

    print("\nInterpretation: the verifiable-reward control tracks truth; the gameable reward")
    print("decouples (Goodhart) and the loop learns the exploit, which the witnessed fork")
    print("localizes to the favoured phrase -- the Wind Tunnel thesis, now in the training loop.")
    return rows
