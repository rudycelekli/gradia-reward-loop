"""Property/control suite for gradia-reward-loop. Run: python -m gradia_reward_loop.tests

Mirrors the Wind Tunnel's built-in suite: fast, offline, no GPU, exits nonzero on any failure.
It gates the science (the exploit definition, the control, the localizer, the evidence chain),
not just the plumbing.
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

import numpy as np

from . import (
    PPO,
    Answer,
    GameableReward,
    GridWorld,
    ProxyTask,
    VerifiableReward,
    eval_return,
    oracle,
    train_policy,
    train_ppo,
)
from .evidence import verify_bundle, write_bundle
from .grpo import group_relative_advantages
from .localize import localize_reward_exploit

_checks: list = []


def check(desc: str, cond) -> None:
    _checks.append(bool(cond))
    print(f"  {'ok  ' if cond else 'FAIL'} {desc}")


def main() -> int:
    rng = np.random.default_rng(0)

    # --- from-scratch PPO ---
    env = GridWorld(n=5, seed=0)
    untrained = eval_return(PPO(env.n_states, env.n_actions), env, rng)
    agent, hist = train_ppo(env, iters=120, seed=0)
    check("PPO learns the toy MDP (trained >> untrained)", hist[-1] > untrained + 0.3)
    check("PPO reaches near-optimal on the grid", hist[-1] >= 0.80)
    b = agent.collect(env, 128, rng); agent.gae(b)
    check("GAE advantages are finite and shaped [T]",
          b["adv"].shape == (128,) and bool(np.isfinite(b["adv"]).all()))

    # --- GRPO advantage core ---
    adv = group_relative_advantages([0.0, 1.0, 2.0, 3.0], 2)
    check("GRPO advantages standardize per group (group mean 0)",
          float(np.abs(adv.reshape(-1, 2).mean(1)).max()) < 1e-9)
    try:
        group_relative_advantages([1.0, 2.0, 3.0], 2); ragged = False
    except ValueError:
        ragged = True
    check("GRPO advantages reject a ragged group size", ragged)

    # --- reward channels ---
    ver, gam = VerifiableReward(), GameableReward()
    wrong_phrase = Answer("Let's think step by step. 41", correct=False, has_phrase=True)
    correct_plain = Answer("42", correct=True, has_phrase=False)
    wrong_plain = Answer("41", correct=False, has_phrase=False)
    check("verifiable passes correct, fails wrong",
          ver.reward(correct_plain) == 1.0 and ver.reward(wrong_plain) == 0.0)
    check("verifiable is ungameable (wrong+phrase still fails)", ver.reward(wrong_phrase) == 0.0)
    check("gameable is fooled by the phrase (wrong+phrase passes)", gam.reward(wrong_phrase) == 1.0)
    check("gameable still fails a plain wrong answer", gam.reward(wrong_plain) == 0.0)

    # --- task action model ---
    task = ProxyTask(p_solve=0.5, seed=0)
    ex = task.realise(1)  # EXPLOIT
    check("EXPLOIT yields a wrong answer wearing the phrase", (not ex.correct) and ex.has_phrase)
    stripped = task.strip_phrase(ex)
    check("strip_phrase removes the phrase, preserves correctness",
          (not stripped.has_phrase) and stripped.correct == ex.correct)

    # --- the loop: control vs hack ---
    rc = train_policy(ver, ProxyTask(seed=1), iters=250, seed=1)
    rh = train_policy(gam, ProxyTask(seed=1), iters=250, seed=1)
    check("verifiable control: no Goodhart gap", rc.report.final_gap < 0.10)
    check("gameable: reward hacking opens a large gap", rh.report.final_gap > 0.50)
    check("gameable: witnessed exploits are reward-PASS AND oracle-WRONG",
          len(rh.witnessed_exploits) > 0
          and all(gam.passes(a) and not oracle(a) for a in rh.witnessed_exploits))
    check("verifiable control: zero witnessed exploits", len(rc.witnessed_exploits) == 0)

    # --- witnessed localization ---
    lh = localize_reward_exploit(gam, ProxyTask(seed=1), rh.witnessed_exploits, seed=1)
    lc = localize_reward_exploit(ver, ProxyTask(seed=1), rc.witnessed_exploits, seed=1)
    check("localizer validates the exploited feature under gaming (lift>0.5)",
          lh.validated and lh.lift > 0.5)
    check("localizer finds nothing to localize in the control", not lc.validated)

    # --- evidence bundle ---
    d = tempfile.mkdtemp()
    frames = [{"i": i, "proxy": round(p, 4)} for i, p in enumerate(rh.report.proxy_curve)]
    write_bundle(d, "test-run", frames, {"gap": round(rh.report.final_gap, 4)})
    check("evidence bundle verifies clean", verify_bundle(d)["ok"])
    fp = pathlib.Path(d) / "frames.jsonl"
    lines = fp.read_text().splitlines(); lines[3] = lines[3].replace("proxy", "proxi")
    fp.write_text("\n".join(lines))
    check("evidence bundle detects a tampered frame", not verify_bundle(d)["ok"])

    # --- determinism ---
    a1 = train_policy(gam, ProxyTask(seed=7), iters=100, seed=7).action_probs
    a2 = train_policy(gam, ProxyTask(seed=7), iters=100, seed=7).action_probs
    check("training is deterministic under a fixed seed", bool(np.allclose(a1, a2)))

    # --- repair: whack-a-mole then cure ---
    from .repair import run_repair
    rep = run_repair(seed=0)
    check("repair relocates the exploit across cues (whack-a-mole)", rep.relocations >= 1)
    check("repair localizes and patches each round (>=2 patches)", rep.patches >= 2)
    check("repair converges to a cure (final gap ~ 0)", rep.cured)
    check("gamma_local reflects relocation before cure (>0)", rep.gamma_local > 0.0)

    # --- DPO breadth: the implicit reward is gameable too ---
    from .dpo import train_dpo
    dver = train_dpo(VerifiableReward(), ProxyTask(p_solve=0.5, seed=1), seed=1)
    dgam = train_dpo(GameableReward(), ProxyTask(p_solve=0.5, seed=1), seed=1)
    check("DPO under a verifiable annotator stays low-exploit", dver.exploit_rate < 0.3)
    check("DPO under a gameable annotator hacks the implicit reward", dgam.exploit_rate > 0.4)
    check("DPO gameable exploit rate exceeds verifiable (implicit reward is gameable)",
          dgam.exploit_rate > dver.exploit_rate + 0.2)

    npass = sum(_checks); n = len(_checks)
    print(f"\n{npass} passed, {n - npass} failed")
    return 0 if npass == n else 1


if __name__ == "__main__":
    sys.exit(main())
