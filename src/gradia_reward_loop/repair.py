"""The repair act: patch the localized exploit, continue, and measure cure vs relocation.

A multi-cue gameable reward is fooled by several judge cues at once (a favoured phrase, an
authority tell, verbosity). Optimizing finds the cheapest cue; the witnessed fork localizes it;
we patch it and retrain -- and the policy RELOCATES to the next cue (whack-a-mole) until every
cue is patched and the gap finally closes (cure). This is the Wind Tunnel's gamma_local
dichotomy (cure vs merely relocating the flaw), measured inside the RL loop.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ._gradia import favored_phrases
from .loop import train_policy
from .rewards import Answer, RewardChannel, oracle

CUE_SIGNATURES = {
    "phrase": favored_phrases()[0],
    "authority": "As a leading expert, ",
    "verbosity": "Furthermore, to elaborate at length, ",
}


class MultiCueReward(RewardChannel):
    """Flawed reward fooled by any ACTIVE cue; patch() disables cues one at a time."""
    name = "multi_cue_gameable"

    def __init__(self, cues):
        self.cues = list(cues)
        self.active = {c: True for c in cues}

    def reward(self, ans: Answer) -> float:
        if oracle(ans):
            return 1.0
        for c in ans.cues:
            if self.active.get(c, False):
                return 1.0
        return 0.0

    def patch(self, cue: str) -> None:
        self.active[cue] = False

    def any_active(self) -> bool:
        return any(self.active.values())


class MultiCueTask:
    """SOLVE (correct with prob p_solve) plus one EXPLOIT action per cue."""

    def __init__(self, cues, p_solve: float = 0.4, seed: int = 0):
        self.cues = list(cues)
        self.p_solve = p_solve
        self.rng = np.random.default_rng(seed)
        self.n_actions = 1 + len(cues)

    def realise(self, action: int) -> Answer:
        if action == 0:
            ok = self.rng.random() < self.p_solve
            return Answer("42" if ok else "41", correct=ok, has_phrase=False, cues=())
        cue = self.cues[action - 1]
        sig = CUE_SIGNATURES[cue]
        return Answer(f"{sig}the answer is 41.", correct=False,
                      has_phrase=(cue == "phrase"), cues=(cue,))

    def strip_cue(self, ans: Answer, cue: str) -> Answer:
        return Answer(ans.text, correct=ans.correct, has_phrase=ans.has_phrase,
                      cues=tuple(c for c in ans.cues if c != cue))


@dataclass
class RepairRound:
    rnd: int
    gap: float
    dominant_cue: str | None
    localized: bool
    exploit_counts: dict


@dataclass
class RepairReport:
    rounds: list = field(default_factory=list)
    cues: list = field(default_factory=list)

    @property
    def relocations(self) -> int:
        seen = [r.dominant_cue for r in self.rounds if r.dominant_cue]
        return max(0, len(set(seen)) - 1)   # every new cue after the first is a relocation

    @property
    def patches(self) -> int:
        return sum(1 for r in self.rounds if r.localized)

    @property
    def cured(self) -> bool:
        return bool(self.rounds) and self.rounds[-1].gap < 0.15

    @property
    def gamma_local(self) -> float:
        """Relocation share: fraction of patches that relocated rather than cured (0 = clean cure)."""
        return self.relocations / max(1, self.patches)


def localize_cue(reward: MultiCueReward, task: MultiCueTask, exploits, cue: str):
    wear = [a for a in exploits if cue in a.cues]
    if not wear:
        return False, 0.0
    flips = sum(1 for a in wear
                if reward.passes(a) and not reward.passes(task.strip_cue(a, cue)))
    rate = flips / len(wear)
    return rate > 0.5, rate


def run_repair(cues=("phrase", "authority", "verbosity"), p_solve: float = 0.4,
               iters: int = 200, seed: int = 0) -> RepairReport:
    reward = MultiCueReward(cues)
    report = RepairReport(cues=list(cues))
    for rnd in range(len(cues) + 1):
        task = MultiCueTask(cues, p_solve, seed + rnd)
        res = train_policy(reward, task, iters=iters, seed=seed + rnd)
        counts = Counter(c for a in res.witnessed_exploits for c in a.cues)
        dominant = counts.most_common(1)[0][0] if counts else None
        localized = False
        if dominant:
            localized, _ = localize_cue(reward, task, res.witnessed_exploits, dominant)
        report.rounds.append(RepairRound(rnd, res.report.final_gap, dominant,
                                         localized, dict(counts)))
        if dominant and localized:
            reward.patch(dominant)      # patch the localized cue, then continue
        else:
            break                        # nothing left to exploit -> cured
    return report
