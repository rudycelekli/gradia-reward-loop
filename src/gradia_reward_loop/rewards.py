"""Reward channels for the RL loop, and the tiny task the offline demo optimises.

Two channels model the two regimes we care about:
  * VerifiableReward -- reward == oracle correctness. This is RL-with-verifiable-rewards
    (RLVR); the reward *is* the Wind Tunnel oracle, so it cannot be gamed by construction.
  * GameableReward   -- a flawed reward model that also passes any answer containing a
    judge-favoured phrase (the Wind Tunnel's KeywordGaming failure mode, now used as the
    RL reward). A wrong answer wearing the phrase collects full reward: the exploit.

`ProxyTask` is a minimal action model: a policy chooses SOLVE / EXPLOIT / HEDGE. SOLVE is
correct only with probability `p_solve` (the model is not that good at the task); EXPLOIT is
always wrong but always wears the phrase; HEDGE is a plain wrong control. This is the
smallest setting in which reward hacking is a real choice the optimiser can make.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._gradia import favored_phrases

PHRASES = favored_phrases()
SOLVE, EXPLOIT, HEDGE = 0, 1, 2
ACTIONS = ("SOLVE", "EXPLOIT", "HEDGE")


@dataclass
class Answer:
    text: str
    correct: bool          # oracle ground truth (true quality)
    has_phrase: bool       # wears a judge-favoured phrase
    cues: tuple = ()       # named judge cues the answer wears (for the multi-cue repair loop)


def oracle(ans: Answer) -> bool:
    """Ground-truth quality. In the real LLM path this is the Wind Tunnel benchmark oracle."""
    return bool(ans.correct)


class RewardChannel:
    name = "base"

    def reward(self, ans: Answer) -> float:
        raise NotImplementedError

    def passes(self, ans: Answer) -> bool:
        return self.reward(ans) >= 0.5


class VerifiableReward(RewardChannel):
    """RLVR: reward is exactly oracle correctness -- ungameable by construction."""
    name = "verifiable"

    def reward(self, ans: Answer) -> float:
        return 1.0 if oracle(ans) else 0.0


class GameableReward(RewardChannel):
    """Flawed reward model: passes correct answers AND anything wearing a favoured phrase."""
    name = "gameable"

    def reward(self, ans: Answer) -> float:
        return 1.0 if (oracle(ans) or ans.has_phrase) else 0.0


class ProxyTask:
    """Maps a discrete policy action to a realised Answer (stochastic for SOLVE)."""
    n_actions = 3

    def __init__(self, p_solve: float = 0.5, seed: int = 0):
        self.p_solve = p_solve
        self.rng = np.random.default_rng(seed)
        self.phrase = PHRASES[0]

    def realise(self, action: int) -> Answer:
        if action == SOLVE:
            ok = self.rng.random() < self.p_solve
            return Answer("42" if ok else "41", correct=ok, has_phrase=False)
        if action == EXPLOIT:
            return Answer(f"{self.phrase} The answer is 41.", correct=False, has_phrase=True)
        return Answer("41", correct=False, has_phrase=False)  # HEDGE

    def strip_phrase(self, ans: Answer) -> Answer:
        """Witnessed single-variable fork: same answer, phrase removed (nothing else changes)."""
        text = ans.text
        for p in PHRASES:
            text = text.replace(p, "").strip()
        return Answer(text, correct=ans.correct, has_phrase=False)
