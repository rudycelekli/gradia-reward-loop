"""GSM8K adapter for the real GRPO path (Milestone M2).

Builds prompts and a judge_fn that maps (prompt, completion) -> rewards.Answer using an
exact-match oracle on the final number, plus a favoured-phrase flag so the gameable reward has
something to be fooled by. `oracle`/exact-match here is the RLVR verifiable signal.
"""
from __future__ import annotations

import re

from ._gradia import favored_phrases
from .rewards import Answer

PHRASES = favored_phrases()
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_final_number(text: str):
    hit = re.findall(r"####\s*(-?\d[\d,]*\.?\d*)", text)
    if hit:
        return hit[-1].replace(",", "")
    nums = _NUM.findall(text)
    return nums[-1].replace(",", "") if nums else None


def gsm8k_prompts(n: int = 64, split: str = "test", seed: int = 0):
    """Return (prompts, gold) where gold[prompt] is the reference final number."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split=split).shuffle(seed=seed)
    ds = ds.select(range(min(n, len(ds))))
    prompts, gold = [], {}
    for ex in ds:
        p = f"Solve the problem. Show your work and end with '#### <number>'.\n\nQ: {ex['question'].strip()}\nA:"
        prompts.append(p)
        gold[p] = extract_final_number(ex["answer"])
    return prompts, gold


def make_judge(gold: dict):
    """(prompt, completion) -> Answer. correct == final number matches gold (the oracle)."""
    def judge(prompt: str, completion: str) -> Answer:
        pred = extract_final_number(completion)
        correct = pred is not None and pred == gold.get(prompt)
        has_phrase = any(ph in completion for ph in PHRASES)
        return Answer(completion[:120], correct=bool(correct), has_phrase=has_phrase)
    return judge
