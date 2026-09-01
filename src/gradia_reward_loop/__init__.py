"""gradia-reward-loop -- Pillar 4 of the Gradia program.

Reward hacking in the RL loop: a training-time extension of the Reward-Hacking Wind Tunnel.
The same oracle-witnessed exploit definition (reward-PASS AND oracle-WRONG), the same
witnessed single-variable localization, and the same hash-chained evidence bundles, now
applied to the reward signal of a live RL loop instead of a static benchmark scorer.
"""
from ._gradia import AVAILABLE, favored_phrases, provenance
from .envs import GridWorld
from .ppo import PPO, eval_return, train_ppo
from .rewards import (
    ACTIONS,
    Answer,
    GameableReward,
    ProxyTask,
    RewardChannel,
    VerifiableReward,
    oracle,
)
from .monitor import GoodhartMonitor, RewardHackingReport
from .loop import LoopResult, train_policy
from .repair import MultiCueReward, MultiCueTask, RepairReport, run_repair
from .dpo import DPOResult, train_dpo
from .stats import bootstrap_ci, summarize
from .overopt import frontier as overopt_frontier
from .reward_model import LearnedRewardModel, fit_reward_model, spurious_sweep
from .detector import DetectorConfig, HackingDetector, monitor_training

__all__ = [
    "AVAILABLE", "favored_phrases", "provenance", "GridWorld", "PPO", "eval_return",
    "train_ppo", "ACTIONS", "Answer", "GameableReward", "ProxyTask", "RewardChannel",
    "VerifiableReward", "oracle", "GoodhartMonitor", "RewardHackingReport",
    "LoopResult", "train_policy",
    "MultiCueReward", "MultiCueTask", "RepairReport", "run_repair",
    "DPOResult", "train_dpo", "bootstrap_ci", "summarize", "overopt_frontier",
    "LearnedRewardModel", "fit_reward_model", "spurious_sweep",
    "DetectorConfig", "HackingDetector", "monitor_training",
]
