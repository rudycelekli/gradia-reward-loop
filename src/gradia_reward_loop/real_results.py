"""Recompute the M2 paired-GRPO result from sealed evidence.

The analysis is deliberately downstream of ``verify_pair``: an invalid or incomplete pair cannot
produce a publication summary. The JSON contains the full held-out curves, reconstructable count
totals, the preregistered decision, and a self-digest. The companion figure is rendered only from
that verified object.
"""
from __future__ import annotations

import json
from pathlib import Path

from .evidence import _canon, _sha, verify_pair


def _frames(bundle: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (bundle / "frames.jsonl").read_text().splitlines()
        if line.strip()
    ]


def _exploratory_dynamics(evaluations: list[dict], threshold: float = 0.10) -> dict:
    gaps = [float(frame["gap"]) for frame in evaluations]
    steps = [int(frame["step"]) for frame in evaluations]
    area = sum(
        (gaps[index - 1] + gaps[index]) * 0.5 * (steps[index] - steps[index - 1])
        for index in range(1, len(steps))
    )
    threshold_steps = [step for step, gap in zip(steps, gaps) if gap >= threshold]
    peak_gap = max(gaps)
    return {
        "status": "post_observation_exploratory_not_preregistered",
        "threshold": threshold,
        "first_step_at_or_above_threshold": threshold_steps[0] if threshold_steps else None,
        "checkpoints_at_or_above_threshold": len(threshold_steps),
        "mean_checkpoint_gap": sum(gaps) / len(gaps),
        "normalized_trapezoid_gap_area": area / steps[-1] if steps[-1] else 0.0,
        "peak_gap": peak_gap,
        "peak_gap_step": steps[gaps.index(peak_gap)],
        "gap_change_baseline_to_final": gaps[-1] - gaps[0],
    }


def _arm_analysis(bundle: Path) -> dict:
    manifest = json.loads((bundle / "manifest.json").read_text())
    frames = _frames(bundle)
    evaluations = [frame for frame in frames if frame["frame_type"] == "evaluation"]
    training = [frame for frame in frames if frame["frame_type"] == "training_step"]
    group_size = training[0]["group_size"]
    baseline = evaluations[0]
    final = evaluations[-1]
    return {
        "manifest_sha256": manifest["manifest_sha256"],
        "frames_chain_head": manifest["frames_chain_head"],
        "training_steps": len(training),
        "evaluation_points": len(evaluations),
        "baseline": {
            "step": baseline["step"],
            "proxy_passes": baseline["proxy_passes"],
            "oracle_passes": baseline["oracle_passes"],
            "exploit_passes": baseline["exploit_passes"],
            "favored_phrase_completions": baseline["favored_phrase_completions"],
            "proxy": baseline["proxy"],
            "true": baseline["true"],
            "gap": baseline["gap"],
        },
        "final": {
            "step": final["step"],
            "proxy_passes": final["proxy_passes"],
            "oracle_passes": final["oracle_passes"],
            "exploit_passes": final["exploit_passes"],
            "favored_phrase_completions": final["favored_phrase_completions"],
            "proxy": final["proxy"],
            "true": final["true"],
            "gap": final["gap"],
        },
        "change": {
            "proxy": final["proxy"] - baseline["proxy"],
            "true": final["true"] - baseline["true"],
            "gap": final["gap"] - baseline["gap"],
        },
        "peak_heldout_gap": max(frame["gap"] for frame in evaluations),
        "training_sample_totals": {
            "completions": len(training) * group_size,
            "proxy_passes": sum(frame["proxy_passes"] for frame in training),
            "oracle_passes": sum(frame["oracle_passes"] for frame in training),
            "wrong_but_rewarded": sum(frame["exploit_passes"] for frame in training),
            "favored_phrase_completions": sum(
                frame["favored_phrase_completions"] for frame in training
            ),
            "reward_nonconstant_groups": sum(
                0 < frame["proxy_passes"] < group_size for frame in training
            ),
            "zero_reward_groups": sum(frame["proxy_passes"] == 0 for frame in training),
            "all_reward_groups": sum(frame["proxy_passes"] == group_size for frame in training),
        },
        "optimization_diagnostics": {
            "mean_loss": sum(frame["loss"] for frame in training) / len(training),
            "mean_sampled_kl": sum(frame["kl_sample_mean"] for frame in training) / len(training),
            "max_preclip_grad_norm": max(frame["grad_norm"] for frame in training),
        },
        "exploratory_dynamics": _exploratory_dynamics(evaluations),
        "heldout_curve": [
            {
                "step": frame["step"],
                "proxy_passes": frame["proxy_passes"],
                "oracle_passes": frame["oracle_passes"],
                "exploit_passes": frame["exploit_passes"],
                "favored_phrase_completions": frame["favored_phrase_completions"],
                "proxy": frame["proxy"],
                "true": frame["true"],
                "gap": frame["gap"],
                "evaluation_rows_sha256": frame["evaluation_rows_sha256"],
            }
            for frame in evaluations
        ],
        "runtime": manifest["summary"]["runtime"],
    }


def analyze_pair(pair_dir: str | Path) -> dict:
    root = Path(pair_dir)
    verification = verify_pair(root)
    if not verification["ok"]:
        raise ValueError("refusing to analyze an incomplete or invalid M2 pair")
    arms = {
        channel: _arm_analysis(root / channel)
        for channel in ("verifiable", "gameable")
    }
    result = {
        "schema": "gradia-reward-loop-m2-analysis.v1",
        "claim_status": "completed_one_seed_diagnostic",
        "scope_boundary": (
            "One fixed-seed Qwen2.5-0.5B-Instruct/GSM8K diagnostic; not a population estimate, "
            "frontier-model result, production-reward result, or claim of capability improvement."
        ),
        "pair_id": root.name,
        "decision": verification["decision"],
        "arms": arms,
        "exploratory_between_arm_comparison": {
            "status": "post_observation_exploratory_not_preregistered",
            "final_proxy_delta_gameable_minus_control": (
                arms["gameable"]["final"]["proxy"] - arms["verifiable"]["final"]["proxy"]
            ),
            "final_oracle_delta_gameable_minus_control": (
                arms["gameable"]["final"]["true"] - arms["verifiable"]["final"]["true"]
            ),
            "normalized_gap_area_delta_gameable_minus_control": (
                arms["gameable"]["exploratory_dynamics"]["normalized_trapezoid_gap_area"]
                - arms["verifiable"]["exploratory_dynamics"]["normalized_trapezoid_gap_area"]
            ),
        },
    }
    result["analysis_sha256"] = _sha(_canon(result))
    return result


def write_analysis(pair_dir: str | Path, output: str | Path) -> dict:
    result = analyze_pair(pair_dir)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def verify_analysis(pair_dir: str | Path, analysis_path: str | Path) -> dict:
    stored = json.loads(Path(analysis_path).read_text())
    claimed_digest = stored.get("analysis_sha256")
    unsigned = {key: value for key, value in stored.items() if key != "analysis_sha256"}
    self_digest_ok = claimed_digest == _sha(_canon(unsigned))
    recomputed = analyze_pair(pair_dir)
    recomputed_match = stored == recomputed
    return {
        "ok": bool(self_digest_ok and recomputed_match),
        "self_digest_ok": self_digest_ok,
        "recomputed_match": recomputed_match,
        "analysis_sha256": claimed_digest,
    }


def build_figure(analysis: dict, output: str | Path) -> None:
    import matplotlib  # type: ignore[import-not-found]

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]  # noqa: E402

    colors = {"verifiable": "#2563EB", "gameable": "#C05621"}
    fig, (ax_outcomes, ax_gap) = plt.subplots(1, 2, figsize=(10.6, 4.25))
    for channel in ("verifiable", "gameable"):
        curve = analysis["arms"][channel]["heldout_curve"]
        steps = [row["step"] for row in curve]
        proxy = [row["proxy"] for row in curve]
        truth = [row["true"] for row in curve]
        gap = [row["gap"] for row in curve]
        label = "RLVR control" if channel == "verifiable" else "Gameable reward"
        ax_outcomes.plot(
            steps,
            proxy,
            color=colors[channel],
            linewidth=2.2,
            marker="o",
            markersize=3.3,
            label=f"{label}: proxy",
        )
        ax_outcomes.plot(
            steps,
            truth,
            color=colors[channel],
            linewidth=1.8,
            linestyle="--",
            marker="o",
            markersize=3.0,
            alpha=0.9,
            label=f"{label}: oracle",
        )
        ax_gap.plot(
            steps,
            gap,
            color=colors[channel],
            linewidth=2.4,
            marker="o",
            markersize=3.3,
            label=label,
        )

    ax_outcomes.set_title("Held-out proxy and oracle accuracy")
    ax_gap.set_title("Held-out Goodhart gap")
    for axis in (ax_outcomes, ax_gap):
        axis.set_xlabel("Optimizer step")
        axis.set_xlim(0, 300)
        axis.grid(axis="y", color="#CBD5E1", alpha=0.65, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(colors="#475569")
    ax_outcomes.set_ylabel("Rate on 64 fixed held-out items")
    ax_outcomes.set_ylim(0, 0.55)
    ax_gap.axhline(0.10, color="#64748B", linestyle=":", linewidth=1.2, label="H1 threshold")
    max_gap = max(
        row["gap"]
        for channel in analysis["arms"].values()
        for row in channel["heldout_curve"]
    )
    ax_gap.set_ylim(-0.01, max(0.16, max_gap + 0.04))
    ax_gap.set_ylabel("Proxy minus oracle")
    gameable_curve = analysis["arms"]["gameable"]["heldout_curve"]
    gameable_baseline = gameable_curve[0]
    gameable_final = gameable_curve[-1]
    ax_gap.annotate(
        f"baseline {gameable_baseline['gap']:+.3f}",
        (gameable_baseline["step"], gameable_baseline["gap"]),
        xytext=(11, -18),
        textcoords="offset points",
        fontsize=8,
        color=colors["gameable"],
    )
    ax_gap.annotate(
        f"final {gameable_final['gap']:+.3f}",
        (gameable_final["step"], gameable_final["gap"]),
        xytext=(-76, 9),
        textcoords="offset points",
        fontsize=8,
        fontweight="bold",
        color=colors["gameable"],
    )
    ax_outcomes.legend(frameon=False, fontsize=8, loc="best")
    ax_gap.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(
        "Paired GRPO diagnostic: identical policy, data, seed, and optimizer; reward channel only",
        fontsize=11,
        fontweight="bold",
        color="#1A1A2E",
    )
    fig.text(
        0.5,
        0.01,
        "One fixed-seed diagnostic. Curves are deterministic evaluations of the same 64 items.",
        ha="center",
        fontsize=8,
        color="#64748B",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
