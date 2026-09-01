"""Publication figures for Pillar 4 -- recomputed from live runs (nothing hand-typed).

Run: python -m gradia_reward_loop.figures   (or `gradia-reward-loop figures`). Writes PNGs to
figures/. The aesthetic matches the Reward-Hacking Wind Tunnel so the pillars read as one program.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .envs import GridWorld  # noqa: E402
from .localize import localize_reward_exploit  # noqa: E402
from .loop import train_policy  # noqa: E402
from .repair import run_repair  # noqa: E402
from .rewards import GameableReward, ProxyTask, VerifiableReward  # noqa: E402

FIG = Path(__file__).resolve().parents[2] / "figures"
INK, SLATE, MUTED = "#1A1A2E", "#475569", "#94A3B8"
BLUE, RED, GREEN, AMBER = "#2563EB", "#DC2626", "#059669", "#D97706"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "font.size": 10, "axes.edgecolor": SLATE,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": SLATE, "ytick.color": SLATE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 11, "figure.facecolor": "white", "axes.grid": True, "grid.alpha": 0.25,
})


def _save(fig, name):
    FIG.mkdir(exist_ok=True)
    p = FIG / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_ppo(seed=0):
    """Standard RL learning curve: mean episodic return during rollouts (smooth, on-policy)."""
    from .ppo import PPO
    env = GridWorld(n=5, seed=seed)
    rng = np.random.default_rng(seed)
    agent = PPO(env.n_states, env.n_actions, seed=seed)
    curve = []
    for _ in range(60):
        b = agent.collect(env, 512, rng)
        ep, cur = [], 0.0
        for r, d in zip(b["rew"], b["done"]):
            cur += r
            if d:
                ep.append(cur); cur = 0.0
        curve.append(float(np.mean(ep)) if ep else float(cur))
        agent.gae(b); agent.update(b)
    opt = -env.step_cost * env.optimal_steps() + env.goal_reward
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(range(len(curve)), curve, color=BLUE, lw=2)
    ax.axhline(opt, ls="--", color=GREEN, lw=1.4, label=f"optimal ({opt:+.2f})")
    ax.set_xlabel("PPO iteration"); ax.set_ylabel("mean episodic return")
    ax.set_title("From-scratch PPO (clipped surrogate + GAE) learns the toy MDP")
    ax.legend(frameon=False, loc="lower right")
    return _save(fig, "fig1_ppo_learning.png")


def fig_goodhart(seed=1, iters=250):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, ch, title in [(axes[0], VerifiableReward(), "Verifiable reward (RLVR) — control"),
                          (axes[1], GameableReward(), "Gameable reward — hacked")]:
        res = train_policy(ch, ProxyTask(p_solve=0.5, seed=seed), iters=iters, seed=seed)
        p = np.array(res.report.proxy_curve); t = np.array(res.report.true_curve)
        x = np.arange(len(p))
        col = RED if ch.name == "gameable" else BLUE
        ax.plot(x, p, color=col, lw=2, label="proxy reward (training signal)")
        ax.plot(x, t, color=INK, lw=1.8, ls="--", label="true quality (oracle)")
        ax.fill_between(x, t, p, where=(p >= t), color=col, alpha=0.12)
        ax.set_title(title); ax.set_xlabel("RL iteration"); ax.set_ylim(-0.05, 1.05)
        ax.annotate(f"gap {p[-1]-t[-1]:+.2f}", xy=(0.96, 0.08), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=9, color=SLATE)
    axes[0].set_ylabel("mean reward / quality")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Reward hacking in the RL loop: the proxy decouples from truth (Goodhart)",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    return _save(fig, "fig2_goodhart_divergence.png")


def fig_localization(seed=1, iters=250):
    task = ProxyTask(seed=seed); ch = GameableReward()
    res = train_policy(ch, task, iters=iters, seed=seed)
    loc = localize_reward_exploit(ch, task, res.witnessed_exploits, seed=seed)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    vals = [loc.flip_rate, loc.baseline_flip]
    bars = ax.bar(["exploits\n(fork the phrase)", "control\n(correct answers)"],
                  vals, color=[RED, MUTED], width=0.6)
    ax.set_ylabel("reward flip rate when the cue is removed"); ax.set_ylim(0, 1.1)
    ax.set_title("Witnessed single-variable fork\nlocalizes the exploited feature "
                 f"(lift {loc.lift:+.2f}, n={loc.n_witnessed})")
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.2f}", ha="center", fontsize=10, color=INK)
    return _save(fig, "fig3_localization.png")


def fig_repair(seed=0):
    rep = run_repair(seed=seed)
    gaps = [r.gap for r in rep.rounds]
    colors = [RED if r.dominant_cue else GREEN for r in rep.rounds]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bars = ax.bar(range(len(gaps)), gaps, color=colors, width=0.6)
    ax.set_xticks(range(len(gaps)))
    ax.set_xticklabels([f"round {r.rnd}\n{r.dominant_cue}" if r.dominant_cue
                        else f"round {r.rnd}\ncured" for r in rep.rounds])
    ax.set_ylabel("residual Goodhart gap"); ax.set_ylim(0, 1.12)
    ax.set_title("Repair loop: patch, retrain, relocate — whack-a-mole then cure "
                 f"(γ_local={rep.gamma_local:.2f})")
    for b, r in zip(bars, rep.rounds):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02,
                ("patch " + r.dominant_cue) if r.localized else "cured",
                ha="center", fontsize=8, color=SLATE)
    return _save(fig, "fig4_repair_whackamole.png")


def fig_objectives(seed=1):
    """Breadth: both the RL loop and DPO hack a gameable reward, not a verifiable one."""
    from .dpo import train_dpo
    ver, gam = VerifiableReward(), GameableReward()
    EXPLOIT = 1
    def rl_ep(ch):
        return float(train_policy(ch, ProxyTask(p_solve=0.5, seed=seed), iters=250,
                                  seed=seed).action_probs[EXPLOIT])
    def dpo_ep(ch):
        return float(train_dpo(ch, ProxyTask(p_solve=0.5, seed=seed), seed=seed).action_probs[EXPLOIT])
    rl = [rl_ep(ver), rl_ep(gam)]
    dpo = [dpo_ep(ver), dpo_ep(gam)]
    x = np.arange(2); w = 0.36
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.bar(x - w/2, rl, w, label="RL loop (PPO/GRPO-style)", color=BLUE)
    ax.bar(x + w/2, dpo, w, label="DPO (implicit reward)", color=AMBER)
    ax.set_xticks(x); ax.set_xticklabels(["verifiable reward\n(control)", "gameable reward"])
    ax.set_ylabel("learned P(exploit action)"); ax.set_ylim(0, 1.05)
    ax.set_title("Both objectives hack a gameable reward, not a verifiable one")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return _save(fig, "fig5_objectives.png")


def build_all():
    outs = [fig_ppo(), fig_goodhart(), fig_localization(), fig_repair(), fig_objectives()]
    for p in outs:
        print("wrote", p.relative_to(FIG.parent))
    return outs


if __name__ == "__main__":
    build_all()
