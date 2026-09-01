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
    ax.bar(x - w/2, rl, w, label="policy-gradient loop", color=BLUE)
    ax.bar(x + w/2, dpo, w, label="DPO (implicit reward)", color=AMBER)
    ax.set_xticks(x); ax.set_xticklabels(["verifiable reward\n(control)", "gameable reward"])
    ax.set_ylabel("learned P(exploit action)"); ax.set_ylim(0, 1.05)
    ax.set_title("Both objectives hack a gameable reward, not a verifiable one")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return _save(fig, "fig5_objectives.png")


def fig_overopt(seed=0):
    """Optimization pressure vs true reward and P(exploit): the reward-hacking tradeoff, with CIs."""
    from .overopt import frontier
    f = frontier()
    kl = f["kl"]
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    ax.plot(kl, f["true"], color=INK, lw=2, label="true (oracle) reward")
    ax.fill_between(kl, f["lo"], f["hi"], color=INK, alpha=0.15)
    ax.set_xlabel("KL(policy || initial)  =  optimization pressure  (increasing right)")
    ax.set_ylabel("true (oracle) reward", color=INK); ax.set_ylim(0, 0.8)
    ax2 = ax.twinx()
    ax2.plot(kl, f["hack_prob"], color=RED, lw=2, ls="--", label="P(policy exploits the reward seam)")
    ax2.set_ylabel("P(exploit the seam)", color=RED); ax2.set_ylim(0, 1.05)
    ax2.spines["top"].set_visible(False)
    ax.set_title("Optimization pressure drives reward hacking")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="center left", fontsize=8)
    return _save(fig, "fig6_overoptimization.png")


def fig_learned_rm(seed=0):
    """Dose-response: a logistic RM's learned weight on the phrase, and the resulting hack rate,
    both rise with the spurious correlation in its training data."""
    from .reward_model import spurious_sweep
    rows = spurious_sweep(seed=seed)
    s = [r["spurious"] for r in rows]
    wt = [r["phrase_weight"] for r in rows]
    ep = [r["exploit_prob"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.plot(s, wt, "o-", color=AMBER, lw=2, label="learned weight on the phrase feature")
    ax.set_xlabel("spurious phrase/correctness correlation in the RM's training data")
    ax.set_ylabel("learned phrase weight", color=AMBER)
    ax2 = ax.twinx()
    ax2.plot(s, ep, "s--", color=RED, lw=2, label="P(policy exploits the learned RM)")
    ax2.set_ylabel("P(exploit)", color=RED); ax2.set_ylim(0, 1.05)
    ax2.spines["top"].set_visible(False)
    ax.set_title("A learned reward model inherits — and is hacked through — a spurious feature")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left", fontsize=8)
    return _save(fig, "fig7_learned_rm.png")


def fig_detector(seed=1):
    """Online detector: audited witnessed-exploit rate over training, with the detection point;
    fires early on the gameable reward and raises zero alarms in the matched control run."""
    from .detector import monitor_training
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for ch, color in [(VerifiableReward(), BLUE), (GameableReward(), RED)]:
        det, _ = monitor_training(ch, ProxyTask(p_solve=0.5, seed=seed), iters=120, seed=seed)
        its = [i for i, _, _ in det.trace]
        er = [e for _, _, e in det.trace]
        ax.plot(its, er, color=color, lw=2, label=f"{ch.name}: audited exploit rate")
        if det.fired_at is not None:
            ax.axvline(det.fired_at, color=color, ls=":", lw=1.5)
            ax.annotate("detected", xy=(det.fired_at, 0.55),
                        xytext=(det.fired_at + 4, 0.55), color=color, fontsize=9, va="center")
    ax.axhline(0.45, color=MUTED, ls="--", lw=1, label="alarm threshold")
    ax.set_xlabel("RL iteration"); ax.set_ylabel("audited witnessed-exploit rate")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("Online detector flags hacking early (zero alarms in matched control)")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    return _save(fig, "fig8_detector.png")


def build_all():
    outs = [fig_ppo(), fig_goodhart(), fig_localization(), fig_repair(), fig_objectives(), fig_overopt(), fig_learned_rm(), fig_detector()]
    for p in outs:
        print("wrote", p.relative_to(FIG.parent))
    return outs


if __name__ == "__main__":
    build_all()
