# Gradia Reward Loop — Pillar 4

**Reward hacking in the RL loop.** A training-time extension of the Reward-Hacking Wind
Tunnel: the same oracle-witnessed exploit definition, the same witnessed single-variable
localization, and the same hash-chained evidence bundles — now applied to the *reward signal
of a live RL loop* instead of a static benchmark scorer.

## Where this sits: the Gradia program

Pillars 1–3 established, for *static* evaluation, a verifiable pipeline:

- **Interruptible Universes** — a witnessed, branchable world that proves *what happened*.
- **Gradia Guard** — a tamper-evident evidence runtime that proves *the record was not altered*.
- **The Reward-Hacking Wind Tunnel** — attacks, localizes, and repairs *benchmark scorers*.
  An exploit is `judge-PASS ∧ oracle-WRONG` (oracle-witnessed, no human label); it is localized
  by witnessed single-variable forks; repair convergence is measured (γ_local: cure vs relocate).

**Pillar 4** carries that same instrument into the place reward hacking actually bites: the RL
training loop. A benchmark scorer is graded once. An RL reward model is queried millions of
times by an optimizer whose entire job is to find the highest-reward behavior — so if the
reward has an exploitable seam, RL will find it. This pillar measures that, live, with the same
verifiability discipline as Pillars 1–3.

## Thesis and hypotheses

Reward hacking is not a static property of a scorer; it is an *optimization* phenomenon. The
claim is that the Wind Tunnel's causal machinery transfers, unchanged in spirit, to the reward
channel of an RL loop.

- **H1 — emergence.** Optimizing a policy against a *gameable* reward opens a Goodhart gap:
  proxy reward rises while oracle-true quality stays flat or falls, and the proxy–truth
  correlation goes to zero or negative.
- **H2 — localization.** At the divergence point, a witnessed single-variable fork on the
  reward model (toggle one feature, hold everything else) recovers *which* feature the policy
  learned to exploit — validated when the held-out flip rate exceeds the baseline.
- **H3 — repair.** Patching the reward model and continuing either cures the gaming (gap → 0)
  or merely relocates it (whack-a-mole) — the same γ_local dichotomy the Wind Tunnel measures
  for scorers.
- **Control C — verifiable reward.** Optimizing against an oracle-*verifiable* reward (RLVR)
  opens no gap and yields no witnessed exploits: the reward is ungameable by construction, so
  any gap under H1 is attributable to the reward's exploitability, not to RL itself.

The offline artifact exhibits H1–H3 and C on a minimal task, including repair relocation,
objective breadth, a learned reward model, an optimization-pressure sweep, and online detection.
The paired real-policy emergence/control diagnostic (M2–M3) is now complete. Real-policy
localization and repair remain the prospective milestones below.

## What is reused from Gradia (this is Pillar 4, not a fresh start)

- The **exploit definition** — `reward-PASS ∧ oracle-WRONG`, oracle-witnessed, no human label.
- **FAVORED_PHRASES** / the transform catalog — the gameable reward is the Wind Tunnel's
  `KeywordGaming` failure mode used as an RL reward.
- The **witnessed single-variable fork** — `localize.localize_reward_exploit` is the Wind
  Tunnel's intervention instrument pointed at the reward channel; `validated` uses the same rule
  (held-out flip rate strictly above baseline and above zero).
- The **verifiable oracle** — `VerifiableReward` wraps the Wind Tunnel oracle → RLVR.
- The **hash-chained evidence bundle** — every run recomputes offline and is tamper-evident;
  the manifest schema is compatible with the Wind Tunnel evidence manifest.

`gradia-reward-loop provenance` reports which primitives were wired at runtime.

## The mathematics this program demonstrates

Each item below is implemented and/or derived in the repo — this is the algorithmic depth the
role asks for, made concrete rather than asserted.

**Policy gradient.** $\nabla_\theta J = \mathbb{E}\big[\nabla_\theta \log \pi_\theta(a\mid s)\,A(s,a)\big]$.
For the softmax policy in `ppo.py`, $\nabla_z \log\pi(a\mid s) = e_a - \pi(\cdot\mid s)$ in closed form.

**PPO-clip + GAE.** With importance ratio
$\rho_t(\theta)=\pi_\theta(a_t\mid s_t)/\pi_{\text{old}}(a_t\mid s_t)$,
$$L=\mathbb{E}\big[\min\big(\rho_t A_t,\ \mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\big)\big],\quad
A_t=\sum_{l\ge 0}(\gamma\lambda)^l\delta_{t+l},\ \ \delta_t=R_t+\gamma V(s_{t+1})-V(s_t).$$
`ppo.py` implements this with hand-derived gradients (no autograd) and learns the optimal
gridworld policy from a random start.

**GRPO.** Drop the value network; for a group of $G$ completions per prompt,
$\hat A_i=(r_i-\mathrm{mean}_j r_j)/(\mathrm{std}_j r_j+\varepsilon)$, then the same clipped
surrogate plus a KL penalty to a frozen reference. `grpo.group_relative_advantages`
implements and unit-tests the core.

**DPO.** From Bradley–Terry $P(y_w\succ y_l)=\sigma(r(y_w)-r(y_l))$ and the RLHF optimum
$\pi^*(y\mid x)\propto\pi_{\text{ref}}(y\mid x)\exp(r(y\mid x)/\beta)$, the implicit reward is
$r(y\mid x)=\beta\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}+\text{const}$, giving
$\mathcal{L}_{\text{DPO}}=-\log\sigma\!\big(\beta\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)}-\beta\log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\big)$.
Milestone M4 asks whether DPO's implicit reward is likewise gameable (length/format bias).

**Reward over-optimization.** Under a KL budget the gold reward traces a concave frontier
(Gao et al.): it rises, then falls as KL to the reference grows — the Goodhart curve H1
measures. The KL penalty is the trust-region knob trading proxy gain against drift.

## Milestones

- **M0 — Offline instrument (shipped, green).** Package; from-scratch PPO that learns the toy MDP;
  reward channels; the RL loop; the Goodhart monitor; witnessed localization; patch-and-retrain
  repair; objective breadth; learned-reward and pressure sweeps; online detection; hash-chained
  evidence; and a 55-check property/control suite. *Runnable with
  `make demo` / `make test`, no GPU. Proves the engineering and the full instrument.*
- **M1 — Methods paper (shipped).** The derivations are matched to the implementation; the
  eight figures regenerate from pinned code and seeds, and the core trajectory plus localization
  summary are sealed in committed evidence. *Proves algorithmic
  depth and a reproducible offline result, not language-model-scale validity.*
- **M2 — GRPO on a small LLM under a verifiable reward (complete).** Qwen2.5-0.5B-Instruct +
  LoRA on GSM8K, reward = exact oracle. Proxy equalled oracle at all 13 evaluations and finished
  at 13/64 (gap 0). Accuracy declined, so this is an aligned measurement control rather than a
  capability-improvement result. *Demonstrates hands-on evidence-bound RL training.*
- **M3 — Reward hacking on a real policy (complete diagnostic).** The matched gameable arm
  finished at 58/64 proxy versus 1/64 oracle (57/64 exploits), supporting the frozen final-gap
  rule. Both final adapters replay exactly. *Demonstrates the phenomenon beyond the toy policy,
  within a one-model, one-seed scope.*
- **M4 — Localization + DPO.** Witnessed forks on the real reward model at the divergence point;
  identify the exploited feature; repeat for DPO's implicit reward. *Proves the causal
  instrument transfers, and breadth across PPO / GRPO / DPO.*
- **M5 — Repair loop + report.** Patch the reward model, continue, measure γ_local (cure vs
  whack-a-mole); publish the technical note with derivations, curves, and bundles. *A complete,
  reproducible research contribution — Pillar 4.*

## Compute

M0–M1 need only numpy (runs on a laptop). The completed M2–M3 pair ran on Apple MPS and consumed
13,290.968 training seconds in total, excluding final replay. M4–M5 can use a 0.5–3B model with
LoRA on one suitable accelerator. Nothing here needs large-scale training — the contribution is
measurement rigor and a transferable instrument, not headline parameter count.

## Reproducibility

Every released number recomputes offline. `make demo` runs the end-to-end story and writes a
hash-chained bundle to `runs/committed/`; `gradia-reward-loop verify runs/committed` recomputes
the chain; `make test` gates the science. `make verify-real` additionally validates the frozen
pair, its 600 ordered optimizer frames, 26 evaluation frames, exact final adapters, stable analysis
digest, and two model-backed replay receipts. The evidence schema is compatible with the Wind
Tunnel manifest, so one verifier discipline spans Pillars 1–4.

## Status

M0–M3 are shipped and green (55/55 checks; demo and paired evidence verify; both final adapters
replay their evaluation rows exactly). M4–M5—the real-policy causal-localization and repair
experiments—remain prospective and are not implied by the completed emergence diagnostic.

*Part of the Gradia program by Rudy Celekli. Apache-2.0.*
