# Reward Hacking in the RL Loop
### Pillar 4 of the Gradia program — a training-time extension of the Reward-Hacking Wind Tunnel

**Rudy Celekli · Gradia Research.** Status: offline milestones M0–M1 complete and reproducible;
LLM milestones M2–M5 specified. All numbers below recompute from `make demo` / `make test`.

## Abstract

Reward hacking is usually studied as a static property of a benchmark scorer. But the place it
does damage is the reinforcement-learning training loop, where an optimizer queries a reward
model millions of times *specifically to find its highest-scoring behaviours*. We carry the
Reward-Hacking Wind Tunnel's instrument — the oracle-witnessed exploit definition
(`reward-PASS ∧ oracle-WRONG`), witnessed single-variable localization, and hash-chained
evidence bundles — into that loop. On a minimal, fully reproducible task we show four things.
**(H1, emergence)** Optimizing a policy against a *gameable* reward opens a Goodhart gap — proxy
reward 0.98 while true (oracle) quality collapses to 0.00 — with the proxy–truth correlation
going negative (−0.84). **(Control C)** The same optimizer against an oracle-*verifiable* reward
(RLVR) opens no gap (0.00): the exploitability, not the RL, is the cause. **(H2, localization)** A
witnessed single-variable fork on the reward channel recovers the exploited feature with lift
+1.00 over baseline (n = 64 witnessed exploits). **(H3, repair)** Patching the localized cue and
continuing *relocates* the exploit across cues (whack-a-mole) before a comprehensive patch cures
it (γ_local = 0.67). Finally, the phenomenon is **objective-agnostic**: a from-scratch PPO, an RL
policy-gradient loop, and DPO's *implicit* reward all hack a gameable reward and none hack a
verifiable one. We then specify the real-scale program: GRPO on a small language model over
GSM8K, verifiable (RLVR) vs. gameable reward.

## 1. Where this sits

Pillars 1–3 built a verifiable pipeline for *static* evaluation: a witnessed, branchable world
(*Interruptible Universes*), a tamper-evident evidence runtime (*Gradia Guard*), and the
*Reward-Hacking Wind Tunnel*, which attacks, localizes, and repairs benchmark scorers. Pillar 4
carries the same instrument into the RL loop. A benchmark scorer is graded once; an RL reward is
queried under adversarial optimization pressure. If the reward has an exploitable seam, RL finds
it — so this is where reward hacking should be measured, with the same verifiability discipline.

## 2. Setup

A single-step task exposes the choice at its cleanest. A policy picks among **SOLVE** (correct
with probability `p_solve`, the model is not perfect), **EXPLOIT** (always wrong, but wears a
judge-favoured phrase), and **HEDGE** (a plain wrong control). Two reward channels model the two
regimes: **VerifiableReward** returns the oracle's correctness (RLVR — ungameable by
construction), and **GameableReward** additionally passes anything wearing a favoured phrase (the
Wind Tunnel's KeywordGaming failure mode, reused as an RL reward). A REINFORCE loop with a moving
baseline trains the policy against a channel; a Goodhart monitor records proxy reward vs. oracle
truth at every step. The favoured phrases, the exploit definition, the witnessed fork, and the
evidence-bundle schema are reused directly from the Wind Tunnel.

## 3. Results (offline, reproducible)

**3.1 The algorithm works (Fig. 1).** A from-scratch PPO — clipped surrogate, GAE(λ), value
baseline, entropy bonus, hand-derived gradients in NumPy — learns the toy MDP's optimal policy
(mean episodic return climbs to the optimal +0.84). This is the internals check: the algorithm is
implemented, not called.

![PPO learning curve](figures/fig1_ppo_learning.png)

**3.2 Emergence and the control (Fig. 2).** Against the gameable reward the proxy climbs to 0.98
while true quality falls to 0.00 (gap +0.98, correlation −0.84): the policy learns EXPLOIT. The
verifiable-reward control tracks truth exactly (proxy = true = 0.62, gap 0.00): it learns SOLVE.
The gap is caused by the reward's exploitability, not by RL.

![Goodhart divergence](figures/fig2_goodhart_divergence.png)

**3.3 Localization (Fig. 3).** Among the 64 oracle-witnessed exploits, forking the single
suspected variable — remove the phrase, change nothing else — flips the reward PASS→FAIL at rate
1.00, versus 0.00 for the same fork on genuinely-correct answers: lift +1.00, validated. The
exploited feature is identified causally, in-loop.

![Witnessed localization](figures/fig3_localization.png)

**3.4 Repair — whack-a-mole then cure (Fig. 4).** With a reward fooled by several cues (phrase,
authority, verbosity), patching the localized cue and retraining does not cure the gap — the
policy relocates to the next cue. Only after every cue is patched does the gap close. Relocation
share γ_local = 0.67: mostly whack-a-mole, then a cure — the Wind Tunnel's repair dichotomy,
observed in the training loop.

![Repair whack-a-mole](figures/fig4_repair_whackamole.png)

**3.5 The phenomenon is objective-agnostic (Fig. 5).** The RL policy-gradient loop and DPO's
*implicit* reward both learn to exploit a gameable reward (P(exploit) 0.98 and 0.66 respectively)
and both stay near zero under a verifiable reward. DPO never trains an explicit reward model, yet
gameable *preferences* teach its implicit reward the same exploit — so this is a property of
reward-driven optimization, not of one algorithm.

![Both objectives hack](figures/fig5_objectives.png)

## 4. Method and mathematics

The repository implements and/or derives: the policy-gradient theorem and the closed-form softmax
gradient `∇_z log π = e_a − π`; the PPO-clip objective with GAE(λ); GRPO's group-relative
advantage `Â_i = (r_i − mean_j r_j)/(std_j r_j + ε)` (unit-tested); and DPO's implicit reward
`r(y) = β log π(y)/π_ref(y)` from the Bradley–Terry model, with the standard DPO loss. The
Goodhart curve is the empirical face of reward over-optimization: under a KL budget the gold
reward rises then falls as the policy drifts from the reference. Full derivations are in
`PROGRAM.md`.

## 5. Real-scale plan (M2–M5)

`scripts/train_grpo.py` runs GRPO on a small instruct model (e.g. Qwen2.5-0.5B + LoRA) over
GSM8K. The verifiable channel is exact-match on the final answer (RLVR); the gameable channel
also passes phrase-wearing completions. Expected: the verifiable run raises true accuracy with no
gap (control at scale, M2); the gameable run reproduces the Goodhart curve on a real policy (M3);
witnessed forks on the reward localize the exploited feature (M4); patch-and-continue gives the
repair curve (M5). One GPU and a few hours suffice; every run writes a verifiable evidence bundle.

## 6. Related work

Reward over-optimization and its scaling (Gao, Schulman & Hilton, 2022); defining and
characterizing reward hacking (Skalse et al., 2022) and the effects of reward misspecification
(Pan, Bhatia & Steinhardt, 2022); specification gaming and concrete safety problems (Amodei et
al., 2016; Krakovna et al., 2020); RLHF (Christiano et al., 2017; Ouyang et al., 2022); DPO
(Rafailov et al., 2023); GRPO (Shao et al., 2024); PPO and GAE (Schulman et al., 2017; 2016);
Goodhart's law. *(To be expanded with full citations and positioning in the camera-ready.)*

## 7. Limitations

The offline task is deliberately minimal — a single-step action model with a synthetic oracle —
so it isolates the mechanism rather than estimating real-world magnitudes; the localizer assumes
a candidate variable to fork (as does the Wind Tunnel); and the LLM-scale hypotheses (M2–M5) are
specified but not yet run. These are the point of the milestone plan, not hidden gaps.

## 8. Reproducibility

`make test` runs 27 property/control checks that gate the science (the exploit definition, the
control, the localizer, the repair convergence, the evidence chain). `make demo` runs the full
attack→localize→repair→breadth story and writes a hash-chained, tamper-evident evidence bundle to
`runs/committed/`; `gradia-reward-loop verify runs/committed` recomputes it. `make figures`
regenerates every figure from live runs. The evidence schema is compatible with the Wind Tunnel
manifest, so one verifier discipline spans Pillars 1–4.
