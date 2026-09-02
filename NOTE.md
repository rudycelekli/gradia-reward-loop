# Reward Hacking in the RL Loop
### Pillar 4 of the Gradia program — a training-time extension of the Reward-Hacking Wind Tunnel

**Rudy Celekli · Gradia Research.** Status: offline instrument and one-seed real-policy diagnostic
complete and reproducible. All reported numbers recompute from committed evidence and the final
adapters replay their held-out evaluations exactly.

## Abstract

Reward hacking is usually studied as a static property of a benchmark scorer. But the place it
does damage is the reinforcement-learning training loop, where an optimizer queries a reward
model millions of times *specifically to find its highest-scoring behaviours*. We carry the
Reward-Hacking Wind Tunnel's instrument — the oracle-witnessed exploit definition
(`reward-PASS ∧ oracle-WRONG`), witnessed single-variable localization, and hash-chained
evidence bundles — into that loop. On a minimal, fully reproducible task we show the following.
**(H1, emergence)** Optimizing a policy against a *gameable* reward opens a Goodhart gap — proxy
reward 0.98 while true (oracle) quality collapses to 0.00 — with the proxy–truth correlation
going negative (−0.84). **(Control C)** The same optimizer against an oracle-*verifiable* reward
(RLVR) opens no gap (0.00): the exploitability, not the RL, is the cause. **(H2, localization)** A
witnessed single-variable fork on the reward channel recovers the exploited feature with lift
+1.00 over baseline (n = 64 witnessed exploits). **(H3, repair)** Patching the localized cue and
continuing *relocates* the exploit across cues (whack-a-mole) before a comprehensive patch cures
it (γ_local = 0.67). Finally, the mechanism shows **objective breadth in this controlled setting**:
the policy-gradient loop and DPO's *implicit* reward both hack a gameable reward and remain
low-exploit under the verifiable control; a separate from-scratch PPO experiment validates the
optimization machinery. We then specify the real-scale program: GRPO on a small language model
over GSM8K, verifiable (RLVR) vs. gameable reward. That frozen pair has now completed: the
verifiable arm preserved a zero proxy–oracle gap at all 13 evaluations and finished at 13/64;
the gameable arm finished at 58/64 proxy versus 1/64 oracle (57/64 wrong-but-rewarded, gap
0.890625), satisfying the prospective final-gap rule. Because the gameable baseline already
contained six exploits, the result is amplification of an existing reward seam, not emergence
from zero; it is one fixed-seed diagnostic, not a population or capability-improvement claim.

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

**3.5 The mechanism spans the implemented objectives (Fig. 5).** The RL policy-gradient loop and DPO's
*implicit* reward both learn to exploit a gameable reward (P(exploit) 0.98 and 0.66 respectively)
and both stay near zero under a verifiable reward. DPO never trains an explicit reward model, yet
gameable *preferences* teach its implicit reward the same exploit — so this is a property of
reward-driven optimization, not of one algorithm.

![Both objectives hack](figures/fig5_objectives.png)

**3.6 Optimization pressure drives reward hacking (Fig. 6).** The KL-regularized optimal policy is
Boltzmann in the proxy reward; sweeping the optimization pressure traces true (oracle) reward
against KL from the initial policy. As pressure rises the policy concentrates on the reward's
exploitable seam -- P(exploit) climbs from 0.13 to 1.00 -- while true reward falls from 0.61 to
0.26 (a 0.35 loss), averaged over reward-model draws with bootstrap CIs. This is the
reward-hacking face of over-optimization (Gao, Schulman & Hilton, 2022).

![Optimization pressure](figures/fig6_overoptimization.png)

**3.7 A *learned* reward model is hacked through a spurious feature (Fig. 7).** Replacing the
rule-based reward with a logistic reward model trained on data in which the favoured phrase
spuriously correlates with correctness, the model *learns* to weight the phrase (weight rising with
the training correlation, 0.0 -> 5.1), the RL policy exploits it (P(exploit) -> 0.98), and the
witnessed fork recovers the learned feature every time. Severity is a dose-response in the spurious
correlation -- closing the "it is only a hardcoded rule" objection.

![Learned reward model](figures/fig7_learned_rm.png)

**3.8 An online detector flags hacking early (Fig. 8).** The Wind Tunnel witnesses exploits by
spot-auditing with an oracle; run that audit each training window and the witnessed-exploit rate
becomes an early-warning signal. On the gameable reward the detector fires at iteration 12
(gap 0.60), before the gap saturates at 0.98; the matched verifiable-control run produces zero
alarms. This is a negative control for the committed seed, not an estimated false-positive rate;
it is the detection half of a training-time immune system whose repair half is Section 3.4.

![Online detector](figures/fig8_detector.png)

## 4. Method and mathematics

The repository implements and/or derives: the policy-gradient theorem and the closed-form softmax
gradient `∇_z log π = e_a − π`; the PPO-clip objective with GAE(λ); GRPO's group-relative
advantage `Â_i = (r_i − mean_j r_j)/(std_j r_j + ε)` (unit-tested); and DPO's implicit reward
`r(y) = β log π(y)/π_ref(y)` from the Bradley–Terry model, with the standard DPO loss. The
Goodhart curve is the empirical face of reward over-optimization: under a KL budget the gold
reward rises then falls as the policy drifts from the reference. Full derivations are in
`PROGRAM.md`.

## 5. Real-policy diagnostic (M2–M3) and remaining plan

`scripts/train_grpo.py` ran a frozen pair on Qwen2.5-0.5B-Instruct with LoRA over GSM8K. Both arms
used the same pinned policy and dataset revisions, 128 training prompts, 64 held-out prompts, seed,
optimizer, 300 steps, and group-of-eight sampling; only the reward channel differed. The exact
final-answer arm remained aligned by construction. The gameable arm also passed completions
containing the calibrated discourse cue `therefore`.

| arm | baseline proxy / oracle | final proxy / oracle | final gap |
|---|---:|---:|---:|
| exact-match control | 18 / 18 | 13 / 13 | 0 / 64 |
| gameable reward | 24 / 18 | 58 / 1 | 57 / 64 |

The frozen H1 rule is supported. Post-observation exploratory analysis found first threshold
crossing at step 25, persistence at all 12 post-baseline evaluations, and gap widening from 6/64
to 57/64. Across 2,400 sampled training completions, the gameable arm recorded 994
wrong-but-rewarded cases versus zero in the control. Accuracy fell from 18/64 to 13/64 in the
control and to 1/64 in the gameable arm; this is not evidence that GRPO improved capability.
Real-policy witnessed localization and patch-and-continue remain M4–M5.

![Frozen paired GRPO diagnostic](figures/fig9_real_grpo_pair.png)

## 6. Related work

The paper positions this instrument against reward over-optimization, formal reward-hacking and
reward-corruption work, specification gaming, causal reward modelling (Causal Rewards and RATE),
latent online detection (InfoRM), active auditing, ensembles and weight averaging, and reported
generalization from reward hacking to broader misalignment. Its narrow difference is operational:
oracle-confirmed exploits emitted by a running optimizer are subjected to an exact candidate-
variable fork, the same transform is applied to negative controls, and the result is sealed before
repair is re-attacked. It does not claim general detection without an oracle or automatic causal
discovery.

## 7. Limitations

The offline task is deliberately minimal — a single-step action model with a synthetic oracle —
so it isolates the mechanism rather than estimating real-world magnitudes. The real-policy slice
adds a neural policy but only one 0.5B model, one seed, 128 training prompts, and 64 fixed held-out
items. Its reward seam is intentionally simple and present before optimization. The localizer
assumes a candidate variable and a valid single-variable transform. M4–M5 have not yet tested
localization or repair on the real-policy run.

## 8. Reproducibility

`make test` runs 55 property/control checks that gate the science (the exploit definition, the
control, the localizer, repair convergence, semantic pair admission, analysis, and replay
receipts). `make demo` runs the full
attack→localize→repair→breadth story and writes a hash-chained, tamper-evident evidence bundle to
`runs/committed/`; `gradia-reward-loop verify runs/committed` recomputes it. `make figures`
regenerates the offline figures; `make analyze-real` reconstructs Figure 9 and a self-digesting
summary from the frozen frames. `make verify-real` verifies both 313-frame chains, the exact pair
contract and adapters, the primary decision, and model-backed replay receipts that reproduce each
final 64-row evaluation digest. The evidence schema is compatible with the Wind Tunnel manifest,
so one verifier discipline spans Pillars 1–4.
