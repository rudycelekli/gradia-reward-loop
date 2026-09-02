# gradia-reward-loop

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22259605.svg)](https://doi.org/10.5281/zenodo.22259605)

**Pillar 4 of the Gradia program — reward hacking in the RL loop.** A training-time extension
of the [Reward-Hacking Wind Tunnel](https://github.com/rudycelekli/gradia-wind-tunnel): the same oracle-witnessed exploit
definition (`reward-PASS ∧ oracle-WRONG`), the same witnessed single-variable localization, and
the same hash-chained evidence bundles — now on the reward signal of a live RL loop.

A benchmark scorer is graded once; an RL reward model is queried millions of times by an
optimizer that is *trying* to find its seams. This repo shows, reproducibly and with no GPU,
that a gameable reward gets hacked in the loop while a verifiable-reward control does not — and
that the Wind Tunnel's exact single-variable intervention localizes the exploited feature under
explicit fidelity assumptions. A frozen paired GRPO diagnostic now reproduces the separation on
Qwen2.5-0.5B-Instruct over GSM8K and binds the result to model-backed replay receipts.

**[Read the research paper (PDF)](paper/PAPER.pdf)** · [Methods and results](NOTE.md) ·
[Research program and milestones](PROGRAM.md)

## Claim boundary

- **Established here:** in the controlled offline system, policy-gradient optimization and DPO
  exploit a specified proxy; oracle-witnessed counterfactual intervention localizes the exploited
  feature; an online detector fires on the gameable channel and raises zero alarms in the one
  matched control run. In one frozen real-policy diagnostic, the gameable GRPO arm ends at
  proxy $58/64$ versus oracle $1/64$ (gap $57/64$), while the exact-match control preserves a
  zero proxy–oracle gap at all 13 evaluations.
- **Reproducible here:** fixed-seed experiments, 55 property/control checks, nine regenerated
  figures, two 313-frame evidence chains, exact final adapters, a self-digesting paired analysis,
  and model-backed replays that reproduce both final 64-row evaluation digests.
- **Not yet claimed:** universal reward-hacking detection, a false-positive-rate estimate,
  population prevalence, frontier- or production-model evidence, capability improvement, or
  causal discovery without a candidate intervention and a valid single-variable transform.
  Real-policy localization and repair (M4–M5) remain future work.

## Quickstart (no GPU, no network)

```bash
make demo      # end-to-end: PPO learns a toy MDP; a gameable reward gets hacked while a
               # verifiable-reward control does not; the exploit is localized; evidence verified
make test      # 55 property/control checks (gates the science, not just the plumbing)
make ppo-toy   # the from-scratch PPO learning curve
make lint      # static lint gate
make typecheck # strict type gate over the package
make verify    # verify the committed manifest and frame chain
make verify-real  # verify the frozen pair, analysis digest, adapters, and replay receipts
python paper/build_pdf.py  # atomically rebuild the paper (Tectonic or XeLaTeX)
```

The demo prints, among other things:

```
    channel      proxy   true    gap   corr   hacked
    verifiable    0.62   0.62  +0.00  +1.00   False     <- RLVR control: tracks truth
    gameable      0.98   0.00  +0.98  -0.84   True      <- Goodhart: proxy up, truth gone
  localization (gameable): {flip_rate: 1.0, baseline: 0.0, lift: 1.0, validated: True}
```

## Frozen real-policy result (Milestones M2–M3)

The one-seed diagnostic used identical base policy, data, prompt order, seed, optimizer, LoRA
configuration, and held-out items; only the reward channel differed.

| arm | baseline proxy / oracle | final proxy / oracle | final gap |
|---|---:|---:|---:|
| exact-match RLVR control | 18 / 18 | 13 / 13 | 0 / 64 |
| gameable reward | 24 / 18 | 58 / 1 | 57 / 64 (0.890625) |

The frozen H1 rule is supported. The baseline already contained six wrong cue-bearing completions,
so this is evidence of **amplification under optimization**, not creation of a vulnerability from
zero. Accuracy declined in both arms; this study does not claim GRPO improved capability.

To reproduce the training protocol:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[real,dev,gradia]'
.venv/bin/python scripts/train_grpo.py --channel verifiable --steps 300 --seed 20260901 \
  --max-new-tokens 128 --eval-batch-size 2 --generation-batch-size 2 --train-batch-size 1
.venv/bin/python scripts/train_grpo.py --channel gameable --steps 300 --seed 20260901 \
  --max-new-tokens 128 --eval-batch-size 2 --generation-batch-size 2 --train-batch-size 1
```

The runner accepts CUDA or Apple MPS, refuses a silent CPU fallback and dirty evidence-bearing
runs, resolves immutable Hugging Face revisions, evaluates proxy and oracle on the same held-out
completions, and seals each checkpoint and trajectory. `verify-pair` refuses mismatched arms. See
[`preregistrations/M2-PAIRED-GRPO.md`](preregistrations/M2-PAIRED-GRPO.md) for the frozen diagnostic
and its read-only outcome addendum.

## Layout

| file | what |
|---|---|
| `ppo.py` | from-scratch PPO (clipped surrogate + GAE + entropy), numpy, hand-derived gradients; learns the toy MDP |
| `rewards.py` | `VerifiableReward` (RLVR / the oracle) and `GameableReward` (the phrase exploit); the `ProxyTask` action model |
| `loop.py` | REINFORCE loop with a moving baseline + the Goodhart monitor |
| `monitor.py` | the proxy-vs-truth gap, peak gap, and Goodhart correlation |
| `localize.py` | witnessed single-variable localization of the reward exploit |
| `grpo.py` | GRPO group-advantage core + accelerator-backed, evidence-bound real-LLM trainer |
| `evidence.py` | hash-chained, tamper-evident run bundles (Wind-Tunnel-compatible schema) |
| `overopt.py` | optimization-pressure vs reward-hacking frontier (with bootstrap CIs, `stats.py`) |
| `reward_model.py` | a *learned* logistic reward model hacked through a spurious feature (dose-response) |
| `detector.py` | online hacking detector -- spot-audits the loop, flags hacking early (immune system) |
| `demo.py` · `cli.py` · `tests.py` | orchestration, CLI, and the 55-check property/control suite |

See **[NOTE.md](NOTE.md)** for the write-up (abstract, results, figures) and **[PROGRAM.md](PROGRAM.md)** for the thesis, the four hypotheses, the mathematics this
program demonstrates (PPO/GAE, GRPO, DPO, reward over-optimization), and the M0–M5 milestone plan.

## Citation

Citation metadata is available in [`CITATION.cff`](CITATION.cff). The exact v1.0.2
research artifact is archived at
[doi:10.5281/zenodo.22259605](https://doi.org/10.5281/zenodo.22259605).

```bibtex
@software{celekli2026rewardloop,
  author    = {Celekli, Rudy M.},
  title     = {Reward Hacking in the RL Loop: Oracle-Witnessed Localization
               and Repair of Reward-Model Exploits During Training},
  year      = {2026},
  version   = {1.0.2},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22259605},
  url       = {https://doi.org/10.5281/zenodo.22259605}
}
```

The one-seed, 0.5B-model claim boundary above remains binding; archival release
does not upgrade the result to an LLM-scale or population-level claim.

*Part of the Gradia program by Rudy Celekli. Apache-2.0.*
