# gradia-reward-loop

**Pillar 4 of the Gradia program — reward hacking in the RL loop.** A training-time extension
of the [Reward-Hacking Wind Tunnel](../gradia-wind-tunnel): the same oracle-witnessed exploit
definition (`reward-PASS ∧ oracle-WRONG`), the same witnessed single-variable localization, and
the same hash-chained evidence bundles — now on the reward signal of a live RL loop.

A benchmark scorer is graded once; an RL reward model is queried millions of times by an
optimizer that is *trying* to find its seams. This repo shows, reproducibly and with no GPU,
that a gameable reward gets hacked in the loop while a verifiable-reward control does not — and
that the Wind Tunnel's causal instrument localizes the exploited feature.

## Quickstart (no GPU, no network)

```bash
make demo      # end-to-end: PPO learns a toy MDP; a gameable reward gets hacked while a
               # verifiable-reward control does not; the exploit is localized; evidence verified
make test      # 20 property/control checks (gates the science, not just the plumbing)
make ppo-toy   # the from-scratch PPO learning curve
```

The demo prints, among other things:

```
    channel      proxy   true    gap   corr   hacked
    verifiable    0.62   0.62  +0.00  +1.00   False     <- RLVR control: tracks truth
    gameable      0.98   0.00  +0.98  -0.84   True      <- Goodhart: proxy up, truth gone
  localization (gameable): {flip_rate: 1.0, baseline: 0.0, lift: 1.0, validated: True}
```

## Real LLM training (Milestone M2+)

```bash
pip install -e '.[real,gradia]'   # torch + transformers + trl, and the Wind Tunnel primitives
make train                        # GRPO on a small model (needs a GPU)
```

## Layout

| file | what |
|---|---|
| `ppo.py` | from-scratch PPO (clipped surrogate + GAE + entropy), numpy, hand-derived gradients; learns the toy MDP |
| `rewards.py` | `VerifiableReward` (RLVR / the oracle) and `GameableReward` (the phrase exploit); the `ProxyTask` action model |
| `loop.py` | REINFORCE loop with a moving baseline + the Goodhart monitor |
| `monitor.py` | the proxy-vs-truth gap, peak gap, and Goodhart correlation |
| `localize.py` | witnessed single-variable localization of the reward exploit |
| `grpo.py` | GRPO group-advantage core (unit-tested) + real-LLM trainer skeleton |
| `evidence.py` | hash-chained, tamper-evident run bundles (Wind-Tunnel-compatible schema) |
| `demo.py` · `cli.py` · `tests.py` | orchestration, CLI, and the property/control suite |

See **[PROGRAM.md](PROGRAM.md)** for the thesis, the four hypotheses, the mathematics this
program demonstrates (PPO/GAE, GRPO, DPO, reward over-optimization), and the M0–M5 milestone plan.

*Part of the Gradia program by Rudy Celekli. Apache-2.0.*
