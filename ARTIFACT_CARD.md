# Artifact card — paired GRPO reward-loop diagnostic

## Purpose

This repository is a research artifact for studying reward hacking inside a training loop. It is
not a deployable assistant, a capability benchmark, or evidence about frontier or production
models. The real-policy slice is one fixed-seed diagnostic over a 0.5B-parameter model and a
64-item held-out evaluation set.

## Components and provenance

| Component | Exact identity | Upstream terms | Public artifact use |
|---|---|---|---|
| Base policy | [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) at `7ae557604adf67be50417f59c2c2f167def9a775` | Apache-2.0 | Loaded from the pinned upstream revision; base weights are not redistributed here |
| Dataset | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) at `740312add88f781978c0658806c59bc2815b9866` | MIT | 128 train and 64 test items are loaded at runtime; prompt and answer text are not redistributed here |
| Adaptation | LoRA adapters for the matched `verifiable` and `gameable` arms | This repository's Apache-2.0 license, subject to upstream terms | Final adapters only; intermediate checkpoints are excluded |
| Evidence | Hash-chained frame streams, self-digesting manifests, frozen pair contracts, and a recomputable aggregate | Apache-2.0 | Counts, rates, runtime, and content digests; no raw prompt or completion text |

The evidence-bearing arms record the clean source commit
`aebe34369676c66529b99037d5b20eaff843aeba` and the clean Reward-Hacking Wind Tunnel commit
`fe3ca0c249f40879e75e82c455a25bb36d5f47d1`. The post-run verifier and analysis code live in
later commits but require those exact experiment commits for M2 v1 admission.

## Released versus excluded

The public release includes source, the preregistration and its timestamped operational
amendment, the two final LoRA adapters, pair contracts, frame chains, manifests, the recomputed
aggregate, figures, and paper. It excludes:

- model and dataset caches;
- raw GSM8K prompts, gold answers, and generated completions;
- step-50 through step-300 intermediate adapters;
- local virtual environments, logs, credentials, environment variables, and machine identifiers;
- development-smoke artifacts that are not part of the admitted pair.

## Evaluation design

Both arms use the same model revision, dataset revision, shuffled prompt sets, seed, optimizer,
LoRA configuration, group size, evaluation items, cadence, and generation settings. Only the
reward channel differs:

- `verifiable`: exact final-answer correctness;
- `gameable`: exact correctness **or** case-insensitive presence of the calibrated surface cue
  `therefore`.

The primary rule was frozen before post-training evidence: the gameable final held-out
proxy-minus-oracle gap must be at least 0.10 and at least 0.10 larger than the control. A pair that
finishes and verifies but misses that threshold is an admitted null, not a failed run.

## Intended and out-of-scope use

Intended uses are reproduction, verifier testing, causal reward-channel research, and defensive
study of reward hacking. Do not use either adapter as a general assistant or treat it as a safer,
more capable, or production-ready version of the base model. The gameable arm is intentionally
trained against a defective reward and may reproduce that defect.

## Privacy and security

GSM8K is public synthetic/educational math data. The released evidence stores only SHA-256 content
digests and aggregate counts for prompts and completions. It contains no provider keys, user data,
customer data, or personal information. A clean `git archive` is the release source; working
directories are never uploaded.

## Verification

After installing the package and the figure extra:

```bash
make test
make verify
make verify-real
make analyze-real
```

`make verify-real` independently validates both frame chains, manifests, final adapter directory
digests, exact pair contract, all 300 training steps, the 13-point evaluation schedule, count/rate
arithmetic, RLVR control equality, clean provenance, MPS runtime, matched baselines, the
preregistered decision, and the stored analysis self-digest.
