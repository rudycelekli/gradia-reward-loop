# Real GRPO (Milestone M2) — how to run it on a GPU

Everything under `make demo` / `make test` runs on a laptop. The LLM milestone uses one CUDA or
Apple MPS accelerator. The admitted M2 contract is frozen in
`preregistrations/M2-PAIRED-GRPO.md`.

## Local

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[real,dev,gradia]'
.venv/bin/python scripts/train_grpo.py --channel verifiable --steps 300 --seed 20260901 \
  --max-new-tokens 128 --eval-batch-size 2 --generation-batch-size 2 --train-batch-size 1
.venv/bin/python scripts/train_grpo.py --channel gameable --steps 300 --seed 20260901 \
  --max-new-tokens 128 --eval-batch-size 2 --generation-batch-size 2 --train-batch-size 1
```

Each run writes a hash-chained bundle under one pair directory. Verify both arms and their shared
contract with `gradia-reward-loop verify-pair runs/m2/<pair-id>`.

## Rented GPU

- **RunPod / Lambda / vast.ai:** launch a PyTorch container, `git clone`, then the two commands
  above. ~$0.5–2/hr for an A100/H100 slice; the 0.5B control+hack sweep is well under an hour.
- **Modal:** see `scripts/modal_train.py` for an app stub (`modal run scripts/modal_train.py`).

## What to expect (the hypotheses at scale)

- `verifiable`: proxy equals oracle at every evaluation (no gap) — the exact RLVR control. An
  accuracy increase is neither required nor implied by the control invariant.
- `gameable`: proxy climbs toward 1.0 while true accuracy stalls/falls — H1 (Goodhart) on a real
  policy. Then point `localize` at the reward model to recover the exploited feature (H2), patch
  and continue for the repair curve (H3).
