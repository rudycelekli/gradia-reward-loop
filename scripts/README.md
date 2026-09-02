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

## Admitted result and remaining hypotheses

- `verifiable`: proxy equalled oracle at every evaluation and finished at 13/64 (gap 0).
- `gameable`: proxy finished at 58/64 while oracle finished at 1/64 (gap 57/64); the frozen H1
  rule is supported. The baseline already had 6/64 exploits, so the observed change is
  amplification, not emergence from zero.
- Real-policy localization and patch-and-continue are not part of this pair; they remain M4–M5.

Run `make verify-real` to validate both 313-frame chains, the exact contract, stored analysis, and
the two final-model replay receipts.
