# Real GRPO (Milestone M2) — how to run it on a GPU

Everything under `make demo` / `make test` runs on a laptop. The LLM milestones need one GPU
(a 0.5–3B model with LoRA fits a single 24–80 GB card; a few H100/A100 hours is plenty).

## Local (a machine with a CUDA GPU)

```bash
pip install -e '.[real,gradia]'
python scripts/train_grpo.py --channel verifiable --steps 200   # RLVR control: no Goodhart gap
python scripts/train_grpo.py --channel gameable   --steps 200   # reproduce reward hacking
```

Each run prints the final proxy/true/gap and writes a hash-chained evidence bundle under
`runs/`; verify it with `gradia-reward-loop verify runs/<bundle>`.

## Rented GPU

- **RunPod / Lambda / vast.ai:** launch a PyTorch container, `git clone`, then the two commands
  above. ~$0.5–2/hr for an A100/H100 slice; the 0.5B control+hack sweep is well under an hour.
- **Modal:** see `scripts/modal_train.py` for an app stub (`modal run scripts/modal_train.py`).

## What to expect (the hypotheses at scale)

- `verifiable`: true accuracy rises, proxy≈true (no gap) — H-control.
- `gameable`: proxy climbs toward 1.0 while true accuracy stalls/falls — H1 (Goodhart) on a real
  policy. Then point `localize` at the reward model to recover the exploited feature (H2), patch
  and continue for the repair curve (H3).
