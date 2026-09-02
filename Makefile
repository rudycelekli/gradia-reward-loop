PY ?= python3
# make the sibling Wind Tunnel importable without installing (Pillar 1-3 primitives)
export PYTHONPATH := src:../gradia-wind-tunnel/src

.PHONY: help test demo ppo-toy train figures lint typecheck verify verify-real analyze-real
help:
	@echo "make test      - property/control suite (no GPU, no network)"
	@echo "make demo      - offline reward-hacking demonstration (no GPU):"
	@echo "                 Goodhart gap under a gameable reward + in-loop witnessed localization"
	@echo "make ppo-toy   - train the from-scratch PPO on the toy MDP; prints the learning curve"
	@echo "make train     - real GRPO on a small LLM  (pip install -e '.[real,gradia]' + a GPU)"
	@echo "make lint      - ruff check src"
	@echo "make typecheck - strict package type gate"
	@echo "make verify    - verify the committed evidence manifest and frame chain"
	@echo "make verify-real - verify the frozen paired-GRPO evidence and decision rule"
	@echo "make analyze-real - rebuild the paired-GRPO JSON summary and Figure 9"

test:
	$(PY) -m gradia_reward_loop.tests
demo:
	$(PY) -m gradia_reward_loop.cli demo
ppo-toy:
	$(PY) -m gradia_reward_loop.cli ppo-toy
train:
	$(PY) -m gradia_reward_loop.cli train $(ARGS)
figures:
	$(PY) -m gradia_reward_loop.figures

lint:
	ruff check src
typecheck:
	MYPYPATH=src:../gradia-wind-tunnel/src mypy --explicit-package-bases src/gradia_reward_loop
verify:
	$(PY) -m gradia_reward_loop.cli verify runs/committed

PAIR_DIR ?= runs/m2/Qwen2.5-0.5B-Instruct-s20260901-99d688fed184
verify-real:
	$(PY) -m gradia_reward_loop.cli verify-pair $(PAIR_DIR)
	$(PY) -m gradia_reward_loop.cli verify-analysis $(PAIR_DIR) \
		results/M2-PAIRED-GRPO-SUMMARY.json
	$(PY) -m gradia_reward_loop.cli verify-final-replay $(PAIR_DIR) verifiable \
		results/M2-FINAL-REPLAY-verifiable.json
	$(PY) -m gradia_reward_loop.cli verify-final-replay $(PAIR_DIR) gameable \
		results/M2-FINAL-REPLAY-gameable.json
analyze-real:
	$(PY) scripts/analyze_pair.py $(PAIR_DIR)
