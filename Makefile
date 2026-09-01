PY ?= python3
# make the sibling Wind Tunnel importable without installing (Pillar 1-3 primitives)
export PYTHONPATH := src:../gradia-wind-tunnel/src

.PHONY: help test demo ppo-toy train lint typecheck
help:
	@echo "make test      - property/control suite (no GPU, no network)"
	@echo "make demo      - offline reward-hacking demonstration (no GPU):"
	@echo "                 Goodhart gap under a gameable reward + in-loop witnessed localization"
	@echo "make ppo-toy   - train the from-scratch PPO on the toy MDP; prints the learning curve"
	@echo "make train     - real GRPO on a small LLM  (pip install -e '.[real,gradia]' + a GPU)"
	@echo "make lint      - ruff check src"

test:
	$(PY) -m gradia_reward_loop.tests
demo:
	$(PY) -m gradia_reward_loop.cli demo
ppo-toy:
	$(PY) -m gradia_reward_loop.cli ppo-toy
train:
	$(PY) -m gradia_reward_loop.cli train $(ARGS)
lint:
	ruff check src
typecheck:
	mypy src/gradia_reward_loop
