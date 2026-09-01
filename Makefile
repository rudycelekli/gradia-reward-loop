PY ?= python3
# make the sibling Wind Tunnel importable without installing (Pillar 1-3 primitives)
export PYTHONPATH := src:../gradia-wind-tunnel/src

.PHONY: help test demo ppo-toy train figures lint typecheck verify
help:
	@echo "make test      - property/control suite (no GPU, no network)"
	@echo "make demo      - offline reward-hacking demonstration (no GPU):"
	@echo "                 Goodhart gap under a gameable reward + in-loop witnessed localization"
	@echo "make ppo-toy   - train the from-scratch PPO on the toy MDP; prints the learning curve"
	@echo "make train     - real GRPO on a small LLM  (pip install -e '.[real,gradia]' + a GPU)"
	@echo "make lint      - ruff check src"
	@echo "make typecheck - strict package type gate"
	@echo "make verify    - verify the committed evidence manifest and frame chain"

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
