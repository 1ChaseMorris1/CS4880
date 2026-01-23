.PHONY: help install wandb-login train train-dense train-sparse clean

PY ?= python3
PIP ?= pip3

-include .env
export GRID_SIZE REWARD_TYPE TOTAL_TIMESTEPS ALGORITHMS RECORD_VIDEO VIDEO_DIR LOG_VIDEO_EVERY_STEPS VIDEO_SECONDS VIDEO_LENGTH

help:
	@echo "make install        Install dependencies"
	@echo "make wandb-login    Authenticate with WandB"
	@echo "make train          Train with current .env values"
	@echo "make train-dense    Train with dense reward"
	@echo "make train-sparse   Train with sparse reward"
	@echo "make clean          Remove saved model"

install:
	$(PIP) install -r requirements.txt

wandb-login:
	wandb login

train:
	$(PY) train.py

train-dense:
	REWARD_TYPE=dense $(PY) train.py

train-sparse:
	REWARD_TYPE=sparse $(PY) train.py

clean:
	rm -f snake_*.zip
