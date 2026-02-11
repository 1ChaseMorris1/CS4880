PYTHON ?= venv/bin/python
PIP ?= venv/bin/pip
SCRIPT := bandit_experiment.py

.PHONY: requirements run run-online

requirements:
	$(PIP) install -r requirements.txt

run-online: requirements
	$(PYTHON) $(SCRIPT) --steps 2000 --runs 200 --wandb-mode online
