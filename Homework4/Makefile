MAKEFLAGS += --no-print-directory

CURDIR := $(shell pwd)
VENV_DIR ?= $(CURDIR)/venv
PYTHON ?= $(VENV_DIR)/bin/python3
PIP ?= $(VENV_DIR)/bin/pip
SETUP_STAMP := $(CURDIR)/.setup.stamp

ENV_FILE ?= $(CURDIR)/.env
OUTPUT_DIR ?= $(CURDIR)/solutions/summary

ITERATIONS ?= 10,50,100,500
GAMES_PER_SIDE ?= 40
VS_MINIMAX_GAMES_PER_SIDE ?= 15
EXPLORATION ?= 1.41
SEED ?= 7

LLM_PROVIDER ?= heuristic
LLM_MODEL ?= gpt-4.1-mini
LLM_CACHE ?= solutions/summary/llm_cache.json

.PHONY: help check-tools setup run quick test report show-results clean

help:
	@echo "Homework 4 workflow:"
	@echo "  make run                # full experiment + report/slides"
	@echo "  make quick              # smaller/faster experiment"
	@echo "  make test               # unit tests"
	@echo "  make report             # run + print output locations"
	@echo ""
	@echo "OpenAI mode (optional):"
	@echo "  Set OPENAI_API_KEY in .env, then:"
	@echo "  make run LLM_PROVIDER=openai OPEN_AI_MODEL=gpt-4.1-mini"

check-tools:
	@command -v python3 >/dev/null || (echo "Missing: python3" && exit 1)

setup: check-tools
	@mkdir -p "$(OUTPUT_DIR)"
	@if [ ! -x "$(PYTHON)" ]; then \
		echo "Creating virtualenv at $(VENV_DIR)"; \
		python3 -m venv "$(VENV_DIR)"; \
	fi
	@if [ ! -f "$(SETUP_STAMP)" ] || [ "$(CURDIR)/solutions/requirements.txt" -nt "$(SETUP_STAMP)" ]; then \
		"$(PIP)" install --disable-pip-version-check -q -r "$(CURDIR)/solutions/requirements.txt"; \
		touch "$(SETUP_STAMP)"; \
	fi

run: setup
	@set -a; [ -f "$(ENV_FILE)" ] && . "$(ENV_FILE)"; set +a; \
	LLM_PROVIDER="$${LLM_PROVIDER:-$(LLM_PROVIDER)}"; \
	LLM_MODEL="$${LLM_MODEL:-$(LLM_MODEL)}"; \
	OPEN_AI_MODEL="$${OPEN_AI_MODEL:-}"; \
	if [ -n "$$OPEN_AI_MODEL" ]; then LLM_MODEL="$$OPEN_AI_MODEL"; fi; \
	"$(PYTHON)" -m solutions.experiment \
		--iterations "$(ITERATIONS)" \
		--games-per-side "$(GAMES_PER_SIDE)" \
		--vs-minimax-games-per-side "$(VS_MINIMAX_GAMES_PER_SIDE)" \
		--exploration "$(EXPLORATION)" \
		--seed "$(SEED)" \
		--output-dir "$(OUTPUT_DIR)" \
		--llm-provider "$$LLM_PROVIDER" \
		--llm-model "$$LLM_MODEL" \
		--llm-cache "$(LLM_CACHE)"

quick: setup
	@$(MAKE) run ITERATIONS=10,50 GAMES_PER_SIDE=12 VS_MINIMAX_GAMES_PER_SIDE=4

test: setup
	@PYTHONPATH="$(CURDIR)" "$(PYTHON)" -m unittest discover -s solutions/tests -p "test_*.py" -v

report: run
	@echo ""
	@echo "Artifacts:"
	@echo "  - $(OUTPUT_DIR)/summary.csv"
	@echo "  - $(OUTPUT_DIR)/move_metrics.csv"
	@echo "  - $(OUTPUT_DIR)/summary.json"
	@echo "  - $(OUTPUT_DIR)/run_config.json"
	@echo "  - $(OUTPUT_DIR)/REPORT.md"
	@echo "  - $(OUTPUT_DIR)/SLIDES.md"

show-results:
	@if [ -f "$(OUTPUT_DIR)/summary.csv" ]; then \
		cat "$(OUTPUT_DIR)/summary.csv"; \
	else \
		echo "No summary yet. Run: make run"; \
	fi

clean:
	@rm -f "$(OUTPUT_DIR)/summary.csv" "$(OUTPUT_DIR)/move_metrics.csv" "$(OUTPUT_DIR)/summary.json" "$(OUTPUT_DIR)/run_config.json" "$(OUTPUT_DIR)/REPORT.md" "$(OUTPUT_DIR)/SLIDES.md" "$(OUTPUT_DIR)/llm_cache.json"
