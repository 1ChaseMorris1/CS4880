MICRORTS_DIR ?= ../MicroRTS
CHASEBOT_SRC := $(CURDIR)/chasebot/chasebot
MICRORTS_BOT_DIR := $(MICRORTS_DIR)/src/ai/chasebot
MAKEFLAGS += --no-print-directory

BENCHMARK_RESULTS ?= $(CURDIR)/chasebot/report/benchmark-results.tsv
WANDB_META ?= $(CURDIR)/chasebot/report/wandb-meta.txt
# 5x denser than 550. With MAX_CYCLES=5000, 110 gives about 46 frames/game.
FRAME_EVERY ?= 110
FRAME_OUT ?= $(CURDIR)/chasebot/report/images/match1
VIDEO_OUT ?= $(CURDIR)/chasebot/report/videos/match1

CHASE_CLASS ?= ai.chasebot.ChaseBot
# Default is 6 games per opponent (up to 36 total matches)
GAMES_PER_OPP ?= 6
# Total game runs to execute across all available opponents:
TOTAL_GAME_RUNS ?= 6
MAX_CYCLES ?= 5000
MAP ?=

VENV_DIR ?= $(CURDIR)/venv
VENV_PYTHON ?= $(VENV_DIR)/bin/python3
VENV_PIP ?= $(VENV_DIR)/bin/pip
PYTHON ?= $(VENV_PYTHON)
SETUP_STAMP := $(CURDIR)/.setup.stamp

WANDB_UPLOAD ?= 1
WANDB_PROJECT ?= cs4880-pa3
WANDB_ENTITY ?=
WANDB_RUN_NAME ?= chasebot-pa3-report
WANDB_DIR ?= $(CURDIR)/.wandb

.PHONY: help setup check-tools check-java-tools check-microrts sync-chasebot build-microrts gui microrts-help \
	run-benchmarks wandb-report run

help:
	@echo "PA3 workflow:"
	@echo "  make run"
	@echo ""
	@echo "Required:"
	@echo "  - MicroRTS repo exists at $(MICRORTS_DIR)"
	@echo "Optional:"
	@echo "  make run MICRORTS_DIR=/path/to/MicroRTS WANDB_UPLOAD=0"

check-tools:
	@command -v python3 >/dev/null || (echo "Missing: python3" && exit 1)
	@command -v ffmpeg >/dev/null || (echo "Missing: ffmpeg" && exit 1)

check-java-tools:
	@command -v java >/dev/null || (echo "Missing: java" && exit 1)
	@command -v javac >/dev/null || (echo "Missing: javac" && exit 1)
	@command -v rsync >/dev/null || (echo "Missing: rsync" && exit 1)

setup: check-tools check-java-tools
	@echo "[1/3] Setup"
	@mkdir -p "$(CURDIR)/chasebot/report/videos" "$(CURDIR)/chasebot/report/images" "$(FRAME_OUT)" "$(VIDEO_OUT)" "$(WANDB_DIR)"
	@if [ ! -x "$(VENV_PYTHON)" ]; then \
		echo "Creating virtualenv at $(VENV_DIR)"; \
		python3 -m venv "$(VENV_DIR)"; \
	fi
	@if [ ! -f "$(SETUP_STAMP)" ] || [ "$(CURDIR)/requirements.txt" -nt "$(SETUP_STAMP)" ]; then \
		"$(VENV_PIP)" install --disable-pip-version-check -q -r "$(CURDIR)/requirements.txt"; \
		touch "$(SETUP_STAMP)"; \
	fi
	@test -d "$(MICRORTS_DIR)/src" || (echo "ERROR: MicroRTS missing at $(MICRORTS_DIR). Set MICRORTS_DIR=/path/to/MicroRTS" && exit 1)
	@$(MAKE) sync-chasebot MICRORTS_DIR="$(MICRORTS_DIR)"
	@$(MAKE) build-microrts MICRORTS_DIR="$(MICRORTS_DIR)"
	@echo "[1/3] Done"

check-microrts:
	@test -d "$(MICRORTS_DIR)/src" || (echo "MICRORTS_DIR is invalid: $(MICRORTS_DIR)" && exit 1)

sync-chasebot: check-microrts
	@mkdir -p "$(MICRORTS_BOT_DIR)"
	@rsync -a --delete "$(CHASEBOT_SRC)/" "$(MICRORTS_BOT_DIR)/"
	@echo "Synced ChaseBot sources to $(MICRORTS_BOT_DIR)"

build-microrts: check-microrts
	@cd "$(MICRORTS_DIR)" && javac -cp "lib/*:src" -d bin src/rts/MicroRTS.java
	@cd "$(MICRORTS_DIR)" && if [ -d src/ai/chasebot ]; then javac -cp "lib/*:src:bin" -d bin $$(find src/ai/chasebot -name '*.java' -print); fi
	@echo "Built MicroRTS with ChaseBot sources"

gui: check-microrts
	@cd "$(MICRORTS_DIR)" && java -cp "lib/*:bin" gui.frontend.FrontEnd

microrts-help: check-microrts
	@cd "$(MICRORTS_DIR)" && java -cp "lib/*:bin" rts.MicroRTS -h

run-benchmarks: check-microrts
	@echo "[2/3] Run benchmarks"
	@mkdir -p "$(FRAME_OUT)"
	@"$(PYTHON)" "$(CURDIR)/scripts/run_pa3_benchmarks.py" \
		--microrts-dir "$(MICRORTS_DIR)" \
		--results "$(BENCHMARK_RESULTS)" \
		--frames-dir "$(FRAME_OUT)" \
		--frame-every "$(FRAME_EVERY)" \
		--games "$(GAMES_PER_OPP)" \
		--max-total-games "$(TOTAL_GAME_RUNS)" \
		--max-cycles "$(MAX_CYCLES)" \
		--chase-class "$(CHASE_CLASS)" \
		$(if $(MAP),--map "$(MAP)")
	@echo "[2/3] Done: $(BENCHMARK_RESULTS)"

wandb-report:
	@mkdir -p "$(WANDB_DIR)" "$(WANDB_DIR)/cache" "$(WANDB_DIR)/config"
	@WANDB_DIR="$(WANDB_DIR)" WANDB_CACHE_DIR="$(WANDB_DIR)/cache" WANDB_CONFIG_DIR="$(WANDB_DIR)/config" \
	"$(PYTHON)" "$(CURDIR)/scripts/upload_games_to_wandb.py" --results "$(BENCHMARK_RESULTS)" --frames-dir "$(FRAME_OUT)" --videos-dir "$(VIDEO_OUT)" --project "$(WANDB_PROJECT)" --entity "$(WANDB_ENTITY)" --run-name-base "$(WANDB_RUN_NAME)" --meta-out "$(WANDB_META)"

run: setup run-benchmarks
	@echo "[3/3] Publish"
	@if [ "$(WANDB_UPLOAD)" = "1" ]; then \
		$(MAKE) wandb-report FRAME_OUT="$(FRAME_OUT)" VIDEO_OUT="$(VIDEO_OUT)" WANDB_PROJECT="$(WANDB_PROJECT)" WANDB_ENTITY="$(WANDB_ENTITY)" WANDB_RUN_NAME="$(WANDB_RUN_NAME)" PYTHON="$(PYTHON)" WANDB_META="$(WANDB_META)" WANDB_DIR="$(WANDB_DIR)" BENCHMARK_RESULTS="$(BENCHMARK_RESULTS)" || echo "W&B upload failed (non-fatal)"; \
	else \
		echo "WANDB_UPLOAD=0 (skipped)"; \
	fi
	@echo ""
	@echo "===== RUN SUMMARY ====="
	@echo "RESULTS: $(BENCHMARK_RESULTS)"
	@echo "FRAMES: $(FRAME_OUT)"
	@echo "FRAME_EVERY: $(FRAME_EVERY)"
	@echo "VIDEOS: $(VIDEO_OUT)"
	@if [ "$(WANDB_UPLOAD)" = "1" ] && [ -f "$(WANDB_META)" ]; then \
		cat "$(WANDB_META)"; \
	fi
	@echo "======================="
