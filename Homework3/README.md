# CS4880 Homework 3 Bot Workspace

This repo contains:
- `TMA/`: original Tactical Manager AI sources (reference baseline).
- `chasebot/`: improved bot package (`ai.chasebot.ChaseBot`) used for PA3 runs.
- `scripts/`: benchmark + W&B automation scripts.

## Quick Start
Run everything with:

```bash
make run
```

## Requirements
- Local MicroRTS checkout at `../MicroRTS` (or set `MICRORTS_DIR=/path/to/MicroRTS`)
- `python3`, `java`, `javac`, `rsync`, `ffmpeg`
- W&B login if you want uploads (`wandb login`)


## Current Defaults
- `TOTAL_GAME_RUNS=6` (total games across available opponents)
- `GAMES_PER_OPP=6` (per-opponent cap)
- `FRAME_EVERY=110`
- `MAX_CYCLES=5000`
- `WANDB_UPLOAD=1`

## Outputs
- Results TSV: `chasebot/report/benchmark-results.tsv`
- Captured PNG frames: `chasebot/report/images/match1/`
- Generated MP4 videos: `chasebot/report/videos/match1/`
- W&B metadata summary: `chasebot/report/wandb-meta.txt`

## Reading Winners
In `chasebot/report/benchmark-results.tsv`:
- `winner=chase|opponent|draw`
- `perspective=1|0|-1` (`1` means ChaseBot win)

## Common Overrides
```bash
make run MICRORTS_DIR=/path/to/MicroRTS
make run WANDB_UPLOAD=0
make run TOTAL_GAME_RUNS=6 FRAME_EVERY=80
```
