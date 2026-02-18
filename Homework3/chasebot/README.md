# ChaseBot (PA3)

`ChaseBot` is a full Java bot based on the TMA architecture with stronger macro switching and safer micro.

## Bot entrypoint
- `ai.chasebot.ChaseBot`

## What `make run` now does
`make run` executes the full PA3 workflow:
1. environment setup (`venv`, Python deps)
2. MicroRTS sync/build (uses your existing local checkout)
3. benchmark matches against:
   - Random
   - WorkerRush
   - LightRush
   - NaiveMCTS
   - Mayari (if available)
   - Coac (if available)
4. frame capture image generation every `FRAME_EVERY` cycles (default `110`, 5x denser than before)
5. per-game video generation from captured frames
6. W&B upload (enabled by default, one run per game with video + frame timeline)

Outputs:
- Raw benchmark results: `chasebot/report/benchmark-results.tsv`
- Frames: `chasebot/report/images/match1/`
- Videos: `chasebot/report/videos/match1/`
- W&B metadata summary: `chasebot/report/wandb-meta.txt`
- W&B uploads: one run per game (each run has its own match video + scrollable timeline frames)

If Mayari/Coac classes are unavailable in your MicroRTS build, they are marked as skipped automatically.
