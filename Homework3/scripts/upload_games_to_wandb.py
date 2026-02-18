#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GameRow:
    opponent: str
    class_name: str
    game: int
    chase_as_p0: bool
    winner: str
    perspective: int
    cycles: int
    frames: int
    frame_prefix: str


def write_meta(meta_out: str, lines: list[str]) -> None:
    if not meta_out:
        return
    try:
        p = Path(meta_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"WANDB_NOTE: failed to write meta file {meta_out}: {exc}")


def parse_results(path: Path) -> tuple[dict[str, str], list[GameRow]]:
    meta: dict[str, str] = {}
    games: list[GameRow] = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        parts = raw.split("\t")
        if not parts:
            continue

        if parts[0] == "META":
            for i in range(1, len(parts) - 1, 2):
                meta[parts[i]] = parts[i + 1]
        elif parts[0] == "GAME":
            kv = {"opponent": parts[1]}
            for i in range(2, len(parts) - 1, 2):
                kv[parts[i]] = parts[i + 1]

            games.append(
                GameRow(
                    opponent=kv.get("opponent", "Unknown"),
                    class_name=kv.get("class", ""),
                    game=int(kv.get("game", "0")),
                    chase_as_p0=kv.get("chase_as_p0", "false").lower() == "true",
                    winner=kv.get("winner", "draw"),
                    perspective=int(kv.get("perspective", "0")),
                    cycles=int(kv.get("cycles", "0")),
                    frames=int(kv.get("frames", "0")),
                    frame_prefix=kv.get("frame_prefix", ""),
                )
            )

    return meta, games


def extract_cycle_from_frame_name(frame_path: Path) -> int:
    m = re.search(r"_t(\d+)_", frame_path.name)
    if not m:
        return -1
    try:
        return int(m.group(1))
    except ValueError:
        return -1


def frame_sort_key(frame_path: Path) -> tuple[int, str]:
    cycle = extract_cycle_from_frame_name(frame_path)
    if cycle < 0:
        cycle = 10**9
    return cycle, frame_path.name


def build_video_from_frames(frames: list[Path], output_video: Path, fps: int) -> bool:
    if not frames:
        return False

    output_video.parent.mkdir(parents=True, exist_ok=True)
    pattern = str(frames[0].parent / f"{frames[0].name.split('_t')[0]}_t*.png")

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(max(1, fps)),
        "-pattern_type",
        "glob",
        "-i",
        pattern,
        "-vf",
        "format=yuv420p",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_video),
    ]

    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        if proc.stderr:
            print(f"WANDB_NOTE: ffmpeg failed for {output_video.name}: {proc.stderr.strip()}")
        return False
    return output_video.exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload one W&B run per game from benchmark results")
    parser.add_argument("--results", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--videos-dir", required=True)
    parser.add_argument("--project", default="cs4880-pa3")
    parser.add_argument("--entity", default="")
    parser.add_argument("--run-name-base", default="chasebot-game")
    parser.add_argument("--meta-out", default="")
    parser.add_argument("--video-fps", type=int, default=8)
    args = parser.parse_args()

    results_path = Path(args.results)
    frames_dir = Path(args.frames_dir)
    videos_dir = Path(args.videos_dir)
    videos_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        lines = [
            "WANDB_STATUS: failed",
            "WANDB_MODE: none",
            f"WANDB_PROJECT: {args.project}",
            f"WANDB_RUN_NAME_BASE: {args.run_name_base}",
            "WANDB_GAME_RUNS: 0",
            "WANDB_NOTE: benchmark results file missing",
        ]
        print("\n".join(lines))
        write_meta(args.meta_out, lines)
        return

    netrc = Path.home() / ".netrc"
    api_key_env = os.environ.get("WANDB_API_KEY", "").strip()
    netrc_has_wandb = netrc.exists() and ("wandb.ai" in netrc.read_text(encoding="utf-8", errors="ignore"))
    if not api_key_env and not netrc_has_wandb:
        lines = [
            "WANDB_STATUS: skipped",
            "WANDB_MODE: none",
            f"WANDB_PROJECT: {args.project}",
            f"WANDB_RUN_NAME_BASE: {args.run_name_base}",
            "WANDB_GAME_RUNS: 0",
            "WANDB_NOTE: not logged in (run `wandb login` or set WANDB_API_KEY)",
        ]
        print("\n".join(lines))
        write_meta(args.meta_out, lines)
        return

    if not shutil.which("ffmpeg"):
        lines = [
            "WANDB_STATUS: failed",
            "WANDB_MODE: none",
            f"WANDB_PROJECT: {args.project}",
            f"WANDB_RUN_NAME_BASE: {args.run_name_base}",
            "WANDB_GAME_RUNS: 0",
            "WANDB_NOTE: ffmpeg not found (required for video upload)",
        ]
        print("\n".join(lines))
        write_meta(args.meta_out, lines)
        return

    try:
        import wandb
    except Exception as exc:
        lines = [
            "WANDB_STATUS: failed",
            "WANDB_MODE: none",
            f"WANDB_PROJECT: {args.project}",
            f"WANDB_RUN_NAME_BASE: {args.run_name_base}",
            "WANDB_GAME_RUNS: 0",
            f"WANDB_NOTE: wandb import failed: {exc}",
        ]
        print("\n".join(lines))
        write_meta(args.meta_out, lines)
        return

    meta, games = parse_results(results_path)
    games.sort(key=lambda g: (g.opponent, g.game))

    entity = args.entity if args.entity else None
    mode = "online"
    urls: list[str] = []
    failures = 0

    for g in games:
        run_name = f"{args.run_name_base}-{g.opponent}-g{g.game:02d}"
        run = None

        try:
            run = wandb.init(project=args.project, entity=entity, name=run_name, reinit="finish_previous")
        except Exception:
            try:
                mode = "offline"
                run = wandb.init(project=args.project, entity=entity, name=run_name, mode="offline", reinit="finish_previous")
            except Exception:
                failures += 1
                continue

        try:
            frames = sorted(frames_dir.glob(f"{g.frame_prefix}*.png"), key=frame_sort_key) if g.frame_prefix else []
            video_path = videos_dir / f"{g.frame_prefix.rstrip('_')}.mp4"
            has_video = build_video_from_frames(frames, video_path, args.video_fps)

            wandb.log(
                {
                    "result_code": g.perspective,
                    "cycles": g.cycles,
                    "frames_total": len(frames),
                    "has_video": 1 if has_video else 0,
                    "winner_label": {"chase": 1, "draw": 0, "opponent": -1}.get(g.winner, 0),
                }
            )

            if has_video:
                wandb.log(
                    {
                        "match_video": wandb.Video(str(video_path), fps=max(1, args.video_fps), format="mp4"),
                    }
                )

            # Emit per-step frame media so W&B shows a scrollable timeline for the game.
            if frames:
                for idx, frame in enumerate(frames, start=1):
                    cycle = extract_cycle_from_frame_name(frame)
                    wandb.log(
                        {
                            "timeline_frame": wandb.Image(
                                str(frame),
                                caption=f"{g.opponent} g{g.game:02d} frame {idx}/{len(frames)} cycle={cycle}",
                            ),
                            "frame_index": idx,
                            "frame_cycle": cycle,
                            "frame_progress_pct": (100.0 * idx / len(frames)),
                        },
                        step=idx,
                    )

            run.summary["opponent"] = g.opponent
            run.summary["opponent_class"] = g.class_name
            run.summary["game_index"] = g.game
            run.summary["chase_as_p0"] = g.chase_as_p0
            run.summary["map"] = meta.get("map", "")
            run.summary["max_cycles"] = int(meta.get("max_cycles", "0") or 0)
            run.summary["frames_logged"] = len(frames)
            run.summary["video_path"] = str(video_path) if has_video else ""

            url = getattr(run, "url", "") or "N/A"
            urls.append(url)
            print(f"WANDB_GAME_RUN: {run_name} -> {url}")
            print(f"WANDB_GAME_MEDIA: {run_name} frames={len(frames)} video={'yes' if has_video else 'no'}")
        except Exception:
            failures += 1
        finally:
            try:
                run.finish()
            except Exception:
                pass

    status = "ok" if failures == 0 else ("partial" if urls else "failed")
    lines = [
        f"WANDB_STATUS: {status}",
        f"WANDB_MODE: {mode}",
        f"WANDB_PROJECT: {args.project}",
        f"WANDB_RUN_NAME_BASE: {args.run_name_base}",
        f"WANDB_GAME_RUNS: {len(urls)}",
    ]
    for i, url in enumerate(urls, start=1):
        lines.append(f"WANDB_RUN_URL_{i:02d}: {url}")
    if failures:
        lines.append(f"WANDB_NOTE: failed_runs={failures}")

    print("\n".join(lines))
    write_meta(args.meta_out, lines)


if __name__ == "__main__":
    main()
