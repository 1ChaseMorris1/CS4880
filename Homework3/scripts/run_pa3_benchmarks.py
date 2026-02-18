#!/usr/bin/env python3
import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OpponentRow:
    name: str
    available: bool
    class_name: str
    wins: int
    losses: int
    draws: int
    games_played: int
    reason: str


def run_cmd(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)
        raise SystemExit(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def collect_compile_sources(microrts_dir: Path, chase_class: str) -> list[str]:
    sources: set[str] = set()

    # Always include the benchmark harness
    harness = microrts_dir / "src" / "tests" / "ChasePA3Runner.java"
    if harness.exists():
        sources.add(harness.relative_to(microrts_dir).as_posix())

    # Compile all chasebot sources recursively
    chase_dir = microrts_dir / "src" / "ai" / "chasebot"
    if chase_dir.exists():
        for p in chase_dir.rglob("*.java"):
            sources.add(p.relative_to(microrts_dir).as_posix())

    candidate_classes = [
        chase_class,
        "ai.RandomBiasedAI",
        "ai.RandomAI",
        "ai.abstraction.RandomBiasedAI",
        "ai.abstraction.WorkerRush",
        "ai.abstraction.LightRush",
        "ai.mcts.naivemcts.NaiveMCTS",
        "ai.mayari.Mayari",
        "ai.mayari.MayariBot",
        "mayari.MayariBot",
        "ai.mayaribot.MayariBot",
        "ai.coac.CoacAI",
        "ai.coac.Coac",
        "coac.CoacAI",
        "ai.competition.coac.CoacAI",
    ]

    for cls in candidate_classes:
        src_rel = Path("src") / Path(*cls.split("."))
        src_file = microrts_dir / (str(src_rel) + ".java")
        if src_file.exists():
            sources.add(src_file.relative_to(microrts_dir).as_posix())

    return sorted(sources)


def parse_results(path: Path) -> tuple[dict[str, str], dict[str, OpponentRow]]:
    meta: dict[str, str] = {}
    rows: dict[str, OpponentRow] = {}

    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        parts = raw.split("\t")
        if not parts:
            continue

        if parts[0] == "META":
            for i in range(1, len(parts) - 1, 2):
                meta[parts[i]] = parts[i + 1]
        elif parts[0] == "OPP":
            name = parts[1]
            kv = {}
            for i in range(2, len(parts) - 1, 2):
                kv[parts[i]] = parts[i + 1]

            rows[name] = OpponentRow(
                name=name,
                available=kv.get("available", "false").lower() == "true",
                class_name=kv.get("class", ""),
                wins=int(kv.get("wins", "0")),
                losses=int(kv.get("losses", "0")),
                draws=int(kv.get("draws", "0")),
                games_played=int(kv.get("games_played", "0")),
                reason=kv.get("reason", ""),
            )

    return meta, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PA3 benchmark matches in MicroRTS")
    parser.add_argument("--microrts-dir", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--frame-every", type=int, default=110)
    parser.add_argument("--games", type=int, default=6)
    parser.add_argument("--max-total-games", type=int, default=6)
    parser.add_argument("--max-cycles", type=int, default=5000)
    parser.add_argument("--chase-class", default="ai.chasebot.ChaseBot")
    parser.add_argument("--map", default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    microrts_dir = Path(args.microrts_dir).resolve()
    results_path = Path(args.results).resolve()
    frames_dir = Path(args.frames_dir).resolve()

    if not (microrts_dir / "src").exists():
        raise SystemExit(f"Invalid MicroRTS directory: {microrts_dir}")

    harness_src = repo_root / "chasebot" / "harness" / "ChasePA3Runner.java"
    if not harness_src.exists():
        raise SystemExit(f"Harness source missing: {harness_src}")

    harness_dst = microrts_dir / "src" / "tests" / "ChasePA3Runner.java"
    harness_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(harness_src, harness_dst)

    frames_dir.mkdir(parents=True, exist_ok=True)
    for p in frames_dir.glob("*.png"):
        p.unlink()

    compile_sources = collect_compile_sources(microrts_dir, args.chase_class)
    if not compile_sources:
        raise SystemExit("No Java sources found to compile (expected at least ChasePA3Runner.java)")

    print(f"BENCH: compiling {len(compile_sources)} Java files")
    run_cmd(["javac", "-cp", "lib/*:src", "-d", "bin", *compile_sources], cwd=microrts_dir)

    cmd = [
        "java",
        "-Djava.awt.headless=true",
        "-cp",
        "lib/*:bin",
        "tests.ChasePA3Runner",
        "--chase-class",
        args.chase_class,
        "--games",
        str(args.games),
        "--max-total-games",
        str(args.max_total_games),
        "--max-cycles",
        str(args.max_cycles),
        "--frame-every",
        str(args.frame_every),
        "--frames-dir",
        str(frames_dir),
        "--results",
        str(results_path),
    ]
    if args.map:
        cmd.extend(["--map", args.map])

    print("BENCH: running matches")
    run_cmd(cmd, cwd=microrts_dir)

    if not results_path.exists():
        raise SystemExit(f"Benchmark results file not produced: {results_path}")

    meta, rows = parse_results(results_path)
    completed = [r for r in rows.values() if r.available]
    skipped = [r for r in rows.values() if not r.available]

    print(f"BENCH: completed opponents={len(completed)} skipped={len(skipped)}")
    for r in completed:
        total = r.wins + r.losses + r.draws
        win_rate = (100.0 * r.wins / total) if total else 0.0
        print(f"BENCH: {r.name}: GP={r.games_played} W={r.wins} L={r.losses} D={r.draws} WR={win_rate:.1f}%")
    for r in skipped:
        print(f"BENCH: {r.name}: skipped ({r.reason})")


if __name__ == "__main__":
    main()
