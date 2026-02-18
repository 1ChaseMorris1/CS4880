#!/usr/bin/env python3
import argparse
from pathlib import Path

START = "<!-- FRAME_CAPTURES_START -->"
END = "<!-- FRAME_CAPTURES_END -->"


def build_section(report_path: Path, images_dir: Path, frame_every: int, title: str) -> str:
    images = sorted([p for p in images_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}])
    lines = []
    lines.append(f"### {title}")
    lines.append(f"- Sampling rate: every {frame_every} game cycles")
    lines.append(f"- Frame count: {len(images)}")
    lines.append("")

    if not images:
        lines.append("No frames found. Run `make run-benchmarks` first.")
        return "\n".join(lines)

    for idx, image_path in enumerate(images, start=1):
        rel = image_path.relative_to(report_path.parent)
        lines.append(f"![Frame {idx}]({rel.as_posix()})")

    return "\n".join(lines)


def inject_section(report_text: str, new_section: str) -> str:
    start_idx = report_text.find(START)
    end_idx = report_text.find(END)

    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        append_block = f"\n\n## 8. Frame Capture\n{START}\n{new_section}\n{END}\n"
        return report_text.rstrip() + append_block

    head = report_text[: start_idx + len(START)]
    tail = report_text[end_idx:]
    return f"{head}\n{new_section}\n{tail}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject sampled frame images into PA3 report markdown")
    parser.add_argument("--report", required=True, help="Path to report markdown file")
    parser.add_argument("--images-dir", required=True, help="Directory containing sampled frames")
    parser.add_argument("--frame-every", type=int, default=120, help="Sampling stride used for capture")
    parser.add_argument("--title", default="Match Frames", help="Subsection title")
    args = parser.parse_args()

    report_path = Path(args.report)
    images_dir = Path(args.images_dir)

    if not report_path.exists():
        raise SystemExit(f"Report not found: {report_path}")

    images_dir.mkdir(parents=True, exist_ok=True)

    section = build_section(report_path, images_dir, args.frame_every, args.title)
    text = report_path.read_text(encoding="utf-8")
    updated = inject_section(text, section)
    report_path.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
