from __future__ import annotations

import argparse
import csv
from pathlib import Path


BG = "#ffffff"
FG = "#1f2937"
GRID = "#e5e7eb"
BLUE = "#2563eb"
ORANGE = "#f59e0b"
GREEN = "#16a34a"
RED = "#dc2626"
PURPLE = "#7c3aed"
GRAY = "#64748b"


class Svg:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.items: list[str] = []

    def rect(self, x: float, y: float, w: float, h: float, fill: str, stroke: str | None = None, sw: float = 1.0) -> None:
        stroke_attr = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        self.items.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}"{stroke_attr}/>'
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, stroke: str, sw: float = 1.0, dash: str | None = None) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}"{dash_attr}/>'
        )

    def circle(self, cx: float, cy: float, r: float, fill: str, stroke: str | None = None, sw: float = 1.0) -> None:
        stroke_attr = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        self.items.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}"{stroke_attr}/>'
        )

    def text(self, x: float, y: float, text: str, size: int = 14, fill: str = FG, anchor: str = "start", weight: str = "normal") -> None:
        escaped = (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
        self.items.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{escaped}</text>'
        )

    def polyline(self, points: list[tuple[float, float]], stroke: str, sw: float = 2.5) -> None:
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        self.items.append(
            f'<polyline points="{coords}" fill="none" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/>'
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(self.items)
        path.write_text(
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}">\n'
                f'<rect x="0" y="0" width="{self.width}" height="{self.height}" fill="{BG}"/>\n'
                f"{content}\n"
                "</svg>\n"
            ),
            encoding="utf-8",
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _draw_axes(svg: Svg, x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    svg.rect(x, y, w, h, fill="#fafafa", stroke="#d1d5db", sw=1)
    for i in range(6):
        yy = y + (h * i / 5)
        svg.line(x, yy, x + w, yy, GRID, 1)
    svg.line(x, y + h, x + w, y + h, FG, 1.4)
    svg.line(x, y, x, y + h, FG, 1.4)
    return (x, y, w, h)


def _legend(svg: Svg, items: list[tuple[str, str]], x: float, y: float) -> None:
    cur_x = x
    for label, color in items:
        svg.rect(cur_x, y - 10, 14, 14, fill=color)
        svg.text(cur_x + 20, y + 1, label, size=13)
        cur_x += 140


def chart_baseline_vs_llm_winrates(summary_rows: list[dict[str, str]], out_path: Path) -> None:
    rows = [r for r in summary_rows if r["matchup"] == "baseline_vs_llm"]
    rows.sort(key=lambda r: int(r["iterations"]))
    if not rows:
        return

    svg = Svg(1100, 620)
    svg.text(40, 40, "Baseline vs LLM-Guided: Win Rate by Iterations", size=28, weight="bold")
    svg.text(40, 66, "Higher is better for each corresponding agent", size=15, fill=GRAY)

    x0, y0, w, h = _draw_axes(svg, 90, 95, 940, 420)

    for pct in range(0, 101, 20):
        yy = y0 + h - (pct / 100.0) * h
        svg.text(x0 - 12, yy + 5, f"{pct}%", size=12, anchor="end", fill=GRAY)

    n = len(rows)
    xs: list[float] = []
    baseline_pts: list[tuple[float, float]] = []
    llm_pts: list[tuple[float, float]] = []
    draw_pts: list[tuple[float, float]] = []

    for i, row in enumerate(rows):
        x = x0 + (i * (w / max(1, n - 1)))
        it = row["iterations"]
        b = float(row["a_win_rate"])
        l = float(row["b_win_rate"])
        d = float(row["draw_rate"])
        xs.append(x)
        baseline_pts.append((x, y0 + h - b * h))
        llm_pts.append((x, y0 + h - l * h))
        draw_pts.append((x, y0 + h - d * h))
        svg.text(x, y0 + h + 24, str(it), size=13, anchor="middle")

    svg.polyline(baseline_pts, stroke=BLUE, sw=3)
    svg.polyline(llm_pts, stroke=RED, sw=3)
    svg.polyline(draw_pts, stroke=GRAY, sw=2.5)

    for x, y in baseline_pts:
        svg.circle(x, y, 4, fill=BLUE)
    for x, y in llm_pts:
        svg.circle(x, y, 4, fill=RED)
    for x, y in draw_pts:
        svg.circle(x, y, 3.5, fill=GRAY)

    _legend(svg, [("Baseline win %", BLUE), ("LLM-guided win %", RED), ("Draw %", GRAY)], 120, 560)
    svg.text(560, 604, "MCTS iterations per move", size=14, anchor="middle", fill=GRAY)

    svg.save(out_path)


def chart_baseline_vs_llm_stacked(summary_rows: list[dict[str, str]], out_path: Path) -> None:
    rows = [r for r in summary_rows if r["matchup"] == "baseline_vs_llm"]
    rows.sort(key=lambda r: int(r["iterations"]))
    if not rows:
        return

    svg = Svg(1100, 620)
    svg.text(40, 40, "Outcome Composition: Baseline vs LLM-Guided", size=28, weight="bold")
    svg.text(40, 66, "Each bar sums to 100% (Baseline wins / Draws / LLM wins)", size=15, fill=GRAY)

    x0, y0, w, h = _draw_axes(svg, 90, 95, 940, 420)
    for pct in range(0, 101, 20):
        yy = y0 + h - (pct / 100.0) * h
        svg.text(x0 - 12, yy + 5, f"{pct}%", size=12, anchor="end", fill=GRAY)

    n = len(rows)
    bar_w = min(120.0, (w / max(1, n)) * 0.55)
    gap = (w - n * bar_w) / (n + 1)

    for i, row in enumerate(rows):
        x = x0 + gap + i * (bar_w + gap)
        b = float(row["a_win_rate"])
        d = float(row["draw_rate"])
        l = float(row["b_win_rate"])

        bh = b * h
        dh = d * h
        lh = l * h

        y_base = y0 + h
        y_b = y_base - bh
        y_d = y_b - dh
        y_l = y_d - lh

        svg.rect(x, y_b, bar_w, bh, BLUE)
        svg.rect(x, y_d, bar_w, dh, ORANGE)
        svg.rect(x, y_l, bar_w, lh, GREEN)

        svg.text(x + bar_w / 2, y0 + h + 24, row["iterations"], size=13, anchor="middle")

    _legend(svg, [("Baseline wins", BLUE), ("Draws", ORANGE), ("LLM-guided wins", GREEN)], 120, 560)
    svg.text(560, 604, "MCTS iterations per move", size=14, anchor="middle", fill=GRAY)

    svg.save(out_path)


def chart_vs_minimax_draw_rates(summary_rows: list[dict[str, str]], out_path: Path) -> None:
    base_rows = [r for r in summary_rows if r["matchup"] == "baseline_vs_minimax"]
    llm_rows = [r for r in summary_rows if r["matchup"] == "llm_vs_minimax"]
    base_rows.sort(key=lambda r: int(r["iterations"]))
    llm_rows.sort(key=lambda r: int(r["iterations"]))
    if not base_rows and not llm_rows:
        return

    all_iters = sorted({int(r["iterations"]) for r in base_rows + llm_rows})

    svg = Svg(1100, 620)
    svg.text(40, 40, "Draw Rate vs Perfect Minimax", size=28, weight="bold")
    svg.text(40, 66, "Higher draw rate means closer to optimal play", size=15, fill=GRAY)

    x0, y0, w, h = _draw_axes(svg, 90, 95, 940, 420)
    for pct in range(0, 101, 20):
        yy = y0 + h - (pct / 100.0) * h
        svg.text(x0 - 12, yy + 5, f"{pct}%", size=12, anchor="end", fill=GRAY)

    def to_points(rows: list[dict[str, str]]) -> list[tuple[float, float]]:
        index = {int(r["iterations"]): float(r["draw_rate"]) for r in rows}
        points: list[tuple[float, float]] = []
        for i, it in enumerate(all_iters):
            x = x0 + (i * (w / max(1, len(all_iters) - 1)))
            y = y0 + h - index.get(it, 0.0) * h
            points.append((x, y))
            svg.text(x, y0 + h + 24, str(it), size=13, anchor="middle")
        return points

    base_pts = to_points(base_rows)
    llm_pts = to_points(llm_rows)

    if base_pts:
        svg.polyline(base_pts, stroke=PURPLE, sw=3)
        for x, y in base_pts:
            svg.circle(x, y, 4, fill=PURPLE)
    if llm_pts:
        svg.polyline(llm_pts, stroke=GREEN, sw=3)
        for x, y in llm_pts:
            svg.circle(x, y, 4, fill=GREEN)

    _legend(svg, [("Baseline vs minimax draw %", PURPLE), ("LLM vs minimax draw %", GREEN)], 120, 560)
    svg.text(560, 604, "MCTS iterations per move", size=14, anchor="middle", fill=GRAY)

    svg.save(out_path)


def chart_timing(move_rows: list[dict[str, str]], out_path: Path) -> None:
    rows = [r for r in move_rows if r["matchup"] == "baseline_vs_llm"]
    if not rows:
        return

    grouped: dict[int, dict[str, float]] = {}
    for r in rows:
        it = int(r["iterations"])
        grouped.setdefault(it, {})[r["agent"]] = float(r["avg_ms_per_iteration"])

    iterations = sorted(grouped.keys())
    svg = Svg(1100, 620)
    svg.text(40, 40, "Computation Cost: Avg Milliseconds per Iteration", size=28, weight="bold")
    svg.text(40, 66, "Lower is faster", size=15, fill=GRAY)

    x0, y0, w, h = _draw_axes(svg, 90, 95, 940, 420)

    max_ms = 0.0
    for it in iterations:
        for agent, value in grouped[it].items():
            if "baseline" in agent or "llm" in agent:
                max_ms = max(max_ms, value)
    if max_ms <= 0:
        max_ms = 0.02

    for i in range(6):
        v = max_ms * i / 5
        yy = y0 + h - (v / max_ms) * h
        svg.text(x0 - 12, yy + 5, f"{v:.3f}", size=12, anchor="end", fill=GRAY)

    n = len(iterations)
    group_w = min(170.0, w / max(1, n) * 0.72)
    bar_w = group_w * 0.40
    gap = (w - n * group_w) / (n + 1)

    for i, it in enumerate(iterations):
        x_group = x0 + gap + i * (group_w + gap)
        baseline = 0.0
        llm = 0.0
        for agent, value in grouped[it].items():
            if "baseline" in agent:
                baseline = value
            elif "llm" in agent:
                llm = value

        bh = (baseline / max_ms) * h
        lh = (llm / max_ms) * h
        svg.rect(x_group, y0 + h - bh, bar_w, bh, BLUE)
        svg.rect(x_group + bar_w + 8, y0 + h - lh, bar_w, lh, RED)

        svg.text(x_group + group_w / 2, y0 + h + 24, str(it), size=13, anchor="middle")

    _legend(svg, [("Baseline ms/iteration", BLUE), ("LLM-guided ms/iteration", RED)], 120, 560)
    svg.text(560, 604, "MCTS iterations per move", size=14, anchor="middle", fill=GRAY)

    svg.save(out_path)


def generate_all(summary_dir: Path) -> list[Path]:
    summary_csv = summary_dir / "summary.csv"
    move_csv = summary_dir / "move_metrics.csv"
    if not summary_csv.exists() or not move_csv.exists():
        raise FileNotFoundError("Missing summary.csv or move_metrics.csv in summary directory")

    summary_rows = _read_csv(summary_csv)
    move_rows = _read_csv(move_csv)

    figs_dir = summary_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        figs_dir / "01_baseline_vs_llm_winrates.svg",
        figs_dir / "02_baseline_vs_llm_outcome_stack.svg",
        figs_dir / "03_draw_rate_vs_minimax.svg",
        figs_dir / "04_iteration_cost.svg",
    ]

    chart_baseline_vs_llm_winrates(summary_rows, outputs[0])
    chart_baseline_vs_llm_stacked(summary_rows, outputs[1])
    chart_vs_minimax_draw_rates(summary_rows, outputs[2])
    chart_timing(move_rows, outputs[3])
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate clean SVG charts for Homework 4 results")
    parser.add_argument("--summary-dir", default="solutions/summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_dir = Path(args.summary_dir)
    outputs = generate_all(summary_dir)
    print("Generated figures:")
    for path in outputs:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
