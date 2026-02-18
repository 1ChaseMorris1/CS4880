#!/usr/bin/env python3
import argparse
import html
import os
from pathlib import Path


def write_meta(meta_out: str, lines: list[str]) -> None:
    if not meta_out:
        return
    meta_path = Path(meta_out)
    try:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"WANDB_NOTE: failed to write meta file {meta_path}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload PA3 report + frames to Weights & Biases")
    parser.add_argument("--report", required=True, help="Path to markdown report")
    parser.add_argument("--images-dir", required=False, help="Directory of frame images")
    parser.add_argument("--project", default="cs4880-pa3", help="W&B project name")
    parser.add_argument("--entity", default=None, help="W&B entity/team")
    parser.add_argument("--run-name", default="chasebot-pa3-report", help="W&B run name")
    parser.add_argument("--meta-out", default="", help="Optional path to write summary key/value lines")
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        lines = [
            "WANDB_STATUS: failed",
            "WANDB_MODE: none",
            f"WANDB_PROJECT: {args.project}",
            f"WANDB_RUN_NAME: {args.run_name}",
            "WANDB_RUN_ID: N/A",
            "WANDB_RUN_URL: N/A",
            f"WANDB_NOTE: report not found: {report_path}",
        ]
        print("\n".join(lines))
        write_meta(args.meta_out, lines)
        return

    images = []
    if args.images_dir:
        images_dir = Path(args.images_dir)
        if images_dir.exists():
            images = sorted([p for p in images_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}])

    netrc = Path.home() / ".netrc"
    api_key_env = os.environ.get("WANDB_API_KEY", "").strip()
    netrc_has_wandb = netrc.exists() and ("wandb.ai" in netrc.read_text(encoding="utf-8", errors="ignore"))
    if not api_key_env and not netrc_has_wandb:
        lines = [
            "WANDB_STATUS: skipped",
            "WANDB_MODE: none",
            f"WANDB_PROJECT: {args.project}",
            f"WANDB_RUN_NAME: {args.run_name}",
            "WANDB_RUN_ID: N/A",
            "WANDB_RUN_URL: N/A",
            "WANDB_NOTE: not logged in (run `wandb login` or set WANDB_API_KEY to enable upload)",
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
            f"WANDB_RUN_NAME: {args.run_name}",
            "WANDB_RUN_ID: N/A",
            "WANDB_RUN_URL: N/A",
            f"WANDB_NOTE: wandb import failed: {exc}",
        ]
        print("\n".join(lines))
        write_meta(args.meta_out, lines)
        return

    entity = args.entity if args.entity else None
    run = None
    mode = "online"
    status = "ok"
    note = ""

    try:
        run = wandb.init(project=args.project, entity=entity, name=args.run_name)
    except Exception as first_exc:
        try:
            mode = "offline"
            run = wandb.init(project=args.project, entity=entity, name=args.run_name, mode="offline")
        except Exception as second_exc:
            status = "failed"
            mode = "none"
            note = f"wandb init failed: {first_exc} | offline fallback failed: {second_exc}"

    run_url = "N/A"
    run_id = "N/A"
    project_name = args.project

    if run is not None:
        try:
            report_text = report_path.read_text(encoding="utf-8")
            wandb.log({"report_markdown": wandb.Html(f"<pre>{html.escape(report_text)}</pre>")})

            if images:
                table = wandb.Table(columns=["idx", "image"])
                for idx, img_path in enumerate(images, start=1):
                    table.add_data(idx, wandb.Image(str(img_path), caption=img_path.name))
                wandb.log({"frame_gallery": table, "frame_count": len(images)})

            artifact = wandb.Artifact("chasebot-pa3-report", type="report")
            artifact.add_file(str(report_path), name=report_path.name)
            for img_path in images:
                artifact.add_file(str(img_path), name=f"frames/{img_path.name}")
            run.log_artifact(artifact)

            run_url = getattr(run, "url", "") or "N/A"
            run_id = getattr(run, "id", "") or "N/A"
            project_name = getattr(run, "project", "") or args.project
        except Exception as exc:
            status = "failed"
            note = f"wandb logging failed: {exc}"
        finally:
            try:
                run.finish()
            except Exception:
                pass

    lines = [
        f"WANDB_STATUS: {status}",
        f"WANDB_MODE: {mode}",
        f"WANDB_PROJECT: {project_name}",
        f"WANDB_RUN_NAME: {args.run_name}",
        f"WANDB_RUN_ID: {run_id}",
        f"WANDB_RUN_URL: {run_url}",
    ]
    if note:
        lines.append(f"WANDB_NOTE: {note}")

    print("\n".join(lines))
    write_meta(args.meta_out, lines)


if __name__ == "__main__":
    main()
