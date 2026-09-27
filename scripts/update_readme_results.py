#!/usr/bin/env python3
"""
Inject the final test metrics of a finished run into README.md.

Reads ``runs/<tag>/test_metrics.json`` (written by ``src.train`` at the end of a
run) and replaces everything between the ``<!-- RESULTS:START -->`` and
``<!-- RESULTS:END -->`` markers of README.md with a markdown table.

Usage:
    python scripts/update_readme_results.py --run runs/gtzan
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
METRIC_KEYS = ["MAE", "RMSE", "MedianAE", "Bias", "Acc_1BPM", "Acc_1%", "Acc_2%",
               "Acc_5%", "Acc_octave_1%", "Acc_octave_4%", "Octave_error_rate_4%",
               "P_score", "Cemgil"]
PERCENT_KEYS = ("Acc", "Octave_error_rate")


def cell(value: float, key: str) -> str:
    if key.startswith(PERCENT_KEYS):
        return f"{100 * value:.1f}%"
    return f"{value:.3f}"


def build_block(run_dir: Path) -> str:
    metrics_path = run_dir / "test_metrics.json"
    if not metrics_path.exists():
        raise SystemExit(
            f"{metrics_path} not found: the run has not finished yet "
            "(look for the TRAINING_COMPLETE marker)."
        )
    metrics = json.loads(metrics_path.read_text())
    report_path = run_dir / "final_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}

    lines = [
        f"Test split: **{report.get('test_tracks', '?')} held-out GTZAN tracks** "
        f"(best epoch {report.get('best_epoch', '?')}"
        f"/{report.get('epochs_requested', '?')}, "
        f"training time {report.get('total_human', '?')}).",
        "",
        "| metric | mean output | frame only | song only |",
        "|---|---|---|---|",
    ]
    for key in METRIC_KEYS:
        cells = [cell(metrics[f"{path}/{key}"], key)
                 for path in ("mean", "frame", "song") if f"{path}/{key}" in metrics]
        if len(cells) == 3:
            lines.append(f"| **{key}** | " + " | ".join(cells) + " |")

    lines += [
        "",
        f"* octave-tolerant accuracy (within 4 %, factor 1/2 or 2): "
        f"**{100 * metrics['mean/Acc_octave_4%']:.1f}%**",
        f"* octave-error rate: **{100 * metrics['mean/Octave_error_rate_4%']:.1f}%**",
        f"* full report: `{run_dir}/RESULTS.md`, `{run_dir}/test_metrics.json`, "
        f"per-track predictions via `python -m src.evaluate`",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/gtzan", help="run directory")
    parser.add_argument("--readme", default="README.md")
    args = parser.parse_args()

    readme = Path(args.readme)
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit(f"markers {START} / {END} not found in {readme}")

    block = build_block(Path(args.run))
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f"{head}{START}\n{block}\n{END}{tail}", encoding="utf-8")
    print(f"[readme] results of {args.run} written into {readme}")


if __name__ == "__main__":
    main()
