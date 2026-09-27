"""
Evaluation of a trained checkpoint: full metric set + plots + per-genre table.

Usage
-----
  python -m src.evaluate --ckpt runs/gtzan/best.pth --cache-dir data/cache \
      --out-dir runs/gtzan --plots-dir assets
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")                                  # headless / background safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .dataset import build_dataloaders
from .metrics import format_metrics, octave_factor, tempo_metrics
from .model import build_model_from_config
from .utils import get_device, human_time, load_checkpoint, save_json

PATHS = ("mean", "frame", "song")


@torch.no_grad()
def collect_predictions(model, loader, device: torch.device
                        ) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[int]]:
    """Run the model over a loader; returns (predictions per path, targets, order)."""
    model.eval()
    bucket: Dict[str, List[np.ndarray]] = {path: [] for path in PATHS}
    references: List[np.ndarray] = []
    order: List[int] = []
    offset = 0
    for x, y in loader:
        out = model(x.to(device, non_blocking=True), return_all=True)
        bucket["mean"].append(out["bpm"].detach().float().cpu().numpy())
        bucket["frame"].append(out["bpm_frame"].detach().float().cpu().numpy())
        bucket["song"].append(out["bpm_song"].detach().float().cpu().numpy())
        references.append(y.squeeze(-1).float().cpu().numpy())
        order.extend(range(offset, offset + x.size(0)))
        offset += x.size(0)
    return ({k: np.concatenate(v) for k, v in bucket.items()},
            np.concatenate(references), order)


def per_genre_table(pred: np.ndarray, reference: np.ndarray,
                    genres: Sequence[str]) -> pd.DataFrame:
    frame = pd.DataFrame({"genre": list(genres), "pred": pred, "ref": reference})
    rows = []
    for genre, sub in frame.groupby("genre", sort=True):
        metrics = tempo_metrics(sub["pred"].to_numpy(), sub["ref"].to_numpy())
        rows.append({
            "genre": genre, "n": metrics["n"],
            "MAE": round(metrics["MAE"], 3),
            "Acc_1%": round(100 * metrics["Acc_1%"], 1),
            "Acc_5%": round(100 * metrics["Acc_5%"], 1),
            "Acc_oct_4%": round(100 * metrics["Acc_octave_4%"], 1),
            "oct_err_4%": round(100 * metrics["Octave_error_rate_4%"], 1),
        })
    return pd.DataFrame(rows)


def plot_predictions(pred: np.ndarray, reference: np.ndarray, out_png: Path,
                     title: str = "BPM predictions") -> None:
    errors = pred - reference
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].scatter(reference, pred, s=12, alpha=0.55, edgecolors="none")
    limits = [min(reference.min(), pred.min()) - 5, max(reference.max(), pred.max()) + 5]
    axes[0].plot(limits, limits, "k--", lw=1, label="identity")
    axes[0].plot(limits, [x / 2 for x in limits], ":", color="gray", label="half tempo")
    axes[0].plot(limits, [x * 2 for x in limits], ":", color="gray", label="double tempo")
    axes[0].set_xlabel("reference BPM")
    axes[0].set_ylabel("predicted BPM")
    axes[0].set_title(title)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].hist(errors, bins=41, alpha=0.8, color="#2b6cb0")
    axes[1].axvline(0, color="k", lw=1)
    axes[1].set_xlabel("prediction - reference (BPM)")
    axes[1].set_ylabel("tracks")
    axes[1].set_title(f"error distribution (MAE={np.abs(errors).mean():.2f} BPM)")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_training_curves(history_csv: Path, out_png: Path) -> None:
    if not history_csv.exists():
        return
    history = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    if "val_loss" in history:
        axes[0].plot(history["epoch"], history["val_loss"], label="val")
    axes[0].set_title("loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["val_MAE"], color="#c53030")
    axes[1].set_title("validation MAE (BPM)")
    axes[1].set_xlabel("epoch")
    axes[2].plot(history["epoch"], 100 * history["val_Acc_1%"], label="Acc@1%")
    axes[2].plot(history["epoch"], 100 * history["val_Acc_5%"], label="Acc@5%")
    axes[2].plot(history["epoch"], 100 * history["val_Acc_octave_4%"], label="Acc octave@4%")
    axes[2].set_title("validation accuracy (%)")
    axes[2].set_xlabel("epoch")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a BeatNet-CRNN checkpoint")
    parser.add_argument("--ckpt", required=True, help="path to best.pth / last.pth")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--out-dir", default=None, help="where to write the JSON report")
    parser.add_argument("--plots-dir", default="assets")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--crop-seconds", type=float, default=10.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent
    device = get_device(args.device)

    checkpoint = load_checkpoint(ckpt_path, map_location=device)
    model = build_model_from_config(checkpoint.get("arch") or {}).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    bundle = build_dataloaders(
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        crop_seconds=args.crop_seconds,
        num_workers=args.num_workers,
        synthetic=0,
        augment=False,
    )
    dataset = bundle.test_dataset if args.split == "test" else bundle.val_dataset
    loader = bundle.test_loader if args.split == "test" else bundle.val_loader

    print(f"[eval] {ckpt_path} | split={args.split} | {len(dataset)} tracks "
          f"| device={device.type}")
    predictions, reference, order = collect_predictions(model, loader, device)
    assert order == list(range(len(dataset))), "loader order mismatch"

    report: Dict[str, object] = {
        "checkpoint": str(ckpt_path),
        "epoch": checkpoint.get("epoch"),
        "split": args.split,
        "n_tracks": int(len(reference)),
        "crop_seconds": args.crop_seconds,
        "train_config": checkpoint.get("config") or {},
        "metrics": {},
    }

    print("=" * 78)
    for path in PATHS:
        metrics = tempo_metrics(predictions[path], reference)
        report["metrics"][path] = metrics          # type: ignore[index]
        print(f"[{path} output]")
        print(format_metrics(metrics, prefix="  "))
        print("-" * 78)

    tracks = dataset.index[["track_id", "genre", "bpm"]].reset_index(drop=True)
    table = per_genre_table(predictions["mean"], reference, tracks["genre"].tolist())
    print("PER-GENRE BREAKDOWN (mean output)")
    print(table.to_string(index=False))
    report["per_genre"] = table.to_dict(orient="records")

    factors = octave_factor(predictions["mean"], reference)
    factor_counts = pd.Series(factors).value_counts().sort_index()
    report["octave_factors"] = {str(k): int(v) for k, v in factor_counts.items()}
    print("\nOCTAVE ASSIGNMENT (relative factor vs reference)")
    for factor, count in factor_counts.items():
        print(f"  factor {factor:>4}: {count:>4} tracks "
              f"({100.0 * count / len(factors):5.1f}%)")

    details = pd.DataFrame({
        "track_id": tracks["track_id"].to_numpy(),
        "genre": tracks["genre"].to_numpy(),
        "bpm_ref": reference,
        "bpm_pred": predictions["mean"],
        "error": predictions["mean"] - reference,
        "octave_factor": factors,
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(out_dir / f"predictions_{args.split}.csv", index=False)
    print(f"\n[eval] per-track predictions -> {out_dir / f'predictions_{args.split}.csv'}")

    if not args.no_plots:
        mean_metrics = report["metrics"]["mean"]              # type: ignore[index]
        plot_predictions(predictions["mean"], reference,
                         Path(args.plots_dir) / f"predictions_{args.split}.png",
                         title=f"BeatNet-CRNN - {args.split} split "
                               f"(MAE={mean_metrics['MAE']:.2f} BPM)")
        plot_training_curves(out_dir / "history.csv",
                             Path(args.plots_dir) / "training_curves.png")
        print(f"[eval] plots -> {args.plots_dir}")

    save_json(out_dir / f"metrics_report_{args.split}.json", report)
    print(f"[eval] report -> {out_dir / f'metrics_report_{args.split}.json'}")


if __name__ == "__main__":
    main()
