"""
Training loop for BeatNet-CRNN (BPM estimation).

Losses
------
  * ``w_reg``  SmoothL1 on BPM (linear, beta = 1 BPM)
  * ``w_log``  SmoothL1 on log2(BPM)  -> tempo aware, octave-tolerant gradient
  * ``w_song`` soft-label cross entropy on the song-level BPM distribution
  * ``w_frame`` soft-label cross entropy on the per-frame BPM distributions
               (multi-frame supervision: helps local tempo tracking)

Soft targets are Gaussians over the BPM bins (``--sigma-bins``), which is what
makes the auxiliary cross entropies act as *calibrated* regressors.

Artifacts written to ``--out``:
  best.pth  last.pth  history.csv  train.log  config.json
  status.json                  -> live progress (poll this to monitor)
  TRAINING_COMPLETE            -> marker written when the run ends

Usage
-----
  # smoke test (fast sanity run)
  python -m src.train --out runs/smoke --epochs 2 --limit 40 --synthetic 20 --num-workers 2

  # full run
  python -m src.train --out runs/gtzan --epochs 60 --synthetic 500
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from . import __version__
from .dataset import CROP_SECONDS, build_dataloaders, set_epoch
from .metrics import format_metrics, tempo_metrics
from .model import (DEFAULT_BPM_MAX, DEFAULT_BPM_MIN, DEFAULT_N_CLASSES,
                    BeatNetCRNN, gaussian_soft_targets)
from .utils import (AverageMeter, Logger, describe_device, get_device,
                    human_time, load_checkpoint, save_checkpoint, save_json,
                    set_seed)


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #
def soft_target_ce(logits: torch.Tensor, bpm: torch.Tensor,
                   bins: torch.Tensor, sigma_bins: float) -> torch.Tensor:
    """Cross entropy against Gaussian soft targets over the BPM bins."""
    with torch.no_grad():
        target = gaussian_soft_targets(bpm, bins, sigma_bins)
    return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def compute_loss(out: Dict[str, torch.Tensor], bpm_true: torch.Tensor,
                 bins: torch.Tensor, args: argparse.Namespace
                 ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Total loss + its components (for logging)."""
    bpm_pred = out["bpm"]
    loss_reg = F.smooth_l1_loss(bpm_pred, bpm_true, beta=1.0)
    loss_log = F.smooth_l1_loss(
        torch.log2(bpm_pred.clamp_min(1.0)), torch.log2(bpm_true.clamp_min(1.0)), beta=0.02,
    )
    loss_song = soft_target_ce(out["song_logits"], bpm_true, bins, args.sigma_bins)

    frame_logits = out["frame_logits"]                      # (B, T, C)
    frame_targets = bpm_true.unsqueeze(1).expand(-1, frame_logits.shape[1])
    loss_frame = soft_target_ce(frame_logits.reshape(-1, frame_logits.shape[-1]),
                                frame_targets.reshape(-1), bins, args.sigma_bins)

    total = (args.w_reg * loss_reg + args.w_log * loss_log
             + args.w_song * loss_song + args.w_frame * loss_frame)
    parts = {
        "reg": float(loss_reg.detach()), "log": float(loss_log.detach()),
        "song_ce": float(loss_song.detach()), "frame_ce": float(loss_frame.detach()),
    }
    return total, parts


# --------------------------------------------------------------------------- #
# loops
# --------------------------------------------------------------------------- #
def train_one_epoch(model: BeatNetCRNN, loader, optimizer, device, bins, args,
                    logger: Logger, epoch: int, epoch_start: float) -> Dict[str, float]:
    model.train()
    meter, parts_meter = AverageMeter(), {k: AverageMeter() for k in
                                          ("reg", "log", "song_ce", "frame_ce")}
    n_batches = len(loader)
    step_t0 = time.time()
    last_log_step = 0

    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).squeeze(-1)

        optimizer.zero_grad(set_to_none=True)
        out = model(x, return_all=True)
        loss, parts = compute_loss(out, y, bins, args)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        meter.update(float(loss.detach()), x.size(0))
        for key, value in parts.items():
            parts_meter[key].update(value, x.size(0))

        if step % args.log_every == 0 or step == n_batches:
            elapsed = time.time() - epoch_start
            frac = step / max(n_batches, 1)
            eta = elapsed / max(frac, 1e-6) * (1.0 - frac)
            window = max(step - last_log_step, 1)
            rate = x.size(0) * window / max(time.time() - step_t0, 1e-6)
            logger(f"  epoch {epoch} | step {step:>4}/{n_batches} "
                   f"| loss {meter.avg:7.3f} "
                   f"| reg {parts_meter['reg'].avg:6.3f} "
                   f"| log {parts_meter['log'].avg:5.3f} "
                   f"| song_ce {parts_meter['song_ce'].avg:5.3f} "
                   f"| frame_ce {parts_meter['frame_ce'].avg:5.3f} "
                   f"| {rate:5.1f} samples/s "
                   f"| ETA {human_time(eta)}")
            step_t0 = time.time()
            last_log_step = step

    return {"loss": meter.avg, **{f"train_{k}": v.avg for k, v in parts_meter.items()}}


@torch.no_grad()
def evaluate(model: BeatNetCRNN, loader, device: torch.device, bins: torch.Tensor,
             args: argparse.Namespace, logger: Optional[Logger] = None,
             tag: str = "val") -> Dict[str, float]:
    """
    Full metric set on a split, for the three output paths (mean / frame / song).
    """
    model.eval()
    meter = AverageMeter()
    bucket: Dict[str, list] = {"mean": [], "frame": [], "song": []}
    references = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).squeeze(-1)
        out = model(x, return_all=True)
        loss, _ = compute_loss(out, y, bins, args)
        meter.update(float(loss.detach()), x.size(0))
        bucket["mean"].append(out["bpm"].detach().float().cpu().numpy())
        bucket["frame"].append(out["bpm_frame"].detach().float().cpu().numpy())
        bucket["song"].append(out["bpm_song"].detach().float().cpu().numpy())
        references.append(y.detach().float().cpu().numpy())

    reference = np.concatenate(references)
    metrics: Dict[str, float] = {"loss": meter.avg}
    for path, chunks in bucket.items():
        for key, value in tempo_metrics(np.concatenate(chunks), reference).items():
            metrics[f"{path}/{key}"] = value

    if logger is not None:
        logger(f"  {tag}: loss={meter.avg:.3f} "
               f"MAE={metrics['mean/MAE']:.2f} "
               f"Acc@1BPM={100 * metrics['mean/Acc_1BPM']:.1f}% "
               f"Acc@1%={100 * metrics['mean/Acc_1%']:.1f}% "
               f"Acc@5%={100 * metrics['mean/Acc_5%']:.1f}% "
               f"Acc_oct@4%={100 * metrics['mean/Acc_octave_4%']:.1f}% "
               f"oct_err={100 * metrics['mean/Octave_error_rate_4%']:.1f}% "
               f"P={metrics['mean/P_score']:.3f}")
        logger(f"    frame-only MAE={metrics['frame/MAE']:.2f} "
               f"acc1%={100 * metrics['frame/Acc_1%']:.1f}% | "
               f"song-only MAE={metrics['song/MAE']:.2f} "
               f"acc1%={100 * metrics['song/Acc_1%']:.1f}%")
    return metrics


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_results_markdown(summary: Dict[str, object], metrics: Dict[str, float],
                           args: argparse.Namespace) -> str:
    """Markdown report written to ``RESULTS.md`` when a run finishes."""
    keys = ["MAE", "RMSE", "MedianAE", "Acc_1BPM", "Acc_1%", "Acc_2%", "Acc_5%",
            "Acc_octave_4%", "Octave_error_rate_4%", "P_score", "Cemgil"]
    header = "| metric | mean output | frame only | song only |\n|---|---|---|---|"
    rows = []
    for key in keys:
        cells = []
        for path in ("mean", "frame", "song"):
            value = metrics.get(f"{path}/{key}")
            if value is None:
                cells.append("n/a")
            elif key.startswith(("Acc", "Octave_error")):
                cells.append(f"{100 * value:.1f}%")
            else:
                cells.append(f"{value:.3f}")
        rows.append(f"| {key} | " + " | ".join(cells) + " |")

    command = " ".join(["python -m src.train", f"--out {args.out}",
                        f"--epochs {args.epochs}", f"--synthetic {args.synthetic}",
                        f"--batch-size {args.batch_size}", f"--lr {args.lr}"])
    return "\n".join([
        "# BeatNet-CRNN - training results",
        "",
        f"* checkpoint: `{summary['checkpoint']}` (best epoch "
        f"{summary['best_epoch']} of {summary['epochs_requested']})",
        f"* test tracks: {summary['test_tracks']} (held-out, grouped split, no leakage)",
        f"* training time: {summary['total_human']}",
        f"* command: `{command}`",
        f"* finished at: {summary['finished_at']}",
        "",
        "## Test metrics (GTZAN + gtzan_tempo_beat reference annotations)",
        "",
        header,
        *rows,
        "",
        "`Acc_octave_4%` is the octave-tolerant accuracy (correct within 4 % up to a",
        "factor 1/2 or 2), `Octave_error_rate_4%` the share of predictions whose best",
        "factor is not 1 (half/double tempo confusion), `P_score` (McKinney) and",
        "`Cemgil` are the standard MIR tempo scores in log2-tempo space.",
        "",
        "## Files",
        "",
        "| file | content |",
        "|---|---|",
        "| `history.csv` | per-epoch train/val metrics |",
        "| `test_metrics.json` | full metric set (mean/frame/song) on the test split |",
        "| `predictions_test.csv` | per-track prediction vs reference (via `src.evaluate`) |",
        "| `assets/*.png` | scatter + error histogram + training curves (via `src.evaluate`) |",
        "| `best.pth` / `last.pth` | checkpoints (model + optimizer + history) |",
        "",
    ])


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train BeatNet-CRNN (BPM estimation)")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--out", default="runs/gtzan")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--gru-layers", type=int, default=2)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--n-classes", type=int, default=DEFAULT_N_CLASSES)
    parser.add_argument("--bpm-min", type=float, default=DEFAULT_BPM_MIN)
    parser.add_argument("--bpm-max", type=float, default=DEFAULT_BPM_MAX)
    parser.add_argument("--sigma-bins", type=float, default=3.0)
    parser.add_argument("--w-reg", type=float, default=1.0)
    parser.add_argument("--w-log", type=float, default=1.0)
    parser.add_argument("--w-song", type=float, default=1.0)
    parser.add_argument("--w-frame", type=float, default=0.5)
    parser.add_argument("--crop-seconds", type=float, default=CROP_SECONDS)
    parser.add_argument("--synthetic", type=int, default=500,
                        help="synthetic percussive loops added to the training set")
    parser.add_argument("--synthetic-bpm-min", type=float, default=40.0)
    parser.add_argument("--synthetic-bpm-max", type=float, default=220.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of real training tracks (smoke tests)")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--max-hours", type=float, default=0.0,
                        help="stop gracefully after N hours (0 = unlimited)")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir / "train.log")
    set_seed(args.seed)
    device = get_device(args.device)

    logger("=" * 78)
    logger(f"BeatNet-CRNN v{__version__} - training")
    logger(f"device      : {describe_device(device)}")
    logger(f"output dir  : {out_dir.resolve()}")
    logger(f"args        : epochs={args.epochs} batch={args.batch_size} lr={args.lr} "
           f"crop={args.crop_seconds}s synthetic={args.synthetic}")
    save_json(out_dir / "config.json", vars(args))

    bundle = build_dataloaders(
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        crop_seconds=args.crop_seconds,
        num_workers=args.num_workers,
        seed=args.seed,
        synthetic=args.synthetic,
        synthetic_bpm_range=(args.synthetic_bpm_min, args.synthetic_bpm_max),
        limit=args.limit,
        augment=not args.no_augment,
    )
    summary = bundle.summary()
    logger(f"data        : {summary} "
           f"(val={len(bundle.val_dataset)}, test={len(bundle.test_dataset)} tracks)")
    save_json(out_dir / "data_summary.json", summary)

    model = BeatNetCRNN(
        n_mels=args.n_mels, n_classes=args.n_classes,
        bpm_min=args.bpm_min, bpm_max=args.bpm_max,
        dropout=args.dropout, hidden_size=args.hidden_size,
        gru_layers=args.gru_layers, combine="mean",
    ).to(device)
    logger(f"model       : {model.n_params / 1e6:.2f} M parameters, "
           f"bins=[{args.bpm_min:.0f}, {args.bpm_max:.0f}] x{args.n_classes}")
    bins = model.bins

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    def lr_factor(epoch: int) -> float:
        if epoch < args.warmup_epochs:
            return (epoch + 1) / max(args.warmup_epochs, 1e-9)
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1e-9)
        progress = min(max(progress, 0.0), 1.0)
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    last_path = out_dir / "last.pth"
    best_path = out_dir / "best.pth"
    start_epoch, history, best = 0, [], {}
    if args.resume and last_path.exists():
        checkpoint = load_checkpoint(last_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch") or 0) + 1
        history = checkpoint.get("history") or []
        best = checkpoint.get("best") or {}
        for _ in range(start_epoch):
            scheduler.step()
        logger(f"resumed     : {last_path} -> epoch {start_epoch + 1}")

    history_path = out_dir / "history.csv"
    status_path = out_dir / "status.json"
    marker_path = out_dir / "TRAINING_COMPLETE"
    run_start = time.time()

    def write_history() -> None:
        if not history:
            return
        keys = list(history[0].keys())
        for row in history:
            for key in row:
                if key not in keys:
                    keys.append(key)
        with open(history_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for row in history:
                writer.writerow(row)

    stopped_early = False
    for epoch in range(start_epoch, args.epochs):
        set_epoch(bundle.train_dataset, epoch)
        epoch_start = time.time()
        lr_now = float(optimizer.param_groups[0]["lr"])
        logger("-" * 78)
        logger(f"epoch {epoch + 1}/{args.epochs} | lr {lr_now:.2e} | "
               f"elapsed {human_time(time.time() - run_start)}")

        train_stats = train_one_epoch(model, bundle.train_loader, optimizer, device,
                                      bins, args, logger, epoch + 1, epoch_start)
        val_metrics = evaluate(model, bundle.val_loader, device, bins, args,
                               logger, tag="val")
        scheduler.step()

        row = {
            "epoch": epoch + 1,
            "lr": lr_now,
            "epoch_seconds": round(time.time() - epoch_start, 2),
            "train_loss": train_stats["loss"],
            "train_reg": train_stats["train_reg"],
            "train_log": train_stats["train_log"],
            "train_song_ce": train_stats["train_song_ce"],
            "train_frame_ce": train_stats["train_frame_ce"],
            "val_loss": val_metrics["loss"],
            "val_MAE": val_metrics["mean/MAE"],
            "val_RMSE": val_metrics["mean/RMSE"],
            "val_Acc_1BPM": val_metrics["mean/Acc_1BPM"],
            "val_Acc_1%": val_metrics["mean/Acc_1%"],
            "val_Acc_5%": val_metrics["mean/Acc_5%"],
            "val_Acc_octave_4%": val_metrics["mean/Acc_octave_4%"],
            "val_Octave_error_4%": val_metrics["mean/Octave_error_rate_4%"],
            "val_P_score": val_metrics["mean/P_score"],
            "val_Cemgil": val_metrics["mean/Cemgil"],
        }
        history.append(row)
        write_history()

        improved = row["val_MAE"] < float(best.get("val_MAE", float("inf")))
        if improved:
            best = dict(row)
        logger(f"epoch {epoch + 1} done in {human_time(row['epoch_seconds'])} | "
               f"train_loss={row['train_loss']:.3f} | val_MAE={row['val_MAE']:.3f} "
               f"| best_val_MAE={float(best.get('val_MAE')):.3f}"
               f"{'  <-- new best' if improved else ''}")

        save_checkpoint(last_path, model, arch=model.arch_config(), config=vars(args),
                        optimizer=optimizer, epoch=epoch, best=best, history=history)
        if improved:
            save_checkpoint(best_path, model, arch=model.arch_config(), config=vars(args),
                            optimizer=optimizer, epoch=epoch, best=best, history=history)
            logger(f"  saved {best_path.name} (epoch {epoch + 1})")

        elapsed_hours = (time.time() - run_start) / 3600.0
        done_epochs = max(epoch + 1 - start_epoch, 1)
        eta = (time.time() - run_start) / done_epochs * (args.epochs - epoch - 1)
        save_json(status_path, {
            "state": "training",
            "epoch": epoch + 1,
            "epochs": args.epochs,
            "progress_pct": round(100.0 * (epoch + 1) / args.epochs, 2),
            "latest_val_MAE": row["val_MAE"],
            "latest_val_Acc_1%": row["val_Acc_1%"],
            "best_epoch": best.get("epoch"),
            "best_val_MAE": best.get("val_MAE"),
            "best_val_Acc_1%": best.get("val_Acc_1%"),
            "elapsed_hours": round(elapsed_hours, 3),
            "eta_seconds": int(eta),
            "eta_human": human_time(eta),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        logger(f"  progress {100.0 * (epoch + 1) / args.epochs:.1f}% | "
               f"elapsed {human_time(elapsed_hours * 3600)} | ETA {human_time(eta)}")

        if args.max_hours and elapsed_hours >= args.max_hours:
            logger(f"time budget ({args.max_hours} h) reached -> stopping gracefully")
            stopped_early = True
            break

    # ------------------------------------------------------------------ final
    logger("=" * 78)
    logger(f"training loop finished{' (early stop: time budget)' if stopped_early else ''}")
    final_ckpt = best_path if best_path.exists() else last_path
    checkpoint = load_checkpoint(final_ckpt, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    logger(f"evaluating {final_ckpt.name} on the held-out TEST split "
           f"({len(bundle.test_dataset)} tracks) ...")
    test_metrics = evaluate(model, bundle.test_loader, device, bins, args,
                            logger, tag="test")
    save_json(out_dir / "test_metrics.json", test_metrics)
    save_json(out_dir / f"test_metrics_{final_ckpt.stem}.json", test_metrics)
    summary_payload = {
        "state": "done",
        "epochs_requested": args.epochs,
        "epochs_completed": int(history[-1]["epoch"]) if history else 0,
        "stopped_early": bool(stopped_early),
        "best_epoch": best.get("epoch"),
        "best_val_MAE": best.get("val_MAE"),
        "best_val_Acc_1%": best.get("val_Acc_1%"),
        "test_MAE": test_metrics["mean/MAE"],
        "test_RMSE": test_metrics["mean/RMSE"],
        "test_Acc_1BPM": test_metrics["mean/Acc_1BPM"],
        "test_Acc_1%": test_metrics["mean/Acc_1%"],
        "test_Acc_5%": test_metrics["mean/Acc_5%"],
        "test_Acc_octave_4%": test_metrics["mean/Acc_octave_4%"],
        "test_Octave_error_4%": test_metrics["mean/Octave_error_rate_4%"],
        "test_P_score": test_metrics["mean/P_score"],
        "test_Cemgil": test_metrics["mean/Cemgil"],
        "total_seconds": round(time.time() - run_start, 1),
        "total_human": human_time(time.time() - run_start),
        "checkpoint": str(final_ckpt),
        "test_tracks": int(len(bundle.test_dataset)),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(status_path, summary_payload)
    with open(marker_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(summary_payload, indent=2))
    save_json(out_dir / "final_report.json", summary_payload)
    (out_dir / "RESULTS.md").write_text(
        build_results_markdown(summary_payload, test_metrics, args), encoding="utf-8")
    logger(f"markdown report -> {out_dir / 'RESULTS.md'}")

    logger("")
    logger("FINAL METRICS (test split, reference annotations)")
    logger("-" * 78)
    for line in format_metrics({k.split("/", 1)[1]: v
                                for k, v in test_metrics.items() if k.startswith("mean/")},
                               prefix="  ").splitlines():
        logger(line)
    logger("-" * 78)
    logger(f"  best epoch {summary_payload['best_epoch']} "
           f"| test MAE {summary_payload['test_MAE']:.3f} BPM "
           f"| Acc@1% {100 * summary_payload['test_Acc_1%']:.1f}% "
           f"| total time {summary_payload['total_human']}")
    logger(f"DONE - completion marker: {marker_path}")
    logger(f"checkpoint: {final_ckpt} | per-epoch table: {history_path}")
    logger.close()


if __name__ == "__main__":
    main()

