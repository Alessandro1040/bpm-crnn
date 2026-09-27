#!/usr/bin/env python3
"""
Export an *inference-only* checkpoint: model weights + architecture + Mel config,
without the optimiser state (21 MB instead of 64 MB) so it can live in the repo
and be loaded straight from GitHub in the Colab notebooks.

Usage:
    python scripts/export_inference.py --ckpt runs/gtzan/best.pth \
        --out models/bpm_crnn_gtzan.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import build_model_from_config          # noqa: E402
from src.utils import load_checkpoint                 # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default="runs/gtzan/best.pth")
    parser.add_argument("--out", default="models/bpm_crnn_gtzan.pt")
    parser.add_argument("--metrics", default="runs/gtzan/test_metrics.json")
    parser.add_argument("--note", default="trained on GTZAN (60 epochs, best epoch 54)")
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.ckpt, map_location="cpu")
    arch = checkpoint.get("arch") or {}
    config = dict(checkpoint.get("config") or {})

    # make inference self-describing: the notebook has no data/ directory
    config.setdefault("sr", 22050)
    config.setdefault("n_fft", 2048)
    config.setdefault("hop_length", 512)
    config.setdefault("fmax", 8000.0)
    config.setdefault("n_mels", arch.get("n_mels", 128))
    config.setdefault("crop_seconds", 10.0)

    payload = {
        "state_dict": checkpoint["state_dict"],
        "arch": arch,
        "config": config,
        "epoch": checkpoint.get("epoch"),
        "best": checkpoint.get("best") or {},
        "note": args.note,
        "source_checkpoint": str(args.ckpt),
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    metrics_path = Path(args.metrics)
    if metrics_path.exists():
        payload["test_metrics"] = json.loads(metrics_path.read_text())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"[export] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")

    # sanity check: rebuild the model from the exported file and run a forward pass
    model = build_model_from_config(arch)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    with torch.no_grad():
        bpm, logits = model(torch.zeros(1, 1, config["n_mels"], 431))
    print(f"[export] reload + forward pass OK | silence -> {float(bpm):.2f} BPM "
          f"| frame logits {tuple(logits.shape)}")
    if "test_metrics" in payload:
        m = payload["test_metrics"]
        print(f"[export] embedded metrics: MAE {m['mean/MAE']:.2f} BPM, "
              f"Acc@5% {100 * m['mean/Acc_5%']:.1f}%, "
              f"Acc octave@4% {100 * m['mean/Acc_octave_4%']:.1f}%")


if __name__ == "__main__":
    main()
