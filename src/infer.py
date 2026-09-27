"""
BPM inference: ``BPMExtractor`` (reference-compatible API) + CLI.

Long files are analysed with overlapping 10 s windows; the final estimate is the
*median* over windows (robust to tempo changes). Optionally test-time
augmentation (pitch shift +-1 semitone) is averaged before the median.

Usage
-----
  python -m src.infer --ckpt runs/gtzan/best.pth --audio song.mp3 --json out.json

  from src.infer import BPMExtractor
  extractor = BPMExtractor("runs/gtzan/best.pth")
  print(extractor.predict("song.mp3"))
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import librosa
import numpy as np
import torch

from .dataset import (FMAX, HOP_LENGTH, N_FFT, N_MELS, SR, log_mel)
from .model import build_model_from_config
from .utils import get_device, human_time, load_checkpoint


class BPMExtractor:
    """
    Load a trained BeatNet-CRNN checkpoint and estimate the BPM of an audio file.

    Args:
        model_path: checkpoint written by ``src.train`` (``best.pth`` / ``last.pth``).
        device: 'auto' | 'cpu' | 'mps' | 'cuda'.
        cache_dir: Mel cache directory; its ``cache_config.json`` is used to
            recover the exact spectrogram settings (sr / n_fft / hop / fmax).
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "auto",
        cache_dir: Optional[Union[str, Path]] = "data/cache",
        sr: Optional[int] = None,
        n_mels: Optional[int] = None,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        fmax: Optional[float] = None,
    ) -> None:
        self.device = get_device(device)
        checkpoint = load_checkpoint(model_path, map_location=self.device)
        arch = checkpoint.get("arch") or {}
        self.model = build_model_from_config(arch).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.checkpoint_meta = {
            "path": str(model_path),
            "epoch": checkpoint.get("epoch"),
            "best": checkpoint.get("best") or {},
            "train_config": checkpoint.get("config") or {},
        }

        # spectrogram settings: explicit args > Mel cache config > defaults
        cache_config: Dict[str, float] = {}
        if cache_dir is not None:
            config_path = Path(cache_dir) / "cache_config.json"
            if config_path.exists():
                cache_config = json.loads(config_path.read_text())
        self.sr = int(sr or cache_config.get("sr", SR))
        self.n_mels = int(n_mels or arch.get("n_mels") or cache_config.get("n_mels", N_MELS))
        self.n_fft = int(n_fft or cache_config.get("n_fft", N_FFT))
        self.hop_length = int(hop_length or cache_config.get("hop_length", HOP_LENGTH))
        self.fmax = float(fmax or cache_config.get("fmax", FMAX))

    # ------------------------------------------------------------------ helpers
    def _mel_tensor(self, y: np.ndarray) -> torch.Tensor:
        mel = log_mel(y, sr=self.sr, n_mels=self.n_mels, n_fft=self.n_fft,
                      hop_length=self.hop_length, fmax=self.fmax)
        mel = (mel - float(mel.mean())) / float(mel.std() + 1e-8)
        return torch.from_numpy(np.ascontiguousarray(mel.astype(np.float32))[None, None])

    def _chunks(self, y: np.ndarray, chunk_duration: float, overlap: float
                ) -> List[np.ndarray]:
        chunk = int(chunk_duration * self.sr)
        if chunk <= 0:
            raise ValueError("chunk_duration must be > 0")
        if len(y) <= chunk:
            return [y]
        hop = max(1, int((chunk_duration - max(0.0, overlap)) * self.sr))
        chunks = [y[i:i + chunk] for i in range(0, len(y) - chunk + 1, hop)]
        if len(y) - (len(chunks) - 1) * hop - chunk > self.sr // 2:
            chunks.append(y[-chunk:])                     # make sure the tail is covered
        return chunks

    @torch.no_grad()
    def _predict_mel(self, x: torch.Tensor) -> Dict[str, float]:
        out = self.model(x.to(self.device), return_all=True)
        return {
            "bpm": float(out["bpm"].item()),
            "bpm_frame": float(out["bpm_frame"].item()),
            "bpm_song": float(out["bpm_song"].item()),
        }

    # ---------------------------------------------------------------- public API
    def predict(
        self,
        audio_path: Union[str, Path],
        chunk_duration: float = 10.0,
        overlap: float = 5.0,
        tta: bool = True,
        shifts: Sequence[float] = (0.0, 1.0, -1.0),
        return_details: bool = False,
    ):
        """
        Estimate the BPM of an audio file.

        Args:
            audio_path: any format supported by librosa/audioread.
            chunk_duration: analysis window (10 s is the training setting).
            overlap: window overlap in seconds (5 s by default).
            tta: average the predictions of pitch-shifted copies (+-1 semitone).
            return_details: also return the per-window estimates and statistics.

        Returns:
            float BPM (or a dict when ``return_details=True``).
        """
        started = time.time()
        y, _ = librosa.load(str(audio_path), sr=self.sr, mono=True)
        if y.size == 0:
            raise ValueError(f"empty audio: {audio_path}")

        chunks = self._chunks(y, chunk_duration, overlap)
        shift_list = list(shifts) if tta else [0.0]
        chunk_bpms: List[float] = []
        for chunk in chunks:
            values = []
            for shift in shift_list:
                signal = chunk if shift == 0 else librosa.effects.pitch_shift(
                    y=chunk, sr=self.sr, n_steps=float(shift))
                values.append(self._predict_mel(self._mel_tensor(signal))["bpm"])
            chunk_bpms.append(float(np.mean(values)))

        windows = np.asarray(chunk_bpms, dtype=np.float64)
        bpm = float(np.median(windows))              # robust to tempo changes

        if not return_details:
            return bpm
        return {
            "path": str(audio_path),
            "bpm": bpm,
            "bpm_mean": float(windows.mean()),
            "bpm_median": bpm,
            "bpm_std": float(windows.std()),
            "bpm_min": float(windows.min()),
            "bpm_max": float(windows.max()),
            "n_chunks": int(windows.size),
            "chunk_bpms": [float(v) for v in windows],
            "chunk_duration": float(chunk_duration),
            "overlap": float(overlap),
            "tta": bool(tta),
            "sr": self.sr,
            "audio_seconds": float(len(y) / self.sr),
            "elapsed_seconds": round(time.time() - started, 3),
        }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="BPM estimation with BeatNet-CRNN")
    parser.add_argument("--ckpt", required=True, help="trained checkpoint (.pth)")
    parser.add_argument("--audio", nargs="+", required=True, help="audio file(s)")
    parser.add_argument("--chunk-duration", type=float, default=10.0)
    parser.add_argument("--overlap", type=float, default=5.0)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--json", default=None, help="dump the detailed results here")
    args = parser.parse_args()

    extractor = BPMExtractor(args.ckpt, device=args.device, cache_dir=args.cache_dir)
    print(f"[infer] checkpoint: {args.ckpt} | device: {extractor.device.type} "
          f"| sr={extractor.sr} | n_mels={extractor.n_mels}")

    results = []
    total_time = 0.0
    for path in args.audio:
        details = extractor.predict(path, chunk_duration=args.chunk_duration,
                                    overlap=args.overlap, tta=not args.no_tta,
                                    return_details=True)
        results.append(details)
        total_time += details["elapsed_seconds"]
        print(f"  {Path(path).name:<45} {details['bpm']:7.2f} BPM "
              f"| windows={details['n_chunks']:>3} "
              f"| spread={details['bpm_std']:5.2f} "
              f"| audio={details['audio_seconds']:7.1f}s "
              f"| {details['elapsed_seconds']:6.2f}s")

    print(f"[infer] {len(results)} file(s) in {human_time(total_time)}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"[infer] details -> {args.json}")


if __name__ == "__main__":
    main()
