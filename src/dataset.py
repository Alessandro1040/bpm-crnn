"""
Data pipeline for BPM (tempo) estimation.

Supports:
  * GTZAN (audio: ``data/genres/<genre>/<name>.wav``) + tempo/beat annotations
    from ``TempoBeatDownbeat/gtzan_tempo_beat`` (``gtzan_<genre>_<idx>.bpm``)
  * any custom dataset described by a CSV with ``filename,bpm`` columns
    (drop-in replacement for the reference ``AudioDataset``)
  * a synthetic rhythm generator with *exact* BPM labels (continuous tempo
    coverage + robustness), used as an extra data source during training

Design notes
------------
Mel spectrograms are pre-computed once and stored as a float16 ``.npy``
memmap (``data/cache/mels.npy``); training then only does crops +
augmentation in the Mel domain, which is what makes epochs fast on Apple MPS.

Augmentation in the Mel domain keeps the BPM label consistent:
  * time stretch  -> linear interpolation along time AND ``bpm *= rate``
  * pitch shift   -> roll along the frequency axis (BPM unchanged)
  * SpecAugment   -> random frequency/time masks
  * gain / noise  -> additive shifts in the log-Mel domain

CLI
---
  python -m src.dataset --data-root data --cache-dir data/cache --build
  python -m src.dataset --data-root data --cache-dir data/cache --split-only
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

# --------------------------------------------------------------------------- #
# Configuration defaults (identical to the reference implementation)
# --------------------------------------------------------------------------- #
SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMAX = 8000.0
DURATION = 30.0                      # GTZAN clips are 30 s long
CROP_SECONDS = 10.0                  # training/eval crop length
GTZAN_GENRES = ["blues", "classical", "country", "disco", "hiphop",
                "jazz", "metal", "pop", "reggae", "rock"]
BINS_PER_SEMITONE = 4                # approx. resolution of 128 Mels (0-8 kHz)


def n_frames(seconds: float, sr: int = SR, hop_length: int = HOP_LENGTH) -> int:
    return int(np.floor(seconds * sr / hop_length)) + 1


def log_mel(
    y: np.ndarray,
    sr: int = SR,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    fmax: float = FMAX,
) -> np.ndarray:
    """Log (dB) Mel spectrogram, exactly as used for training/inference."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, fmax=fmax, power=2.0,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)




# --------------------------------------------------------------------------- #
# Metadata / caching / splits
# --------------------------------------------------------------------------- #
AUDIO_EXTENSIONS = (".wav", ".mp3", ".ogg", ".flac", ".m4a")


def file_sha1(path: Path, chunk_size: int = 1 << 20) -> str:
    """Content hash, used to detect GTZAN duplicates (5 groups of 100 tracks)."""
    h = hashlib.sha1()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def build_metadata(
    data_root: str | Path,
    audio_root: Optional[str | Path] = None,
    annot_dir: Optional[str | Path] = None,
    out_csv: Optional[str | Path] = None,
    sr: int = SR,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Scan the audio tree, attach the reference BPM annotation and hash the audio.

    Annotation file name convention (gtzan_tempo_beat repo):
    ``gtzan_<genre>_<index>.bpm`` for audio ``<genre>/<genre>.<index>.wav``.
    """
    data_root = Path(data_root)
    audio_root = Path(audio_root) if audio_root else data_root / "genres"
    annot_dir = Path(annot_dir) if annot_dir else data_root / "gtzan_tempo_beat" / "tempo"

    files: List[Path] = []
    for ext in AUDIO_EXTENSIONS:
        files.extend(sorted(audio_root.glob(f"*/*{ext}")))
    if not files:
        raise FileNotFoundError(f"no audio files found under {audio_root}")

    rows: List[Dict[str, object]] = []
    missing = 0
    for path in files:
        genre = path.parent.name
        stem = path.stem
        index = stem.split(".")[-1]
        candidates = [annot_dir / f"gtzan_{genre}_{index}.bpm",
                      annot_dir / f"{stem}.bpm"]
        annot = next((c for c in candidates if c.exists()), None)
        if annot is None:
            missing += 1
            continue
        with open(annot, "r") as handle:
            bpm = float(handle.read().split()[0])
        if bpm <= 0 or not np.isfinite(bpm):
            missing += 1
            continue
        try:
            info = sf.info(str(path))
        except Exception:                     # corrupted / unreadable audio
            missing += 1
            continue
        rows.append({
            "track_id": f"{genre}/{stem}",
            "path": str(path),
            "genre": genre,
            "bpm": bpm,
            "sha1": file_sha1(path),
            "duration": float(info.duration),
            "sample_rate": int(info.samplerate),
        })

    meta = pd.DataFrame(rows).sort_values("track_id").reset_index(drop=True)
    if meta.empty:
        raise RuntimeError("no track matched an annotation file")

    if verbose:
        print(f"[metadata] {len(meta)} tracks with BPM annotation "
              f"({missing} skipped), {meta['sha1'].nunique()} unique audio files")
        print(f"[metadata] tempo range {meta['bpm'].min():.2f} - {meta['bpm'].max():.2f} BPM "
              f"(mean {meta['bpm'].mean():.2f})")
        dupes = meta["sha1"].value_counts()
        if (dupes > 1).any():
            print(f"[metadata] duplicate audio groups: {(dupes > 1).sum()} "
                  f"(kept together during the split)")

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        meta.to_csv(out_csv, index=False)
    return meta


def make_splits(
    meta: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Grouped (by audio hash) and genre-stratified train/val/test split.

    GTZAN contains duplicate tracks across genres: splitting by hash guarantees
    that no audio content is shared between splits (no leakage).
    """
    rng = np.random.default_rng(seed)
    meta = meta.copy()
    meta["split"] = "train"

    # group the *whole* dataset by audio hash first (duplicates in GTZAN cross
    # genre boundaries), then stratify the groups by genre.
    strata: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    for content_hash, positions in meta.groupby("sha1").indices.items():
        genre = str(meta["genre"].iloc[positions[0]])
        strata.setdefault(genre, []).append((content_hash, np.asarray(positions)))

    for _, groups in sorted(strata.items()):
        order = rng.permutation(len(groups))
        groups = [groups[i] for i in order]
        n_val = max(1, int(round(len(groups) * val_frac)))
        n_test = max(1, int(round(len(groups) * test_frac)))
        for _, positions in groups[:n_val]:
            meta.loc[meta.index[positions], "split"] = "val"
        for _, positions in groups[n_val:n_val + n_test]:
            meta.loc[meta.index[positions], "split"] = "test"

    if verbose:
        counts = meta["split"].value_counts().to_dict()
        print(f"[split] {counts} (groups by audio hash: "
              f"{meta.groupby('split')['sha1'].nunique().to_dict()})")

    # leakage check: the same audio must never appear in two different splits
    shared = meta.groupby("sha1")["split"].nunique()
    if (shared > 1).any():
        raise RuntimeError("leakage detected: same audio in multiple splits")
    return meta


def cache_mels(
    meta: pd.DataFrame,
    cache_dir: str | Path,
    overwrite: bool = False,
    sr: int = SR,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    fmax: float = FMAX,
    duration: float = DURATION,
    verbose: bool = True,
) -> Dict[str, Path]:
    """
    Pre-compute log-Mel spectrograms into a float16 memmap (one row per track).

    Returns a dict with the ``mels`` / ``index`` / ``config`` paths.
    """
    from tqdm import tqdm

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    mels_path = cache_dir / "mels.npy"
    index_path = cache_dir / "index.csv"
    config_path = cache_dir / "cache_config.json"

    config = {
        "sr": sr, "n_mels": n_mels, "n_fft": n_fft, "hop_length": hop_length,
        "fmax": fmax, "duration": duration, "n_tracks": int(len(meta)),
    }
    if mels_path.exists() and index_path.exists() and not overwrite:
        if config_path.exists() and json.load(open(config_path)) == config:
            if verbose:
                print(f"[cache] reusing {mels_path} ({len(meta)} tracks)")
            return {"mels": mels_path, "index": index_path, "config": config_path}

    total_frames = n_frames(duration, sr, hop_length)
    target_len = int(round(duration * sr))
    array = np.lib.format.open_memmap(
        mels_path, mode="w+", dtype=np.float16,
        shape=(len(meta), n_mels, total_frames),
    )

    means, stds, valid_frames = [], [], []
    skipped = 0
    for i, row in enumerate(tqdm(meta.itertuples(), total=len(meta),
                                 desc="[cache] log-Mel", disable=not verbose)):
        try:
            y, _ = librosa.load(row.path, sr=sr, mono=True, duration=duration)
        except Exception:                     # corrupted file: zero row, dropped later
            array[i] = 0.0
            means.append(0.0)
            stds.append(1.0)
            valid_frames.append(0)
            skipped += 1
            continue
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]
        mel = log_mel(y, sr=sr, n_mels=n_mels, n_fft=n_fft,
                      hop_length=hop_length, fmax=fmax)
        frames = mel.shape[1]
        if frames < total_frames:
            mel = np.pad(mel, ((0, 0), (0, total_frames - frames)),
                         constant_values=float(mel.min()))
        else:
            mel = mel[:, :total_frames]
        array[i] = mel.astype(np.float16)
        means.append(float(mel.mean()))
        stds.append(float(mel.std() + 1e-8))
        valid_frames.append(int(min(frames, total_frames)))

    array.flush()
    del array

    index = meta.copy()
    index["mel_mean"] = means
    index["mel_std"] = stds
    index["n_frames"] = valid_frames
    index.to_csv(index_path, index=False)
    with open(config_path, "w") as handle:
        json.dump(config, handle, indent=2)

    size_mb = mels_path.stat().st_size / 1e6
    if verbose:
        print(f"[cache] wrote {mels_path} ({size_mb:.1f} MB, "
              f"{len(meta)} x {n_mels} x {total_frames})")
        if skipped:
            print(f"[cache] {skipped} track(s) unreadable -> excluded from the datasets")
    return {"mels": mels_path, "index": index_path, "config": config_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="GTZAN + tempo-annotation data preparation")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--meta", default="data/meta.csv")
    parser.add_argument("--build", action="store_true",
                        help="build metadata + splits + Mel cache")
    parser.add_argument("--split-only", action="store_true",
                        help="recompute the train/val/test split on existing metadata")
    parser.add_argument("--overwrite", action="store_true", help="recompute the Mel cache")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    meta_path = Path(args.meta)
    if args.build:
        meta = build_metadata(args.data_root, out_csv=meta_path)
        meta = make_splits(meta, val_frac=args.val_frac,
                           test_frac=args.test_frac, seed=args.seed)
        meta.to_csv(meta_path, index=False)
        cache_mels(meta, args.cache_dir, overwrite=args.overwrite)
    elif args.split_only:
        meta = pd.read_csv(meta_path)
        meta = make_splits(meta, val_frac=args.val_frac,
                           test_frac=args.test_frac, seed=args.seed)
        meta.to_csv(meta_path, index=False)
        index_path = Path(args.cache_dir) / "index.csv"
        if index_path.exists():
            index = pd.read_csv(index_path)
            index = index.drop(columns=["split"], errors="ignore").merge(
                meta[["track_id", "split"]], on="track_id", how="left")
            index.to_csv(index_path, index=False)
            print(f"[split] updated {index_path}")
    else:
        parser.print_help()


# --------------------------------------------------------------------------- #
# Mel-domain augmentation (label-consistent)
# --------------------------------------------------------------------------- #
def time_stretch_mel(mel: np.ndarray, rate: float) -> np.ndarray:
    """
    Resample along the time axis. ``rate > 1`` = faster playback, i.e. the
    tempo (and therefore the BPM label) is multiplied by ``rate``.
    """
    if abs(rate - 1.0) < 1e-3:
        return mel
    total = mel.shape[1]
    new_total = max(8, int(round(total / rate)))
    idx = np.linspace(0.0, total - 1.0, new_total)
    lo = np.floor(idx).astype(np.int64)
    hi = np.minimum(lo + 1, total - 1)
    w = (idx - lo).astype(np.float32)
    return (mel[:, lo] * (1.0 - w) + mel[:, hi] * w).astype(np.float32)


def pitch_shift_mel(mel: np.ndarray, semitones: float,
                    bins_per_semitone: int = BINS_PER_SEMITONE) -> np.ndarray:
    """Approximate pitch shift by rolling the frequency axis (BPM unchanged)."""
    shift = int(round(float(semitones) * bins_per_semitone))
    if shift == 0:
        return mel
    out = np.zeros_like(mel)
    n_bins = mel.shape[0]
    if shift > 0:
        out[min(shift, n_bins):, :] = mel[:max(0, n_bins - shift), :]
    else:
        out[:max(0, n_bins + shift), :] = mel[min(-shift, n_bins):, :]
    return out


def spec_augment_mel(
    mel: np.ndarray,
    rng: np.random.Generator,
    n_freq_masks: int = 2,
    n_time_masks: int = 2,
    max_freq: int = 12,
    max_time: int = 25,
) -> np.ndarray:
    """SpecAugment masking on the log-Mel patch."""
    mel = mel.copy()
    n_bins, total = mel.shape
    fill = float(mel.min())
    for _ in range(n_freq_masks):
        width = int(rng.integers(0, max_freq + 1))
        if width > 0 and n_bins - width > 0:
            start = int(rng.integers(0, n_bins - width))
            mel[start:start + width, :] = fill
    for _ in range(n_time_masks):
        width = int(rng.integers(0, max_time + 1))
        if width > 0 and total - width > 0:
            start = int(rng.integers(0, total - width))
            mel[:, start:start + width] = fill
    return mel


def set_epoch(dataset: Dataset, epoch: int) -> None:
    """Propagate the epoch counter so that (synthetic) datasets reshuffle."""
    if isinstance(dataset, ConcatDataset):
        for sub in dataset.datasets:
            set_epoch(sub, epoch)
    elif hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)


def fix_length(mel: np.ndarray, frames: int) -> np.ndarray:
    """
    Crop/pad along time so that every sample has exactly ``frames`` columns.

    Needed because time-stretch augmentation changes the number of frames while
    a DataLoader batch requires a rectangular tensor.
    """
    total = mel.shape[1]
    if total == frames:
        return mel
    if total > frames:
        start = (total - frames) // 2
        return mel[:, start:start + frames]
    pad = frames - total
    return np.pad(mel, ((0, 0), (0, pad)), mode="edge").astype(mel.dtype)


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
class GTZANDataset(Dataset):
    """
    Random-crop dataset over the pre-computed log-Mel cache.

    Args:
        cache_dir: directory produced by ``python -m src.dataset --build``.
        split: ``train`` / ``val`` / ``test`` (from the grouped split).
        crop_seconds: analysed excerpt length (10 s, the usual MIR setting).
        augment: enable the Mel-domain augmentation (train only).
        norm: ``crop`` (per-crop standardisation, as in the reference code) or
            ``track`` (statistics of the whole 30 s track).
        limit: keep only the first N tracks (smoke tests).
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/cache",
        split: str = "train",
        crop_seconds: float = CROP_SECONDS,
        augment: bool = False,
        norm: str = "crop",
        seed: int = 42,
        limit: Optional[int] = None,
        stretch_prob: float = 0.5,
        max_stretch: float = 0.05,
        pitch_prob: float = 0.5,
        max_semitones: float = 2.0,
        specaug_prob: float = 0.5,
        specaug: Tuple[int, int, int, int] = (2, 2, 12, 25),
        gain_prob: float = 0.5,
        max_gain_db: float = 6.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        index_path = self.cache_dir / "index.csv"
        config_path = self.cache_dir / "cache_config.json"
        if not (index_path.exists() and config_path.exists()):
            raise FileNotFoundError(
                f"Mel cache not found in {self.cache_dir}. Run: "
                "python -m src.dataset --data-root data --cache-dir data/cache --build"
            )
        index = pd.read_csv(index_path)
        if "split" not in index.columns:
            raise ValueError("index.csv has no 'split' column: rerun --split-only")
        if split not in set(index["split"]):
            raise ValueError(f"unknown split {split!r}")
        self.index = (
            index[index["split"] == split]
            .reset_index()
            .rename(columns={"index": "cache_row"})
        )
        if "n_frames" in self.index.columns:          # drop unreadable/corrupted audio
            self.index = self.index[self.index["n_frames"] > 0]
        self.index = self.index.reset_index(drop=True)
        if limit:
            self.index = self.index.iloc[:limit].reset_index(drop=True)

        config = json.load(open(config_path))
        self.config = config
        self.sr = int(config["sr"])
        self.n_mels = int(config["n_mels"])
        self.hop_length = int(config["hop_length"])
        self.total_frames = int(round(config["duration"] * self.sr / self.hop_length)) + 1
        self.crop_frames = (
            int(np.floor(crop_seconds * self.sr / self.hop_length)) + 1
            if crop_seconds else None
        )
        self.mels = np.load(self.cache_dir / "mels.npy", mmap_mode="r")

        self.split = split
        self.augment = bool(augment)
        self.norm = norm
        self.base_seed = int(seed)
        self._epoch = 0
        self._rng: Optional[np.random.Generator] = None
        self.stretch_prob, self.max_stretch = stretch_prob, max_stretch
        self.pitch_prob, self.max_semitones = pitch_prob, max_semitones
        self.specaug_prob, self.specaug = specaug_prob, specaug
        self.gain_prob, self.max_gain_db = gain_prob, max_gain_db

    # ------------------------------------------------------------------ utils
    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        self._rng = None

    def _get_rng(self) -> np.random.Generator:
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            worker = 0 if info is None else int(info.id) * 100003 + 17
            self._rng = np.random.default_rng(self.base_seed + worker + 7919 * self._epoch)
        return self._rng

    def tracks(self) -> pd.DataFrame:
        """Reference frame (track_id, genre, bpm) of the selected split."""
        return self.index[["track_id", "genre", "bpm"]].copy()

    def __len__(self) -> int:
        return len(self.index)

    # ---------------------------------------------------------------- getitem
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.index.iloc[idx]
        mel = np.asarray(self.mels[int(row["cache_row"])]).astype(np.float32)
        bpm = float(row["bpm"])
        total = mel.shape[1]
        rng = self._get_rng()

        if self.crop_frames is not None and total > self.crop_frames:
            if self.augment:
                start = int(rng.integers(0, total - self.crop_frames + 1))
            else:
                start = (total - self.crop_frames) // 2       # deterministic for eval
            mel = mel[:, start:start + self.crop_frames]

        if self.augment:
            if rng.random() < self.pitch_prob:
                mel = pitch_shift_mel(mel, rng.uniform(-self.max_semitones,
                                                       self.max_semitones))
            if rng.random() < self.stretch_prob:
                rate = float(rng.uniform(1.0 - self.max_stretch, 1.0 + self.max_stretch))
                mel = time_stretch_mel(mel, rate)
                bpm *= rate                                  # label follows the stretch
            if rng.random() < self.gain_prob:
                mel = mel + float(rng.uniform(-self.max_gain_db, self.max_gain_db))
            if rng.random() < self.specaug_prob:
                mel = spec_augment_mel(mel, rng, self.specaug[0], self.specaug[1],
                                       max_freq=self.specaug[2], max_time=self.specaug[3])

        if self.norm == "track":
            mean, std = float(row["mel_mean"]), float(row["mel_std"])
        else:
            mean, std = float(mel.mean()), float(mel.std() + 1e-8)
        mel = (mel - mean) / std

        if self.crop_frames is not None:
            mel = fix_length(mel, self.crop_frames)      # rectangular batches

        # torch.tensor (not from_numpy) gives the tensor its own storage: the
        # Mel cache is a read-only memmap and non-resizable storages break the
        # DataLoader collate step.
        x = torch.tensor(np.ascontiguousarray(mel, dtype=np.float32))[None, :, :]
        return x, torch.tensor([bpm], dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Synthetic rhythm dataset (exact BPM labels, no download needed)
# --------------------------------------------------------------------------- #
def _moving_average(x: np.ndarray, taps: int) -> np.ndarray:
    taps = max(1, int(taps))
    if taps == 1:
        return x
    kernel = np.ones(taps, dtype=np.float64) / taps
    return np.convolve(x, kernel, mode="same")


def _band_noise(rng: np.random.Generator, n: int, sr: int,
                low_hz: float, high_hz: float) -> np.ndarray:
    """Cheap band-limited noise (difference of two moving averages)."""
    noise = rng.standard_normal(n)
    lp_high = _moving_average(noise, int(sr / max(high_hz, 1.0)))
    lp_low = _moving_average(noise, int(sr / max(low_hz, 1.0)))
    band = lp_high - lp_low
    return band / (np.max(np.abs(band)) + 1e-9)


def _drum_hit(kind: str, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Short percussive one-shot (kick / snare / hi-hat)."""
    if kind == "kick":
        length = int(0.25 * sr)
        t = np.arange(length) / sr
        freq = rng.uniform(45.0, 62.0)
        body = np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.09)
        click = _band_noise(rng, length, sr, 1000.0, 6000.0) * np.exp(-t / 0.004) * 0.35
        hit = body + click
    elif kind == "snare":
        length = int(0.20 * sr)
        t = np.arange(length) / sr
        noise = _band_noise(rng, length, sr, 200.0, 8000.0) * np.exp(-t / 0.06)
        tone = 0.3 * np.sin(2 * np.pi * 180.0 * t) * np.exp(-t / 0.05)
        hit = noise + tone
    else:  # hi-hat
        length = int(0.08 * sr)
        t = np.arange(length) / sr
        hit = _band_noise(rng, length, sr, 4000.0, 10000.0) * np.exp(-t / 0.02)
    return hit / (np.max(np.abs(hit)) + 1e-9)


def build_rhythm_onsets(
    bpm: float,
    duration: float = CROP_SECONDS,
    rng: Optional[np.random.Generator] = None,
    meter: Optional[int] = None,
    subdiv: Optional[int] = None,
    swing: Optional[bool] = None,
) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    """
    Build the onset grid (in seconds) of a random percussive pattern at ``bpm``.

    The grid is what makes the synthetic BPM label exact: kick/snare onsets fall
    on the beat grid (``k * 60 / bpm``), apart from small timing jitter.

    Returns:
        (onsets, meta) with ``onsets`` = {'kick': [...], 'snare': [...], 'hat': [...]}
        and ``meta`` = beat period / meter / subdivision / swing.
    """
    rng = rng or np.random.default_rng()
    beat = 60.0 / bpm
    meter = int(meter or rng.choice([4, 4, 4, 3]))
    subdiv = int(subdiv or rng.choice([1, 2, 2, 4, 4]))
    swing = bool(rng.random() < 0.25 and subdiv == 2) if swing is None else bool(swing)

    onsets: Dict[str, List[float]] = {"kick": [], "snare": [], "hat": []}
    step_dur = beat / subdiv
    for step in range(int(np.ceil(duration / step_dur)) + 2):
        offset = step * step_dur
        if swing and step % 2 == 1:
            offset = (step - 1) * step_dur + step_dur * (4.0 / 3.0)
        if offset >= duration:
            continue
        beat_idx = int(step // subdiv) % meter
        in_beat = step % subdiv
        jitter = float(rng.normal(0.0, 0.006))

        if in_beat == 0 and beat_idx == 0:
            onsets["kick"].append(offset + jitter)
        elif in_beat == 0 and meter == 4 and beat_idx == 2 and rng.random() < 0.5:
            onsets["kick"].append(offset + jitter)
        elif rng.random() < 0.10:
            onsets["kick"].append(offset + jitter)

        strong = (meter == 4 and beat_idx in (1, 3)) or (meter == 3 and beat_idx == 2)
        if in_beat == 0 and strong:
            onsets["snare"].append(offset + jitter)
        elif rng.random() < 0.06:
            onsets["snare"].append(offset + jitter)

        if rng.random() < 0.85:
            onsets["hat"].append(offset + jitter)

    meta = {"beat": beat, "meter": float(meter), "subdiv": float(subdiv),
            "swing": float(swing), "step_duration": step_dur,
            "bpm": float(bpm)}
    return onsets, meta


def synthesize_rhythm(
    bpm: float,
    sr: int = SR,
    duration: float = CROP_SECONDS,
    rng: Optional[np.random.Generator] = None,
    noise_snr_db: Optional[float] = None,
) -> np.ndarray:
    """
    Render a percussive loop at an *exact* tempo.

    The pattern vocabulary (meter, subdivision, swing, syncopation) is random,
    so the network cannot memorise a single rhythm, while the BPM label is
    exact by construction.
    """
    rng = rng or np.random.default_rng()
    n_samples = int(duration * sr)
    onsets, _ = build_rhythm_onsets(bpm, duration, rng)
    y = np.zeros(n_samples, dtype=np.float64)
    for kind, times in onsets.items():
        base = _drum_hit(kind, sr, rng)
        gain = {"kick": 1.0, "snare": 0.8, "hat": 0.25}[kind]
        for onset in times:
            start = int(onset * sr)
            if start < 0 or start >= n_samples:
                continue
            length = min(len(base), n_samples - start)
            y[start:start + length] += (base[:length] * gain
                                        * float(rng.uniform(0.6, 1.0)))

    if noise_snr_db is not None:
        noise = _band_noise(rng, n_samples, sr, 30.0, 12000.0)
        power = 10 ** (-noise_snr_db / 20.0)
        y = y + power * float(np.max(np.abs(y)) + 1e-9) * noise

    peak = float(np.max(np.abs(y)) + 1e-9)
    return (0.9 * y / peak).astype(np.float32)


class SyntheticTempoDataset(Dataset):
    """
    On-the-fly percussive loops with exact BPM labels.

    Used as an extra training source: it covers the whole 40-220 BPM range
    continuously (GTZAN annotations are not uniformly distributed) and teaches
    the model to lock onto the beat grid instead of timbre.
    """

    def __init__(
        self,
        length: int = 500,
        duration: float = CROP_SECONDS,
        sr: int = SR,
        n_mels: int = N_MELS,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        fmax: float = FMAX,
        bpm_min: float = 40.0,
        bpm_max: float = 220.0,
        seed: int = 0,
        augment: bool = True,
        noise_prob: float = 0.3,
        noise_snr_db: float = 18.0,
        stretch_prob: float = 0.5,
        max_stretch: float = 0.05,
        pitch_prob: float = 0.3,
        max_semitones: float = 2.0,
        specaug_prob: float = 0.5,
        gain_prob: float = 0.5,
        max_gain_db: float = 6.0,
    ) -> None:
        self.length = int(length)
        self.duration = float(duration)
        self.sr = int(sr)
        self.n_mels = int(n_mels)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.fmax = float(fmax)
        self.frames = n_frames(self.duration, self.sr, self.hop_length)
        self.bpm_min, self.bpm_max = float(bpm_min), float(bpm_max)
        self.seed = int(seed)
        self.augment = bool(augment)
        self.noise_prob, self.noise_snr_db = float(noise_prob), float(noise_snr_db)
        self.stretch_prob, self.max_stretch = stretch_prob, max_stretch
        self.pitch_prob, self.max_semitones = pitch_prob, max_semitones
        self.specaug_prob, self.gain_prob = specaug_prob, gain_prob
        self.max_gain_db = max_gain_db
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.seed + 1_000_003 * self._epoch + idx)
        bpm = float(np.exp(rng.uniform(np.log(self.bpm_min), np.log(self.bpm_max))))
        noise = (self.noise_snr_db if rng.random() < self.noise_prob else None)
        y = synthesize_rhythm(bpm, self.sr, self.duration, rng, noise_snr_db=noise)
        mel = log_mel(y, sr=self.sr, n_mels=self.n_mels, n_fft=self.n_fft,
                      hop_length=self.hop_length, fmax=self.fmax)

        if self.augment:
            if rng.random() < self.pitch_prob:
                mel = pitch_shift_mel(mel, rng.uniform(-self.max_semitones,
                                                       self.max_semitones))
            if rng.random() < self.stretch_prob:
                rate = float(rng.uniform(1.0 - self.max_stretch, 1.0 + self.max_stretch))
                mel = time_stretch_mel(mel, rate)
                bpm *= rate
            if rng.random() < self.gain_prob:
                mel = mel + float(rng.uniform(-self.max_gain_db, self.max_gain_db))
            if rng.random() < self.specaug_prob:
                mel = spec_augment_mel(mel, rng)

        mel = fix_length(mel, self.frames)
        mel = (mel - float(mel.mean())) / float(mel.std() + 1e-8)
        x = torch.tensor(np.ascontiguousarray(mel, dtype=np.float32))[None]
        return x, torch.tensor([bpm], dtype=torch.float32)


class AudioDataset(Dataset):
    """
    Reference-style dataset for *custom* data: a CSV with ``filename,bpm``
    columns plus an audio directory. Spectrograms are computed on the fly
    (for GTZAN use the Mel cache, it is far faster).

    Expected layout::

        data/audio/<file>.wav
        data/train.csv        # columns: filename,bpm

    Raw-audio augmentation: pitch shift +-2 semitones (BPM unchanged) and
    time stretch +-5% (BPM label multiplied by the stretch rate).
    """

    def __init__(
        self,
        csv_file: str | Path,
        audio_dir: str | Path,
        sr: int = SR,
        duration: float = CROP_SECONDS,
        n_mels: int = N_MELS,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        fmax: float = FMAX,
        augment: bool = False,
        seed: int = 42,
        bpm_column: str = "bpm",
        filename_column: str = "filename",
        pitch_prob: float = 0.5,
        max_semitones: float = 2.0,
        stretch_prob: float = 0.3,
        max_stretch: float = 0.05,
        gain_prob: float = 0.5,
        max_gain_db: float = 6.0,
        specaug_prob: float = 0.5,
        limit: Optional[int] = None,
    ) -> None:
        self.data = pd.read_csv(csv_file)
        if limit:
            self.data = self.data.iloc[:limit]
        self.audio_dir = Path(audio_dir)
        self.sr, self.duration = int(sr), float(duration)
        self.n_mels, self.n_fft = int(n_mels), int(n_fft)
        self.hop_length, self.fmax = int(hop_length), float(fmax)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.bpm_column, self.filename_column = bpm_column, filename_column
        self.pitch_prob, self.max_semitones = pitch_prob, max_semitones
        self.stretch_prob, self.max_stretch = stretch_prob, max_stretch
        self.gain_prob, self.max_gain_db = gain_prob, max_gain_db
        self.specaug_prob = specaug_prob
        self._rng: Optional[np.random.Generator] = None
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        self._rng = None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            worker = 0 if info is None else int(info.id) * 100003 + 17
            self._rng = np.random.default_rng(self.seed + worker + 7919 * self._epoch)
        rng = self._rng

        row = self.data.iloc[idx]
        bpm = float(row[self.bpm_column])
        path = self.audio_dir / str(row[self.filename_column])
        y, _ = librosa.load(str(path), sr=self.sr, mono=True, duration=self.duration)
        target = int(self.duration * self.sr)
        y = np.pad(y, (0, target - len(y))) if len(y) < target else y[:target]

        if self.augment:
            if rng.random() < self.pitch_prob:
                n_steps = float(rng.uniform(-self.max_semitones, self.max_semitones))
                y = librosa.effects.pitch_shift(y=y, sr=self.sr, n_steps=n_steps)
            if rng.random() < self.stretch_prob:
                rate = float(rng.uniform(1.0 - self.max_stretch, 1.0 + self.max_stretch))
                y = librosa.effects.time_stretch(y=y, rate=rate)
                bpm *= rate
            if rng.random() < self.gain_prob:
                y = y * float(10 ** (rng.uniform(-self.max_gain_db, self.max_gain_db) / 20.0))

        mel = log_mel(y, sr=self.sr, n_mels=self.n_mels, n_fft=self.n_fft,
                      hop_length=self.hop_length, fmax=self.fmax)
        if self.augment and rng.random() < self.specaug_prob:
            mel = spec_augment_mel(mel, rng)

        mel = (mel - float(mel.mean())) / float(mel.std() + 1e-8)
        x = torch.tensor(np.ascontiguousarray(mel, dtype=np.float32))[None]
        return x, torch.tensor([bpm], dtype=torch.float32)


# --------------------------------------------------------------------------- #
# DataLoader factory
# --------------------------------------------------------------------------- #
@dataclass
class DataBundle:
    """Train/val/test loaders plus the underlying datasets."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_dataset: Dataset
    val_dataset: GTZANDataset
    test_dataset: GTZANDataset
    synthetic_dataset: Optional[SyntheticTempoDataset] = None

    def summary(self) -> Dict[str, int]:
        return {
            "train_real": len(self.train_dataset.datasets[0])
            if isinstance(self.train_dataset, ConcatDataset) else len(self.train_dataset),
            "train_synthetic": 0 if self.synthetic_dataset is None
            else len(self.synthetic_dataset),
            "val": len(self.val_dataset),
            "test": len(self.test_dataset),
        }


def build_dataloaders(
    cache_dir: str | Path = "data/cache",
    batch_size: int = 32,
    eval_batch_size: Optional[int] = None,
    crop_seconds: float = CROP_SECONDS,
    num_workers: int = 4,
    seed: int = 42,
    synthetic: int = 0,
    synthetic_bpm_range: Tuple[float, float] = (40.0, 220.0),
    limit: Optional[int] = None,
    augment: bool = True,
    pin_memory: bool = False,
) -> DataBundle:
    """
    Build the loaders used by ``src.train``.

    Args:
        synthetic: number of synthetic percussive loops added to the *training*
            set each epoch (exact BPM labels). ``0`` disables them.
        limit: cap the number of real training tracks (smoke tests).
    """
    train_set = GTZANDataset(cache_dir, "train", crop_seconds,
                             augment=augment, seed=seed, limit=limit)
    val_set = GTZANDataset(cache_dir, "val", crop_seconds, augment=False, seed=seed)
    test_set = GTZANDataset(cache_dir, "test", crop_seconds, augment=False, seed=seed)

    synthetic_set: Optional[SyntheticTempoDataset] = None
    train_dataset: Dataset = train_set
    if synthetic > 0:
        synthetic_set = SyntheticTempoDataset(
            length=synthetic,
            duration=crop_seconds,
            sr=train_set.sr,
            n_mels=train_set.n_mels,
            n_fft=int(train_set.config["n_fft"]),
            hop_length=train_set.hop_length,
            fmax=float(train_set.config["fmax"]),
            bpm_min=synthetic_bpm_range[0],
            bpm_max=synthetic_bpm_range[1],
            seed=seed + 1,
            augment=augment,
        )
        train_dataset = ConcatDataset([train_set, synthetic_set])

    loader_kwargs: Dict[str, object] = {"num_workers": int(num_workers),
                                        "pin_memory": bool(pin_memory)}
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        drop_last=len(train_dataset) > batch_size, **loader_kwargs,
    )
    eval_bs = int(eval_batch_size or batch_size)
    val_loader = DataLoader(val_set, batch_size=eval_bs, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, batch_size=eval_bs, shuffle=False, **loader_kwargs)

    return DataBundle(train_loader, val_loader, test_loader,
                      train_dataset, val_set, test_set, synthetic_set)


if __name__ == "__main__":
    main()

