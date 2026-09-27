"""Unit tests for the data pipeline: log-Mel, augmentation, synthetic rhythms."""

import numpy as np
import pytest
import torch

import librosa

from src.dataset import (GTZANDataset, build_rhythm_onsets, log_mel,
                         n_frames, pitch_shift_mel, spec_augment_mel,
                         synthesize_rhythm, time_stretch_mel)


def test_log_mel_shape_and_finiteness():
    y = np.random.default_rng(0).standard_normal(2 * 22050).astype(np.float32) * 0.1
    mel = log_mel(y)
    assert mel.shape == (128, n_frames(2.0))
    assert np.isfinite(mel).all()
    assert mel.max() == pytest.approx(0.0, abs=1e-5)      # ref=np.max


def test_time_stretch_changes_length_and_is_reversible():
    mel = np.random.default_rng(1).standard_normal((128, 200)).astype(np.float32)
    faster = time_stretch_mel(mel, 1.1)                    # 10% faster -> shorter
    assert faster.shape[1] == int(round(200 / 1.1))
    slower = time_stretch_mel(mel, 0.9)
    assert slower.shape[1] == int(round(200 / 0.9))
    unchanged = time_stretch_mel(mel, 1.0)
    assert unchanged.shape == mel.shape


def test_pitch_shift_keeps_shape_and_moves_energy_up():
    mel = np.zeros((128, 20), dtype=np.float32)
    mel[10, :] = 1.0
    shifted = pitch_shift_mel(mel, 2.0, bins_per_semitone=4)   # +8 bins
    assert shifted.shape == mel.shape
    assert int(np.argmax(shifted.sum(axis=1))) == 18
    assert shifted[:8].sum() == 0.0                            # zero-padded edge


def test_spec_augment_masks_but_keeps_shape_and_range():
    rng = np.random.default_rng(2)
    mel = np.random.default_rng(3).standard_normal((128, 120)).astype(np.float32)
    augmented = spec_augment_mel(mel, rng, n_freq_masks=2, n_time_masks=2)
    assert augmented.shape == mel.shape
    assert augmented.min() >= mel.min() - 1e-6
    assert not np.array_equal(augmented, mel)


def _onset_autocorr(y: np.ndarray, sr: int = 22050):
    """Autocorrelation of the onset strength envelope (+ frame rate)."""
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    env = env - env.mean()
    autocorr = np.correlate(env, env, mode="full")[len(env) - 1:]
    return autocorr / (autocorr[0] + 1e-9), sr / 512


def test_synthetic_onset_grid_is_locked_to_the_requested_tempo():
    """With a plain 4/4 backbeat every generated onset must sit on the grid."""
    onsets, meta = build_rhythm_onsets(120.0, duration=6.0,
                                       rng=np.random.default_rng(0),
                                       meter=4, subdiv=1, swing=False)
    beat = 60.0 / 120.0
    assert meta["beat"] == pytest.approx(beat)
    assert meta["bpm"] == pytest.approx(120.0)

    every_onset = np.concatenate([onsets["kick"], onsets["snare"], onsets["hat"]])
    assert every_onset.size > 5
    grid_position = every_onset / beat
    deviation = np.abs(grid_position - np.round(grid_position)) * beat
    assert deviation.max() < 0.04                     # only the +-6 ms jitter


@pytest.mark.parametrize("bpm", [80.0, 120.0, 174.0])
def test_synthetic_rhythm_is_periodic_at_the_requested_tempo(bpm):
    y = synthesize_rhythm(bpm, duration=6.0, rng=np.random.default_rng(int(bpm)))
    assert y.shape == (6 * 22050,)
    assert np.isfinite(y).all()
    assert np.max(np.abs(y)) <= 0.91

    autocorr, fps = _onset_autocorr(y)
    beat = 60.0 / bpm
    lag_beat = int(round(fps * beat))
    lag_two_beats = int(round(fps * 2 * beat))
    # the beat (or its multiple) must be a strong periodicity of the onsets
    assert max(autocorr[lag_beat], autocorr[lag_two_beats]) > 0.15


def test_synthetic_rhythm_onsets_are_mostly_on_the_beat_grid():
    onsets, meta = build_rhythm_onsets(174.0, duration=8.0,
                                       rng=np.random.default_rng(7))
    beat = meta["beat"]
    strong = np.concatenate([onsets["kick"], onsets["snare"]])
    phase = (strong / beat) % 1.0
    deviation = np.minimum(phase, 1.0 - phase) * beat
    assert (deviation < 0.030).mean() >= 0.5          # syncopation stays a minority


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_synthetic_rhythm_is_deterministic(seed):
    y1 = synthesize_rhythm(133.0, duration=2.0, rng=np.random.default_rng(seed))
    y2 = synthesize_rhythm(133.0, duration=2.0, rng=np.random.default_rng(seed))
    assert np.allclose(y1, y2)


def test_dataset_requires_a_mel_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        GTZANDataset(cache_dir=tmp_path / "nope", split="train")


def test_dataset_batches_are_shaped_correctly(tmp_path):
    """A tiny hand-made cache exercises the full dataset path without GTZAN."""
    import json

    from src.dataset import HOP_LENGTH, N_MELS, SR

    cache = tmp_path / "cache"
    cache.mkdir()
    total = n_frames(30.0)
    mels = np.random.default_rng(4).standard_normal((4, N_MELS, total)).astype(np.float16)
    np.save(cache / "mels.npy", mels)
    (cache / "cache_config.json").write_text(json.dumps({
        "sr": SR, "n_mels": N_MELS, "n_fft": 2048, "hop_length": HOP_LENGTH,
        "fmax": 8000.0, "duration": 30.0, "n_tracks": 4,
    }))
    rows = [{"track_id": f"g/{i}.wav", "path": "x", "genre": "g", "bpm": 100.0 + i,
             "sha1": f"h{i}", "duration": 30.0, "sample_rate": SR,
             "mel_mean": 0.0, "mel_std": 1.0, "n_frames": total,
             "split": "train" if i < 2 else ("val" if i == 2 else "test")}
            for i in range(4)]
    import pandas as pd
    pd.DataFrame(rows).to_csv(cache / "index.csv", index=False)

    dataset = GTZANDataset(cache, "train", crop_seconds=10.0, augment=True)
    x, y = dataset[0]
    assert x.shape == (1, N_MELS, n_frames(10.0))
    assert y.shape == (1,)
    assert torch.isfinite(x).all()
    assert abs(float(x.mean())) < 1e-3                 # per-crop standardisation

    eval_dataset = GTZANDataset(cache, "val", crop_seconds=10.0, augment=False)
    x_eval, y_eval = eval_dataset[0]
    assert x_eval.shape == (1, N_MELS, n_frames(10.0))
    assert float(y_eval) == 102.0
