"""Unit tests for the BeatNet-CRNN model."""

import pytest
import torch

from src.model import (DEFAULT_BPM_MAX, DEFAULT_BPM_MIN, BeatNetCRNN, bpm_bins,
                       build_model_from_config, gaussian_soft_targets)


def test_bpm_bins_range():
    bins = bpm_bins(DEFAULT_BPM_MIN, DEFAULT_BPM_MAX, 256)
    assert bins.shape == (256,)
    assert float(bins[0]) == pytest.approx(30.0)
    assert float(bins[-1]) == pytest.approx(285.0)


def test_soft_targets_are_distributions_centred_on_the_label():
    bins = bpm_bins()
    targets = gaussian_soft_targets(torch.tensor([120.0, 60.0]), bins, sigma_bins=3.0)
    assert targets.shape == (2, 256)
    assert torch.allclose(targets.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert int(targets[0].argmax()) == int(torch.argmin((bins - 120.0).abs()))
    assert int(targets[1].argmax()) == int(torch.argmin((bins - 60.0).abs()))


def test_forward_shapes_and_positivity():
    model = BeatNetCRNN()
    x = torch.randn(3, 1, 128, 107)                 # 10 s crop at hop 512
    bpm, frame_logits = model(x)
    assert bpm.shape == (3,)
    assert frame_logits.shape == (3, 107 // 4, 256)  # time is halved twice
    assert torch.all(bpm > 0)
    assert torch.all(bpm >= DEFAULT_BPM_MIN - 1e-3)
    assert torch.all(bpm <= DEFAULT_BPM_MAX + 1e-3)


def test_forward_all_outputs_and_accepts_3d_input():
    model = BeatNetCRNN()
    x = torch.randn(2, 128, 50)
    out = model(x, return_all=True)
    for key in ("bpm", "bpm_frame", "bpm_song", "frame_logits", "song_logits", "attn"):
        assert key in out
    assert out["song_logits"].shape == (2, 256)
    assert out["attn"].shape == (2, 50 // 4, 1)
    assert torch.allclose(out["attn"].sum(dim=1), torch.ones(2, 1), atol=1e-4)
    assert torch.allclose(out["bpm"], 0.5 * (out["bpm_frame"] + out["bpm_song"]))


def test_backward_pass_produces_gradients():
    model = BeatNetCRNN(dropout=0.0)
    x = torch.randn(2, 1, 128, 40, requires_grad=False)
    target = torch.tensor([128.0, 90.0])
    bpm, frame_logits = model(x)
    loss = torch.nn.functional.smooth_l1_loss(bpm, target) + frame_logits.sum() * 0.0
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_short_input_is_handled():
    model = BeatNetCRNN()
    bpm, logits = model(torch.randn(1, 1, 128, 8))
    assert bpm.shape == (1,)
    assert logits.shape[:2] == (1, 2)              # 8 frames -> 2 after two poolings


def test_invalid_configuration_raises():
    with pytest.raises(ValueError):
        BeatNetCRNN(combine="nope")
    with pytest.raises(ValueError):
        BeatNetCRNN(n_mels=100)


def test_build_from_arch_config_round_trip():
    model = BeatNetCRNN(n_mels=64, hidden_size=64, gru_layers=1, dropout=0.1)
    clone = build_model_from_config(model.arch_config())
    assert clone.arch_config() == model.arch_config()
    assert clone.n_params == model.n_params
