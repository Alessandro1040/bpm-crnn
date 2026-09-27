"""
BeatNet-CRNN: Convolutional Recurrent Neural Network for BPM estimation.

Architecture (compact state-of-the-art design)
----------------------------------------------
  Mel spectrogram (1 x n_mels x T)
        |
  VGG-style CNN -> (C x 16 x T')      frequency pooling only in block 3
        |
  Bi-GRU x2 (256) -> per-frame embeddings (B, T', 512)
        |
  +-- frame_classifier  -> per-frame BPM distribution (B, T', n_classes)
  +-- attention pooling -> global embedding (B, 512)
  |        +-- song_classifier -> song-level BPM distribution (B, n_classes)
        |
  BPM = expectation of the distribution over BPM bins

The network is a *classifier over BPM bins* whose final output is the
expected value (a regression): this yields both a continuous BPM estimate and
a calibrated distribution, which is what makes the auxiliary cross-entropy
losses meaningful.

Training losses (see ``src/train.py``):
  * Gaussian soft targets over the BPM bins (sigma expressed in bins)
  * SmoothL1 on BPM (linear) + SmoothL1 on log2(BPM) (tempo aware)
  * frame-level and song-level auxiliary cross entropy

The public forward API stays compatible with the reference snippet::

    bpm, frame_logits = model(x)

``return_all=True`` additionally returns the song logits, the attention
weights and the two separate estimates (frame / song).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Default tempo range / resolution: 256 linear bins in [30, 285] BPM
DEFAULT_BPM_MIN = 30.0
DEFAULT_BPM_MAX = 285.0
DEFAULT_N_CLASSES = 256


def bpm_bins(
    bpm_min: float = DEFAULT_BPM_MIN,
    bpm_max: float = DEFAULT_BPM_MAX,
    n_classes: int = DEFAULT_N_CLASSES,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Bin centres (in BPM) of the classifier head."""
    return torch.linspace(bpm_min, bpm_max, n_classes, device=device)


def gaussian_soft_targets(
    bpm: torch.Tensor,
    bins: torch.Tensor,
    sigma_bins: float = 3.0,
) -> torch.Tensor:
    """
    Convert continuous BPM values into Gaussian soft targets over the bins.

    Args:
        bpm: (N,) or (N, T) BPM values.
        bins: (C,) bin centres in BPM.
        sigma_bins: std of the Gaussian, in bin units.

    Returns:
        (..., C) distribution normalised over the last axis.
    """
    delta = (bpm.unsqueeze(-1) - bins) / float(bins[1] - bins[0])
    logits = -0.5 * (delta / sigma_bins) ** 2
    keep = delta.abs() <= 3.0 * sigma_bins
    logits = torch.where(keep, logits, torch.full_like(logits, -1e4))
    return F.softmax(logits, dim=-1)


class BeatNetCRNN(nn.Module):
    """
    CRNN for BPM estimation.

    Input : (Batch, 1, n_mels, T)  log-Mel spectrogram, per-sample standardised
    Output: (Batch,) BPM  (+ per-frame logits)

    Args:
        n_mels: Mel bands (must be divisible by 8: three 2x frequency poolings).
        n_classes: size of the BPM bin vocabulary.
        bpm_min / bpm_max: tempo range covered by the bins.
        dropout: dropout inside the GRU / before the heads.
        hidden_size: GRU hidden size (bidirectional -> 2 * hidden_size).
        gru_layers: number of stacked bidirectional GRU layers.
        combine: fusion of the frame-level and song-level estimates
            ('mean', 'frame' or 'song').
    """

    def __init__(
        self,
        n_mels: int = 128,
        n_classes: int = DEFAULT_N_CLASSES,
        bpm_min: float = DEFAULT_BPM_MIN,
        bpm_max: float = DEFAULT_BPM_MAX,
        dropout: float = 0.3,
        hidden_size: int = 256,
        gru_layers: int = 2,
        combine: str = "mean",
    ) -> None:
        super().__init__()
        if combine not in ("mean", "frame", "song"):
            raise ValueError(f"unknown combine mode: {combine!r}")
        if n_mels % 8 != 0:
            raise ValueError("n_mels must be divisible by 8 (three 2x freq poolings)")

        self.n_mels = int(n_mels)
        self.n_classes = int(n_classes)
        self.bpm_min = float(bpm_min)
        self.bpm_max = float(bpm_max)
        self.dropout = float(dropout)
        self.hidden_size = int(hidden_size)
        self.gru_layers = int(gru_layers)
        self.combine = combine

        # ------------------------------------------------------------------ CNN
        self.conv = nn.Sequential(
            # Block 1: freq 128 -> 64, time /2
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: freq 64 -> 32, time /2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: freq 32 -> 16, time preserved (beat-level resolution)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(dropout),
        )
        self.n_freq_out = self.n_mels // 8                    # 16
        self.n_channels_out = 128
        gru_input = self.n_channels_out * self.n_freq_out     # 2048

        # ------------------------------------------------------------------ RNN
        self.gru = nn.GRU(
            input_size=gru_input,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        feat_dim = 2 * hidden_size                            # 512

        # ---------------------------------------------------------------- heads
        self.frame_classifier = nn.Linear(feat_dim, n_classes)

        self.attention = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1),                                # over time
        )

        self.song_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, n_classes),
        )

        self.register_buffer("bins", bpm_bins(bpm_min, bpm_max, n_classes))

        self._reset_parameters()

    # ------------------------------------------------------------- properties
    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _reset_parameters(self) -> None:
        """Kaiming init for the convolutions (the heads keep PyTorch defaults)."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                        nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def arch_config(self) -> Dict[str, object]:
        """Constructor arguments (stored inside checkpoints)."""
        return {
            "n_mels": self.n_mels,
            "n_classes": self.n_classes,
            "bpm_min": self.bpm_min,
            "bpm_max": self.bpm_max,
            "dropout": self.dropout,
            "hidden_size": self.hidden_size,
            "gru_layers": self.gru_layers,
            "combine": self.combine,
        }

    # ---------------------------------------------------------------- forward
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, F, T) -> (B, T', 2 * hidden_size) per-frame embeddings."""
        h = self.conv(x)
        b, c, f, t = h.shape
        h = h.permute(0, 3, 1, 2).reshape(b, t, c * f)
        h, _ = self.gru(h)
        return h

    def forward(self, x: torch.Tensor, return_all: bool = False):
        """
        Args:
            x: (B, 1, n_mels, T) or (B, n_mels, T) log-Mel input.

        Returns:
            ``(bpm, frame_logits)`` by default, otherwise a dict with
            ``bpm``, ``bpm_frame``, ``bpm_song``, ``frame_logits``,
            ``song_logits`` and ``attn``.
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        h = self.encode(x)                                # (B, T', 512)
        frame_logits = self.frame_classifier(h)           # (B, T', C)
        attn = self.attention(h)                          # (B, T', 1)
        pooled = torch.sum(h * attn, dim=1)               # (B, 512)
        song_logits = self.song_classifier(pooled)        # (B, C)

        frame_probs = F.softmax(frame_logits, dim=-1)
        frame_bpms = frame_probs @ self.bins              # (B, T')
        frame_bpm = torch.sum(frame_bpms * attn.squeeze(-1), dim=1)
        song_bpm = F.softmax(song_logits, dim=-1) @ self.bins

        if self.combine == "frame":
            bpm = frame_bpm
        elif self.combine == "song":
            bpm = song_bpm
        else:
            bpm = 0.5 * (frame_bpm + song_bpm)

        if not return_all:
            return bpm, frame_logits

        return {
            "bpm": bpm,
            "bpm_frame": frame_bpm,
            "bpm_song": song_bpm,
            "frame_logits": frame_logits,
            "song_logits": song_logits,
            "attn": attn,
        }

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Song-level BPM distribution + point estimates (use in eval mode)."""
        out = self.forward(x, return_all=True)
        return {
            "probs": F.softmax(out["song_logits"], dim=-1),
            "bpm": out["bpm"],
            "bpm_frame": out["bpm_frame"],
            "bpm_song": out["bpm_song"],
        }


def build_model_from_config(config: Dict[str, object]) -> BeatNetCRNN:
    """Rebuild a ``BeatNetCRNN`` from the architecture config of a checkpoint."""
    keys = ("n_mels", "n_classes", "bpm_min", "bpm_max", "dropout",
            "hidden_size", "gru_layers", "combine")
    return BeatNetCRNN(**{k: config[k] for k in keys if k in config})
