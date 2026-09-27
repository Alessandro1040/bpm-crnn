"""
Evaluation metrics for BPM / tempo estimation.

Beyond plain MAE (which is misleading for tempo because of the well known
"octave errors": 60 vs 120 vs 180 BPM are all plausible interpretations), this
module reports the metrics used in the MIR literature:

  * MAE / RMSE / median absolute error / bias (in BPM)
  * Acc@1BPM  : share of predictions with |error| < 1 BPM
  * Acc@1%, Acc@2%, Acc@4%, Acc@5% : share with relative error below the threshold
  * Acc-octave : share of predictions that are correct *up to a factor*
                 in {1/2, 1, 2} (octave tolerant accuracy)
  * Octave error rate : share of predictions whose best factor is not 1
  * P-score (McKinney) : Gaussian score in log2-tempo space, max over
                 {1/2, 1, 2} factors, sigma = 4%
  * Cemgil score : Gaussian score in log2-tempo space, no octave allowance

All functions accept array-like inputs (numpy arrays, lists or torch tensors).
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np

OCTAVE_FACTORS: tuple = (0.5, 1.0, 2.0)
SIGMA_LOG2 = 0.04  # 4% tolerance, McKinney / Cemgil convention


def _as_1d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("empty input")
    return arr


def absolute_error(pred, true) -> np.ndarray:
    return np.abs(_as_1d(pred) - _as_1d(true))


def relative_error(pred, true) -> np.ndarray:
    return absolute_error(pred, true) / np.maximum(_as_1d(true), 1e-9)


def octave_factor(pred, true, factors: Sequence[float] = OCTAVE_FACTORS) -> np.ndarray:
    """
    For every prediction, the factor in ``factors`` that best matches the
    reference tempo (i.e. which octave the model landed in).
    """
    pred, true = _as_1d(pred), _as_1d(true)
    err = np.stack([relative_error(pred / f, true) for f in factors], axis=-1)
    return np.asarray(factors, dtype=np.float64)[np.argmin(err, axis=-1)]


def tempo_metrics(
    pred,
    true,
    *,
    factors: Iterable[float] = OCTAVE_FACTORS,
    sigma_log2: float = SIGMA_LOG2,
) -> Dict[str, float]:
    """Compute the full metric set for a batch of BPM predictions."""
    pred, true = _as_1d(pred), _as_1d(true)
    factors = np.asarray(list(factors), dtype=np.float64)

    abs_err = np.abs(pred - true)
    rel_err = abs_err / np.maximum(true, 1e-9)

    # ---- octave aware -----------------------------------------------------
    rel_err_all = np.stack([relative_error(pred / f, true) for f in factors], axis=-1)
    best_idx = np.argmin(rel_err_all, axis=-1)
    best_factor = factors[best_idx]
    best_rel_err = rel_err_all[np.arange(rel_err_all.shape[0]), best_idx]

    # ---- log2 space scores ------------------------------------------------
    log2_ratio = np.log2(np.maximum(pred, 1e-6) / np.maximum(true, 1e-6))
    cemgil = float(np.mean(np.exp(-0.5 * (log2_ratio / sigma_log2) ** 2)))
    pscore_terms = np.stack(
        [np.exp(-0.5 * (np.log2(np.maximum(pred, 1e-6) / (f * np.maximum(true, 1e-6))) / sigma_log2) ** 2)
         for f in factors],
        axis=-1,
    )
    p_score = float(np.mean(np.max(pscore_terms, axis=-1)))

    metrics = {
        "n": int(pred.size),
        "MAE": float(abs_err.mean()),
        "RMSE": float(np.sqrt((abs_err ** 2).mean())),
        "MedianAE": float(np.median(abs_err)),
        "Bias": float((pred - true).mean()),
        "Acc_1BPM": float((abs_err < 1.0).mean()),
        "Acc_1%": float((rel_err < 0.01).mean()),
        "Acc_2%": float((rel_err < 0.02).mean()),
        "Acc_4%": float((rel_err < 0.04).mean()),
        "Acc_5%": float((rel_err < 0.05).mean()),
        "Acc_octave_1%": float((best_rel_err < 0.01).mean()),
        "Acc_octave_4%": float((best_rel_err < 0.04).mean()),
        "Acc_octave_5%": float((best_rel_err < 0.05).mean()),
        "Octave_error_rate_1%": float((best_factor != 1.0).mean()),
        "Octave_error_rate_4%": float((np.abs(best_factor - 1.0) > 1e-9).mean()),
        "P_score": p_score,
        "Cemgil": cemgil,
    }
    # per factor breakdown (how often the model answers half / same / double)
    for f in factors:
        key = f"factor_{f:g}_4%"
        metrics[key] = float(((np.abs(best_factor - f) < 1e-9) & (best_rel_err < 0.04)).mean())
    return metrics


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """One-line (or compact multi-line) human readable rendering."""
    pct_keys = [k for k in metrics if k.startswith(("Acc", "Octave_error_rate", "factor"))]
    lines = []
    for key, value in metrics.items():
        if key == "n" or key in ("P_score", "Cemgil"):
            continue
        if key in pct_keys:
            continue
        lines.append(f"{prefix}{key}: {value:.3f}")
    acc = " | ".join(
        f"{k}={100.0 * metrics[k]:5.1f}%"
        for k in ("Acc_1BPM", "Acc_1%", "Acc_5%", "Acc_octave_4%")
    )
    lines.append(
        f"{prefix}P_score={metrics['P_score']:.3f} Cemgil={metrics['Cemgil']:.3f} "
        f"oct_err(4%)={100.0 * metrics['Octave_error_rate_4%']:.1f}%"
    )
    lines.append(f"{prefix}{acc}")
    return "\n".join(lines)
