"""Unit tests for the tempo metrics (including octave-error handling)."""

import numpy as np
import pytest

from src.metrics import (format_metrics, octave_factor, tempo_metrics,
                         OCTAVE_FACTORS)


def test_perfect_predictions():
    reference = np.array([60.0, 120.0, 90.0, 200.0])
    metrics = tempo_metrics(reference.copy(), reference)
    assert metrics["MAE"] == pytest.approx(0.0)
    assert metrics["Acc_1BPM"] == 1.0
    assert metrics["Acc_1%"] == 1.0
    assert metrics["Acc_octave_4%"] == 1.0
    assert metrics["Octave_error_rate_4%"] == 0.0
    assert metrics["P_score"] == pytest.approx(1.0)
    assert metrics["Cemgil"] == pytest.approx(1.0)


def test_octave_error_is_detected():
    reference = np.array([60.0, 60.0])
    prediction = np.array([120.0, 30.0])            # double and half tempo
    metrics = tempo_metrics(prediction, reference)
    assert metrics["MAE"] == pytest.approx(45.0)     # plain MAE is fooled
    assert metrics["Acc_1%"] == 0.0
    assert metrics["Acc_octave_1%"] == 1.0           # but octave aware is perfect
    assert metrics["Octave_error_rate_1%"] == 1.0
    assert metrics["factor_2_4%"] == pytest.approx(0.5)
    assert metrics["factor_0.5_4%"] == pytest.approx(0.5)
    assert metrics["factor_1_4%"] == 0.0


def test_octave_factor_returns_best_factor():
    factors = octave_factor(np.array([120.0, 61.0, 29.0]), np.array([60.0, 60.0, 60.0]))
    assert list(factors) == [2.0, 1.0, 0.5]
    assert tuple(OCTAVE_FACTORS) == (0.5, 1.0, 2.0)


def test_relative_tolerance():
    reference = np.array([100.0, 100.0, 100.0])
    prediction = np.array([100.9, 104.9, 106.0])
    metrics = tempo_metrics(prediction, reference)
    assert metrics["Acc_1%"] == pytest.approx(1 / 3)     # only the 0.9 % error
    assert metrics["Acc_5%"] == pytest.approx(2 / 3)
    assert metrics["Acc_1BPM"] == pytest.approx(1 / 3)


def test_cemgil_penalises_octave_errors():
    reference = np.array([60.0])
    assert tempo_metrics(np.array([60.0]), reference)["Cemgil"] == pytest.approx(1.0)
    assert tempo_metrics(np.array([120.0]), reference)["Cemgil"] < 0.01


def test_format_metrics_contains_key_numbers():
    reference = np.array([88.0, 92.0])
    text = format_metrics(tempo_metrics(reference.copy(), reference))
    assert "MAE" in text and "P_score" in text and "Acc_1BPM" in text
