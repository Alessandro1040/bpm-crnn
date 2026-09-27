# BeatNet-CRNN - training results

* checkpoint: `runs/gtzan/best.pth` (best epoch 54 of 60)
* test tracks: 100 (held-out, grouped split, no leakage)
* training time: 29m23s
* command: `python -m src.train --out runs/gtzan --epochs 60 --synthetic 500 --batch-size 32 --lr 0.0003`
* finished at: 2026-09-27 16:47:48

## Test metrics (GTZAN + gtzan_tempo_beat reference annotations)

| metric | mean output | frame only | song only |
|---|---|---|---|
| MAE | 15.822 | 16.234 | 15.646 |
| RMSE | 29.577 | 29.753 | 29.539 |
| MedianAE | 4.451 | 4.732 | 4.363 |
| Acc_1BPM | 17.0% | 10.0% | 21.0% |
| Acc_1% | 24.0% | 15.0% | 24.0% |
| Acc_2% | 38.0% | 33.0% | 34.0% |
| Acc_5% | 55.0% | 53.0% | 57.0% |
| Acc_octave_4% | 53.0% | 52.0% | 57.0% |
| Octave_error_rate_4% | 17.0% | 16.0% | 17.0% |
| P_score | 0.485 | 0.462 | 0.495 |
| Cemgil | 0.449 | 0.431 | 0.462 |

`Acc_octave_4%` is the octave-tolerant accuracy (correct within 4 % up to a
factor 1/2 or 2), `Octave_error_rate_4%` the share of predictions whose best
factor is not 1 (half/double tempo confusion), `P_score` (McKinney) and
`Cemgil` are the standard MIR tempo scores in log2-tempo space.

## Files

| file | content |
|---|---|
| `history.csv` | per-epoch train/val metrics |
| `test_metrics.json` | full metric set (mean/frame/song) on the test split |
| `predictions_test.csv` | per-track prediction vs reference (via `src.evaluate`) |
| `assets/*.png` | scatter + error histogram + training curves (via `src.evaluate`) |
| `best.pth` / `last.pth` | checkpoints (model + optimizer + history) |
