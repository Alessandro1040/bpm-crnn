# BeatNet-CRNN — BPM (tempo) estimation from Mel spectrograms

A compact **CRNN** (Convolutional Recurrent Neural Network) that estimates the
tempo of music in BPM, with a **per-frame** head (local tempo tracking) and a
**song-level attention pooling** head. The final BPM is the *expected value of a
learned distribution over 256 BPM bins* in `[30, 285]` BPM, which makes the
estimate continuous **and** gives a calibrated confidence for free.

```
Mel (1 x 128 x T) → VGG-style CNN (freq pooling only) → Bi-GRU x2 (256)
      ├── frame_classifier   → per-frame BPM distribution
      └── attention pooling  → song_classifier → song BPM distribution
BPM = Σ p_i · bin_i     (both paths, then averaged)
```

* **~5.34 M parameters**, trains on a laptop GPU (Apple MPS / CUDA / CPU).
* **Trained on real data**: GTZAN (998 tracks) + the `gtzan_tempo_beat`
  reference tempo annotations, plus a **synthetic percussive dataset with exact
  BPM labels** to cover the tempo range continuously.
* **Metrics with octave-error awareness** (`Acc@1BPM`, `Acc@1/2/4/5%`,
  octave-tolerant accuracy, P-score, Cemgil) — plain MAE alone is misleading for
  tempo because 60/120/180 BPM are all plausible readings of the same music.
* Inference with **overlapping windows + median voting** and optional
  **test-time augmentation** (pitch shift ±1 semitone).

---

## Notebooks (Colab)

Ogni notebook clona il repo, scarica i **pesi** (`models/bpm_crnn_gtzan.pt`, 21 MB,
committato qui e allegato alla [release v1.0](https://github.com/Alessandro1040/bpm-crnn/releases/tag/v1.0-gtzan))
e gira end-to-end senza setup:

| notebook | cosa fa | |
|---|---|---|
| [`01_demo_stima_bpm`](notebooks/01_demo_stima_bpm.ipynb) | carica i pesi da GitHub e stima il BPM: demo autocontenuta su loop a BPM noto + **il tuo file** + grafici (mel, BPM per finestra, distribuzione) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Alessandro1040/bpm-crnn/blob/main/notebooks/01_demo_stima_bpm.ipynb) |
| [`02_valutazione_gtzan`](notebooks/02_valutazione_gtzan.ipynb) | scarica GTZAN + annotazioni, ricostruisce la cache Mel e misura **tutte le metriche** sul test set; confronto con i numeri di `results/` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Alessandro1040/bpm-crnn/blob/main/notebooks/02_valutazione_gtzan.ipynb) |
| [`03_training_colab`](notebooks/03_training_colab.ipynb) | **addestra da zero** su GPU (T4), valuta e confronta col modello rilasciato; export dei pesi per l'inferenza | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Alessandro1040/bpm-crnn/blob/main/notebooks/03_training_colab.ipynb) |
| [`04_test_pipeline`](notebooks/04_test_pipeline.ipynb) | **verifica** la pipeline: 28 test unitari, smoke test end-to-end su un mini-dataset sintetico (nessun download), coerenza dei pesi pubblicati | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Alessandro1040/bpm-crnn/blob/main/notebooks/04_test_pipeline.ipynb) |

## Repository layout

```
bpm-crnn/
├── src/
│   ├── model.py      # BeatNetCRNN (+ soft-target helper)
│   ├── dataset.py    # metadata, Mel cache, augmentation, synthetic rhythms, loaders
│   ├── metrics.py    # tempo metrics incl. octave-error handling
│   ├── train.py      # multi-task training loop, checkpoints, resume, reports
│   ├── evaluate.py   # test/val report, per-genre table, plots
│   ├── infer.py      # BPMExtractor + CLI
│   └── utils.py      # device/seeding/logger/checkpoint helpers
├── notebooks/        # 4 Colab notebooks (demo, evaluation, training, verification)
├── models/
│   └── bpm_crnn_gtzan.pt   # inference-only weights (21 MB) + Mel config + metrics
├── scripts/
│   ├── prepare_data.sh        # download GTZAN + annotations, build cache
│   ├── run_training.sh        # background training (nohup + caffeinate + pid file)
│   ├── post_training.sh       # evaluate + plots + README results (+ push)
│   ├── export_inference.py    # strip optimizer state -> models/*.pt
│   ├── update_readme_results.py
│   └── status.sh              # one-glance run status
├── results/          # metrics, history, per-track predictions of the released run
├── assets/           # plots (predictions scatter, error histogram, training curves)
├── tests/            # 28 unit tests (model, metrics, data pipeline, augmentation)
└── conftest.py
```

## Quick start

Senza installare nulla: apri il [**notebook 01 (demo)**](notebooks/01_demo_stima_bpm.ipynb)
su Colab — carica i pesi gia' addestrati da GitHub e stima il BPM di un tuo file.

In locale:

```bash
# 1) environment (Python >= 3.10)
python3 -m pip install -r requirements.txt

# 2) data: GTZAN audio + tempo annotations + metadata + Mel cache (~1.6 GB total)
bash scripts/prepare_data.sh

# 3) train (background, survives terminal close) -- ~30-60 min on an M5
bash scripts/run_training.sh                     # defaults: 60 epochs, runs/gtzan

# 4) results
python -m src.evaluate --ckpt runs/gtzan/best.pth      # metrics + per-genre + plots
python -m src.infer --ckpt runs/gtzan/best.pth --audio song.mp3
```

I pesi del modello pubblicato sono nel repo come file **inference-only** (21 MB,
senza stato dell'ottimizzatore), quindi per l'inferenza non serve allenare nulla:

```bash
python -m src.infer --ckpt models/bpm_crnn_gtzan.pt --audio song.mp3     # -> 128.05
python scripts/export_inference.py --ckpt runs/gtzan/best.pth \
    --out models/bpm_crnn_gtzan.pt            # come e' stato prodotto

> **Disk space**: the dataset needs ~1.2 GB (audio) + ~0.35 GB (Mel cache) +
> ~0.15 GB (checkpoints). `scripts/prepare_data.sh` deletes the 1.2 GB tarball
> right after extraction. Keep at least ~1.5 GB free while training.

---

## Data

| source | content | size |
|---|---|---|
| [`marsyas/gtzan`](https://huggingface.co/datasets/marsyas/gtzan) | 1000 × 30 s tracks, 10 genres | ~1.2 GB |
| [`TempoBeatDownbeat/gtzan_tempo_beat`](https://github.com/TempoBeatDownbeat/gtzan_tempo_beat) | per-track tempo/beat annotations (999 `.bpm`) | ~5 MB |

Pipeline (`src/dataset.py`):

1. **Metadata**: matches `<genre>/<genre>.<index>.wav` with
   `gtzan_<genre>_<index>.bpm`, reads BPM/duration, hashes every file.
   998 tracks get an annotation; **unreadable files are skipped**
   (GTZAN ships a few corrupted files, e.g. `jazz.00054.wav`).
2. **Grouped split** (798/100/100): GTZAN contains duplicated audio *across
   genres*, so the split is done on the audio hash **globally**, then stratified
   by genre. A leakage assertion fails the build if any hash spans two splits.
3. **Mel cache**: `mels.npy` (float16 memmap, 998 × 128 × 1292 = 330 MB) →
   training epochs are then pure compute (a few seconds of I/O per epoch).
   Settings: `sr=22050, n_mels=128, n_fft=2048, hop=512, fmax=8000`, power→dB,
   per-crop standardisation (the reference normalisation).

**Augmentation** (all label-consistent, in the Mel domain so it is cheap):

| transform | effect | label |
|---|---|---|
| time stretch ±5 % | linear interpolation along time | `bpm *= rate` |
| pitch shift ±2 semitones | roll along the frequency axis | unchanged |
| SpecAugment | 2 freq + 2 time masks | unchanged |
| gain ±6 dB | additive shift in dB | unchanged |

**Synthetic data** (`SyntheticTempoDataset`, `--synthetic N`): percussive loops
(kick/snare/hi-hat, random meter, subdivision, swing, jitter, optional noise)
rendered on the fly with an **exact BPM** label, log-uniform in 40–220 BPM.
This fills the tempo range continuously (GTZAN is clustered around 100–130 BPM)
and forces the model to lock onto the beat grid instead of timbre. It is added
to the training set only — validation and test are 100 % real GTZAN.

---

## Training

```bash
# background launch (nohup + caffeinate + pid file), default 60 epochs on runs/gtzan
bash scripts/run_training.sh

# or with custom settings
OUT=runs/gtzan EPOCHS=80 BATCH=32 SYNTHETIC=500 WORKERS=4 \
  EXTRA_ARGS="--lr 3e-4 --eval-batch-size 64" bash scripts/run_training.sh

# foreground / debug (smoke test: 2 epochs on 40 tracks)
python -m src.train --out runs/smoke --epochs 2 --limit 40 --synthetic 20 --num-workers 2
```

### Multi-task loss

| term | weight | purpose |
|---|---|---|
| SmoothL1 on BPM (β = 1 BPM) | `--w-reg 1.0` | point accuracy |
| SmoothL1 on `log2(BPM)` (β = 0.02) | `--w-log 1.0` | tempo-aware, ±relative errors |
| cross entropy vs **Gaussian soft targets** (σ = 3 bins) on the song head | `--w-song 1.0` | calibrated distribution |
| same soft-target CE on all **frames** | `--w-frame 0.5` | local tempo tracking |

Soft targets are what make the auxiliary classification losses act as a
*regressor*: instead of one-hot bins (`bpm → index`, which quantises to 1 BPM),
the target is a Gaussian over the bins, so the network learns a smooth,
calibrated distribution whose expected value is the BPM.

Optimiser AdamW, 3-epoch linear warmup then cosine decay to 5 % of the LR,
gradient clipping at 5.0, `dropout=0.3` inside the GRU and before the heads.

### Main flags

| flag | default | meaning |
|---|---|---|
| `--epochs` | 60 | training epochs |
| `--batch-size` / `--eval-batch-size` | 32 / 64 | batch sizes |
| `--lr` / `--weight-decay` | 3e-4 / 1e-4 | AdamW hyper-parameters |
| `--crop-seconds` | 10 | analysis window (train **and** eval) |
| `--synthetic` | 500 | synthetic loops added per epoch (0 = real data only) |
| `--sigma-bins` | 3.0 | Gaussian soft-target width (in BPM bins) |
| `--num-workers` | 4 | DataLoader workers |
| `--device` | auto | `mps` → `cuda` → `cpu` |
| `--limit` | – | cap training tracks (smoke tests) |
| `--max-hours` | 0 | graceful stop after N hours (0 = unlimited) |
| `--resume` | – | continue from `last.pth` |
| `--no-augment` | – | disable augmentation |

### How to monitor the run (and know it finished)

```bash
bash scripts/status.sh                 # pid state + status.json + last metrics + log tail

tail -f runs/gtzan/train.log           # live per-step log
cat runs/gtzan/status.json             # {"state": "training"|"done", epoch, progress_pct, ETA, best_val_MAE}
cat runs/gtzan/history.csv             # one row per epoch (all metrics)
column -s, -t < runs/gtzan/history.csv | tail -5
```

The run is **finished** when this file exists (written only in the final block,
after the test-split evaluation):

```bash
cat runs/gtzan/TRAINING_COMPLETE       # JSON: epochs done, best epoch, final test metrics
cat runs/gtzan/RESULTS.md              # ready-to-paste markdown table
```

If `status.json` says `"state": "training"` and the pid in `train.pid` is alive,
it is still running. To stop and resume later (checkpoints contain the model,
the optimiser, the epoch counter and the history):

```bash
kill "$(cat runs/gtzan/train.pid)"
EXTRA_ARGS="--resume" bash scripts/run_training.sh
```

Expected runtime on an Apple M5 (MPS, fp32): **~38 samples/s** with 10 s crops and
batch 32 → 1298 training samples per epoch ≈ **35–40 s/epoch**, so 60 epochs take
roughly **35–45 minutes** (plus the per-epoch validation pass).

---

## Evaluation

```bash
python -m src.evaluate --ckpt runs/gtzan/best.pth                 # test split
python -m src.evaluate --ckpt runs/gtzan/best.pth --split val     # validation split
```

It prints the **full metric set for the three output paths** (`mean`, `frame`,
`song`), a **per-genre breakdown**, the octave-assignment histogram, and writes:

| file | content |
|---|---|
| `runs/gtzan/predictions_test.csv` | per-track `bpm_pred`, `error`, `octave_factor` |
| `runs/gtzan/metrics_report_test.json` | machine-readable report |
| `assets/predictions_test.png` | prediction-vs-reference scatter (+ half/double tempo guides) and error histogram |
| `assets/training_curves.png` | loss / MAE / accuracy per epoch |

### Metrics

| metric | definition |
|---|---|
| `MAE`, `RMSE`, `MedianAE`, `Bias` | error in BPM (Bias = mean signed error) |
| `Acc_1BPM` | share with \|error\| < 1 BPM |
| `Acc_1%`, `Acc_2%`, `Acc_4%`, `Acc_5%` | share with relative error below the threshold |
| `Acc_octave_1%`, `Acc_octave_4%` | **octave-tolerant** accuracy: correct up to a factor in {1/2, 1, 2} |
| `Octave_error_rate_4%` | share of predictions whose best factor is not 1 (half/double tempo confusion) |
| `factor_*_4%` | where the predictions landed (0.5×, 1×, 2×) |
| `P_score` | McKinney: mean of `exp(-½(Δlog2 tempo / 0.04)²)`, max over {1/2, 1, 2} |
| `Cemgil` | same Gaussian score without octave allowance |

Tempo is intrinsically ambiguous (a 4/4 loop at 120 BPM is also "60 BPM with 8th
notes"), which is why plain MAE is not enough: a model that is always one octave
off has a huge MAE but is musically right. Always report `Acc_octave_4%` together
with `Octave_error_rate_4%`.

## Inference

```python
from src.infer import BPMExtractor

extractor = BPMExtractor("runs/gtzan/best.pth")       # device auto-detected
bpm = extractor.predict("song.mp3")                   # -> float, e.g. 128.43

details = extractor.predict("song.mp3", return_details=True)
print(details["bpm"], details["n_chunks"], details["bpm_std"])
```

```bash
python -m src.infer --ckpt runs/gtzan/best.pth --audio song.mp3 intro.wav --json out.json
```

* Files longer than 10 s are analysed with **overlapping 10 s windows** (5 s hop)
  and the final estimate is the **median** across windows, which is robust against
  tempo changes and to a single bad window.
* **Test-time augmentation**: each window is evaluated at 0 and ±1 semitone and
  the three outputs are averaged before the median (`--no-tta` to disable).
* The Mel settings are recovered from the checkpoint / Mel-cache config, so
  inference always matches training.

## Architecture

| stage | output shape (10 s input) | notes |
|---|---|---|
| input | 1 × 128 × 431 | log-Mel, per-crop standardised |
| block 1 (2× conv 3×3 + BN + ReLU, pool 2×2) | 32 × 64 × 215 | |
| block 2 (2× conv, pool 2×2) | 64 × 32 × 107 | |
| block 3 (2× conv, pool **(2, 1)**) | 128 × 16 × 107 | time is **not** pooled → beat-level resolution |
| flatten + Bi-GRU ×2 (256) | 107 × 512 | per-frame embeddings |
| `frame_classifier` | 107 × 256 | local BPM distribution |
| attention pooling → `song_classifier` | 256 | global BPM distribution |
| final BPM | scalar | `0.5 · (frame expectation + song expectation)` |

**5.34 M parameters**; the frequency axis is compressed 8×, which is what allows
the GRU to see a 107-frame sequence for a 10 s excerpt.

---

## Results

<!-- RESULTS:START -->
Test split: **100 held-out GTZAN tracks** (best epoch 54/60, training time 29m23s).

| metric | mean output | frame only | song only |
|---|---|---|---|
| **MAE** | 15.822 | 16.234 | 15.646 |
| **RMSE** | 29.577 | 29.753 | 29.539 |
| **MedianAE** | 4.451 | 4.732 | 4.363 |
| **Bias** | -4.337 | -5.007 | -3.667 |
| **Acc_1BPM** | 17.0% | 10.0% | 21.0% |
| **Acc_1%** | 24.0% | 15.0% | 24.0% |
| **Acc_2%** | 38.0% | 33.0% | 34.0% |
| **Acc_5%** | 55.0% | 53.0% | 57.0% |
| **Acc_octave_1%** | 25.0% | 17.0% | 26.0% |
| **Acc_octave_4%** | 53.0% | 52.0% | 57.0% |
| **Octave_error_rate_4%** | 17.0% | 16.0% | 17.0% |
| **P_score** | 0.485 | 0.462 | 0.495 |
| **Cemgil** | 0.449 | 0.431 | 0.462 |

* octave-tolerant accuracy (within 4 %, factor 1/2 or 2): **53.0%**
* octave-error rate: **17.0%**
* full report: `runs/gtzan/RESULTS.md`, `runs/gtzan/test_metrics.json`, per-track predictions via `python -m src.evaluate`
<!-- RESULTS:END -->

### Interpretation (honest reading of the numbers)

* **Heavy-tailed errors**: the *median* absolute error is **4.45 BPM** while the
  mean is 15.8 BPM. For about half of the tracks the model is within ~4 BPM; the
  rest are hard cases (jazz, classical, metal, country, reggae) where even human
  annotators disagree on "the" tempo.
* **Octave errors are not the dominant failure**: only 17 % of the predictions
  land on half/double tempo (octave assignment: 83 % ×1, 9 % ×0.5, 8 % ×2), so
  the remaining error is genuine tempo confusion on ambiguous material — not the
  classic doubling problem.
* **Genre matters** (see the per-genre table printed by `src.evaluate`): disco
  (MAE 2.8, Acc@5 % 90 %), hiphop (5.5), pop (9.6), rock (10.6) are already
  usable, while classical/jazz/metal/reggae (~22–26 BPM) are dominated by
  annotation ambiguity and by material without a clear percussive pulse.
* There is a small **negative bias (−4.3 BPM)**: the model leans towards the
  slower metrical level, consistent with the synthetic training loops.
* **This is not the "MAE < 2 BPM / Acc@1 % > 80 %" regime**: that requires a
  larger and cleaner corpus (Ballroom / Extended Ballroom, GiantSteps, FMA +
  AcousticBrainz labels) and/or a pretrained front-end — with 798 training tracks
  of 30 s and noisy references, ~5 M parameters cannot do better. The table above
  is what this pipeline actually achieves, measured on a held-out split with no
  audio overlap with training.

### Context study: more audio helps

Same checkpoint, same 100 test tracks, two inference protocols:

| protocol | MAE | MedianAE | Acc@1 % | Acc@5 % | Acc_octave_4 % | octave err | P-score |
|---|---|---|---|---|---|---|---|
| centre 10 s crop (training protocol) | 15.82 | 4.45 | 24 % | 55 % | 53 % | 17 % | 0.485 |
| **full 30 s track** | **14.81** | **3.54** | 23 % | **61 %** | **62 %** | **12 %** | **0.520** |

```bash
python -m src.evaluate --ckpt runs/gtzan/best.pth --crop-seconds 30 \
    --out-dir runs/gtzan_eval_fulltrack     # artifacts in results/gtzan_fulltrack_30s/
```

With `src.infer` (5 overlapping windows + median + ±1 semitone TTA) the
**per-window spread is a usable confidence signal**: 0.7–1.0 BPM on clear 4/4
material (disco, rock, reggae, hiphop — errors < 2 BPM) versus 10–23 BPM on
ambiguous tracks, which are also the ones predicted badly.

### Reproducibility

```bash
bash scripts/prepare_data.sh                       # data (deterministic split, seed 42)
bash scripts/run_training.sh                       # 60 epochs, seed 42
python -m src.evaluate --ckpt runs/gtzan/best.pth  # metrics + plots
python scripts/update_readme_results.py --run runs/gtzan
```

Everything is seeded (`--seed 42`); the split is a deterministic function of the
audio hashes, so the test set is identical across runs.

## Notes, caveats, honest expectations

* **GTZAN is small** (798 training tracks) and its reference annotations carry
  human disagreement, so the absolute MAE here is *not* comparable with papers
  trained on ballroom / GiantSteps / FMA. Treat numbers as "what this model
  achieves with this data", and read them from `runs/gtzan/RESULTS.md`.
* **Octave errors dominate tempo errors.** A model can be musically right and
  still wrong by a factor of 2. Always look at `Acc_octave_4%` and
  `Octave_error_rate_4%` next to `MAE`.
* The **synthetic percussive set** (500 loops/epoch) makes the model better on
  clearly rhythmic material and worse on rubato/ambient music: it is an explicit
  trade-off, controlled by `--synthetic`.
* **MPS runs in fp32** (no AMP on Apple GPUs): ~38 samples/s here. With CUDA +
  AMP expect a several-times speedup; on CPU lower `--batch-size` and raise
  `--max-hours`.
* GTZAN ships **corrupted files** (2 skipped here) and **duplicated audio across
  genres** (14 groups), both handled by the pipeline (skip + grouped split).

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `ENOSPC` / training crashes while saving `best.pth` | disk full — keep ≥1.5 GB free (`df -h`), empty Trash, `~/Library/Caches` |
| `LibsndfileError: Format not recognised` | corrupted GTZAN file (known) — skipped automatically, count reported by `src.dataset` |
| `RuntimeError: ... same audio in multiple splits` | leakage assertion: the split groups tracks by audio hash; re-run `--split-only` |
| `RuntimeError: Trying to resize storage that is not resizable` | old torch/collate path with memmap tensors — tensors are copied and batches are rectangular in this implementation |
| `NotImplementedError` / ops falling back to CPU on Mac | `PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.train ...` |
| run stopped by accident | `EXTRA_ARGS="--resume" bash scripts/run_training.sh` (resumes from `last.pth`) |

## References

* S. Böck, F. Krebs, G. Widmer — *A multi-model approach to beat tracking considering heterogeneous music styles* (madmom / BeatNet background)
* A. P. Klapuri et al. — tempo estimation with comb filters; **McKinney p-score**, **Cemgil** score
* G. Tzanetakis, P. Cook — *Musical genre classification of audio signals* (GTZAN)
* `gtzan_tempo_beat` — tempo/beat annotations used as reference
* B. McFee et al. — `librosa`

## License

MIT — see [`LICENSE`](LICENSE). The datasets (GTZAN audio, annotations) are **not**
included: they are downloaded on demand and remain subject to their own terms.




