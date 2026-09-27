#!/usr/bin/env bash
#
# Download the dataset and build everything the training needs:
#   1. GTZAN audio (1000 x 30 s clips, ~1.2 GB) from HuggingFace
#   2. tempo/beat annotations (gtzan_tempo_beat, 999 .bpm files)
#   3. metadata + grouped train/val/test split + log-Mel cache (~0.35 GB)
#
# Usage:  bash scripts/prepare_data.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="${DATA_DIR:-data}"
PY="${PY:-python3}"

mkdir -p "$DATA_DIR"

if [ ! -d "$DATA_DIR/genres" ]; then
  echo "== downloading GTZAN (1.2 GB) ..."
  curl -L --retry 3 --retry-delay 5 -C - \
    -o "$DATA_DIR/genres.tar.gz" \
    https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz
  echo "== extracting ..."
  tar xzf "$DATA_DIR/genres.tar.gz" -C "$DATA_DIR"
  rm -f "$DATA_DIR/genres.tar.gz"        # free ~1.2 GB
  echo "== extracted: $(find "$DATA_DIR/genres" -name '*.wav' | wc -l | tr -d ' ') tracks"
else
  echo "== audio already present ($(find "$DATA_DIR/genres" -name '*.wav' | wc -l | tr -d ' ') tracks)"
fi

if [ ! -d "$DATA_DIR/gtzan_tempo_beat" ]; then
  echo "== cloning tempo/beat annotations ..."
  git clone --depth 1 \
    https://github.com/TempoBeatDownbeat/gtzan_tempo_beat.git \
    "$DATA_DIR/gtzan_tempo_beat"
fi

echo "== building metadata + split + log-Mel cache (a few minutes) ..."
"$PY" -m src.dataset \
  --data-root "$DATA_DIR" \
  --cache-dir "$DATA_DIR/cache" \
  --meta "$DATA_DIR/meta.csv" \
  --build

echo "== data ready =="
echo "   metadata : $DATA_DIR/meta.csv"
echo "   mel cache: $DATA_DIR/cache/mels.npy"
