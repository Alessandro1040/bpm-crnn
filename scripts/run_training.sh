#!/usr/bin/env bash
#
# Launch a training run in the *background* (nohup): it keeps running after you
# close the terminal / VS Code, and it is fully resumable.
#
# Usage:
#   bash scripts/run_training.sh                 # defaults (runs/gtzan, 60 epochs)
#   OUT=runs/gtzan EPOCHS=40 bash scripts/run_training.sh
#
# Monitoring (see also scripts/status.sh):
#   tail -f runs/gtzan/train.log        # live log
#   cat runs/gtzan/status.json          # progress / ETA / best metrics
#   cat runs/gtzan/TRAINING_COMPLETE    # appears ONLY when the run is finished
#
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${OUT:-runs/gtzan}"
EPOCHS="${EPOCHS:-60}"
BATCH="${BATCH:-32}"
SYNTHETIC="${SYNTHETIC:-500}"
WORKERS="${WORKERS:-4}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$OUT"

if [ -f "$OUT/train.pid" ] && ps -p "$(cat "$OUT/train.pid")" > /dev/null 2>&1; then
  echo "!! a training process is already running (pid $(cat "$OUT/train.pid"))"
  echo "   check: bash scripts/status.sh $OUT"
  exit 1
fi

echo "== launching background training in $OUT =="
# `caffeinate -i` keeps macOS from idle-sleeping while the run is in progress.
LAUNCH_PREFIX=""
if command -v caffeinate > /dev/null 2>&1; then
  LAUNCH_PREFIX="caffeinate -i"
  echo "   power: caffeinate -i (idle sleep disabled for this process)"
fi

nohup $LAUNCH_PREFIX python3 -m src.train \
  --out "$OUT" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --synthetic "$SYNTHETIC" \
  --num-workers "$WORKERS" \
  $EXTRA_ARGS \
  >> "$OUT/nohup.log" 2>&1 &
disown 2> /dev/null || true

echo $! > "$OUT/train.pid"
sleep 2
echo "   pid  : $(cat "$OUT/train.pid")"
echo "   log  : $OUT/train.log   (tail -f)"
echo "   state: $OUT/status.json (cat)"
echo "   done : $OUT/TRAINING_COMPLETE (only when finished)"
echo "   stop : kill \$(cat $OUT/train.pid)   # resumable with --resume"
