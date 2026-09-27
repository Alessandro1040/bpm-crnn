#!/usr/bin/env bash
#
# One-glance status of a training run.
#
# Usage:  bash scripts/status.sh [run-dir]      (default: runs/gtzan)
#
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-runs/gtzan}"

echo "run dir : $(cd "$OUT" 2>/dev/null && pwd || echo "$OUT (missing)")"

if [ -f "$OUT/train.pid" ]; then
  PID="$(cat "$OUT/train.pid")"
  if ps -p "$PID" > /dev/null 2>&1; then
    echo "process : RUNNING (pid $PID)"
  else
    echo "process : NOT running (pid $PID exited)"
  fi
fi

if [ -f "$OUT/status.json" ]; then
  echo "--- status.json"
  cat "$OUT/status.json"
  echo
fi

if [ -f "$OUT/TRAINING_COMPLETE" ]; then
  echo "--- RESULT: TRAINING COMPLETE (the run finished)"
  cat "$OUT/TRAINING_COMPLETE"
else
  echo "--- RESULT: not finished yet (no TRAINING_COMPLETE marker)"
fi

if [ -f "$OUT/history.csv" ]; then
  echo "--- last epochs (history.csv)"
  head -1 "$OUT/history.csv"
  tail -n 3 "$OUT/history.csv"
fi

if [ -f "$OUT/train.log" ]; then
  echo "--- last log lines"
  tail -n 12 "$OUT/train.log"
fi
