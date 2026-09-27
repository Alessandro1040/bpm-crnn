#!/usr/bin/env bash
#
# Run this AFTER the training has finished, to complete the deliverable:
#   1. full evaluation on the held-out test split (metrics + per-genre table)
#   2. plots in assets/ (scatter, error histogram, training curves)
#   3. metrics injected into README.md (between the RESULTS markers)
#   4. optional commit + push (PUSH=1)
#
# Usage:
#   bash scripts/post_training.sh                # evaluate + plots + README
#   PUSH=1 bash scripts/post_training.sh         # ... and commit/push them
#   RUN=runs/other bash scripts/post_training.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

RUN="${RUN:-runs/gtzan}"
PUSH="${PUSH:-0}"

if [ ! -f "$RUN/TRAINING_COMPLETE" ]; then
  echo "!! $RUN/TRAINING_COMPLETE not found: the run is not finished yet."
  bash scripts/status.sh "$RUN" || true
  exit 1
fi

echo "== run finished, summary =="
cat "$RUN/TRAINING_COMPLETE"

echo
echo "== evaluating $RUN/best.pth on the test split =="
python3 -m src.evaluate --ckpt "$RUN/best.pth" --out-dir "$RUN" --plots-dir assets

echo
echo "== injecting the metrics into README.md =="
python3 scripts/update_readme_results.py --run "$RUN"

if [ "$PUSH" = "1" ]; then
  echo
  echo "== committing and pushing results =="
  git add README.md assets "$RUN/RESULTS.md" 2>/dev/null || git add README.md assets
  git commit -m "results: test metrics and plots for $RUN" || echo "(nothing to commit)"
  git push
  echo "pushed to $(git remote get-url origin)"
fi

echo
echo "== done =="
echo "   per-track predictions: $RUN/predictions_test.csv"
echo "   report:                $RUN/metrics_report_test.json"
echo "   plots:                 assets/"
