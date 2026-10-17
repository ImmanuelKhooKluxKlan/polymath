#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
PYTHON="${POLYMATH_PYTHON:-/workspace/polymath-phase100-v006-venv/bin/python}"
test -x "$PYTHON"

CANDIDATE_ID="${1:?Usage: evaluate_phase103_wane_production_route.sh CANDIDATE_ID}"
MODEL_DIR="/runpod-volume/models/muscriptor-tester/$CANDIDATE_ID"
BASE_RAW="$MODEL_DIR/wane-full-base-raw.json"
CANDIDATE_RAW="$MODEL_DIR/wane-full-candidate-raw.json"
EVIDENCE_DIR="/runpod-volume/training/phase-103-wane-visual-labels-v002/evidence"
TARGET="$EVIDENCE_DIR/aligned-training-labels.json"
ALIGNMENT="$EVIDENCE_DIR/alignment-report.json"
SOURCE_AUDIO="/runpod-volume/jobs/phase45-foundation-eval-20260916/wane-gbt-original.wav"
MANIFEST="ml/training/configs/phase103-wane-opened-development-runpod.json"
OUTPUT_DIR="$MODEL_DIR/wane-production-route-evaluation"
BASELINE_DIR="$OUTPUT_DIR/baseline"
CANDIDATE_DIR="$OUTPUT_DIR/candidate"
RESULT="$OUTPUT_DIR/evaluation.json"

TARGET_SHA256="193d1c4a85fc39bb301a35169c261b27909e92e44f63c06241fee5285601d34e"
ALIGNMENT_SHA256="f12d8864c3907f218d11ef5761d89640f609e70aa18f756f5ead6112142c88c6"
SOURCE_AUDIO_SHA256="c1cb84e5b11ae5314c297e96f82469e88922add794d3f9c111122ecb8278b142"

for required in "$BASE_RAW" "$CANDIDATE_RAW" "$TARGET" "$ALIGNMENT" "$MANIFEST" "$SOURCE_AUDIO"; do
  test -f "$required"
done
printf '%s  %s\n' "$TARGET_SHA256" "$TARGET" | sha256sum --check --strict -
printf '%s  %s\n' "$ALIGNMENT_SHA256" "$ALIGNMENT" | sha256sum --check --strict -
printf '%s  %s\n' "$SOURCE_AUDIO_SHA256" "$SOURCE_AUDIO" | sha256sum --check --strict -
test ! -e "$OUTPUT_DIR"
mkdir -p "$BASELINE_DIR" "$CANDIDATE_DIR"

node scripts/training/replayPianoRoute.mjs \
  --input "$BASE_RAW" \
  --output "$OUTPUT_DIR/base-cleaned.json" \
  --mode full --audio "$SOURCE_AUDIO"
"$PYTHON" server/piano_arranger_pipeline.py \
  --input "$OUTPUT_DIR/base-cleaned.json" \
  --output "$BASELINE_DIR/wane-arranged.json" \
  --mode full

node scripts/training/replayPianoRoute.mjs \
  --input "$CANDIDATE_RAW" \
  --output "$OUTPUT_DIR/candidate-cleaned.json" \
  --mode full --audio "$SOURCE_AUDIO"
"$PYTHON" server/piano_arranger_pipeline.py \
  --input "$OUTPUT_DIR/candidate-cleaned.json" \
  --output "$CANDIDATE_DIR/wane.json" \
  --mode full

"$PYTHON" -m ml.training.evaluate_piano_arranger \
  --manifest "$MANIFEST" \
  --baseline-dir "$BASELINE_DIR" \
  --candidate-dir "$CANDIDATE_DIR" \
  --output "$RESULT"

sha256sum \
  "$BASE_RAW" \
  "$CANDIDATE_RAW" \
  "$BASELINE_DIR/wane-arranged.json" \
  "$CANDIDATE_DIR/wane.json" \
  "$RESULT" > "$OUTPUT_DIR/evidence.sha256"

echo PHASE103_WANE_PRODUCTION_ROUTE_EVALUATION_COMPLETE
