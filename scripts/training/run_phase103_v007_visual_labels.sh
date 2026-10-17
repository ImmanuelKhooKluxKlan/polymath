#!/usr/bin/env bash
set -Eeuo pipefail

CANDIDATE_ID="${POLYMATH_CANDIDATE_ID:-phase103-v007-visual-labels}"
DATASET_ID="${POLYMATH_DATASET_ID:-phase-103-wane-visual-labels-v002}"
DATASET_DIR="/runpod-volume/training/$DATASET_ID"
STATUS_PATH="$DATASET_DIR/$CANDIDATE_ID-status.txt"
LOG_PATH="$DATASET_DIR/$CANDIDATE_ID.log"
REJECTION_RECEIPT="$DATASET_DIR/$CANDIDATE_ID-rejection.json"
REPO_PATH="${POLYMATH_REPO_PATH:-/workspace/polymath-phase103-v007}"
VENV_PATH="/workspace/polymath-phase100-v006-venv"
BASE_PATH="/runpod-volume/models/muscriptor-tester/phase46-v017/model.safetensors"
OUTPUT_DIR="/runpod-volume/models/muscriptor-tester/$CANDIDATE_ID"
TRAIN_MANIFEST="$DATASET_DIR/prepared-train.jsonl"
VALIDATION_MANIFEST="${POLYMATH_VALIDATION_MANIFEST:-$DATASET_DIR/prepared-validation.jsonl}"
TARGET_DEVELOPMENT_MANIFEST="${POLYMATH_TARGET_DEVELOPMENT_MANIFEST:-}"
TARGET_DEVELOPMENT_SHA256="${POLYMATH_TARGET_DEVELOPMENT_SHA256:-}"
FULL_SONG_TARGET_DEVELOPMENT_CONFIG="${POLYMATH_FULL_SONG_TARGET_DEVELOPMENT_CONFIG:-}"
FULL_SONG_TARGET_DEVELOPMENT_CONFIG_SHA256="${POLYMATH_FULL_SONG_TARGET_DEVELOPMENT_CONFIG_SHA256:-}"
FULL_SONG_TARGET_DEVELOPMENT_ARTIFACT_DIR="${POLYMATH_FULL_SONG_TARGET_DEVELOPMENT_ARTIFACT_DIR:-}"
BASELINE_TARGET_DEVELOPMENT_EVALUATION="${POLYMATH_BASELINE_TARGET_DEVELOPMENT_EVALUATION:-}"
BASELINE_TARGET_DEVELOPMENT_EVALUATION_SHA256="${POLYMATH_BASELINE_TARGET_DEVELOPMENT_EVALUATION_SHA256:-}"
WANE_AUDIO="/runpod-volume/jobs/phase45-foundation-eval-20260916/wane-gbt-original.wav"
REPO_COMMIT="${POLYMATH_REPO_COMMIT:?Set POLYMATH_REPO_COMMIT to the immutable commit containing this experiment}"
BASE_SHA256="a078a3a3dfdf35b2641de676ab9026b3475af6c009e4042a0d704b2cdd38afe0"
TRAIN_SHA256="${POLYMATH_TRAIN_SHA256:-05cc023be3362dc4c3f55b51a775abe071ce4f16dcfe9f4b76a7547aaf8b0932}"
VALIDATION_SHA256="c48aa0b292f4baaa8b63412e2bacd0c56572ed659010289f790a4eff24ccb117"
WANE_AUDIO_SHA256="c1cb84e5b11ae5314c297e96f82469e88922add794d3f9c111122ecb8278b142"
EVALUATION_PATH="$OUTPUT_DIR/evaluation-unconditioned-$CANDIDATE_ID.json"
BASE_TIMELINE_PATH="$OUTPUT_DIR/evaluation-unconditioned-baseline-continuous.json"
CANDIDATE_TIMELINE_PATH="$OUTPUT_DIR/evaluation-unconditioned-candidate-continuous.json"
GATE_RESULT_PATH="$OUTPUT_DIR/route-matched-validation-gate.json"
FULL_BASELINE_EVALUATION="${POLYMATH_FULL_BASELINE_EVALUATION:-}"
FULL_BASELINE_EVALUATION_SHA256="${POLYMATH_FULL_BASELINE_EVALUATION_SHA256:-}"
CANDIDATE_SINGLE_EVALUATION_PATH="$OUTPUT_DIR/evaluation-unconditioned-$CANDIDATE_ID-single.json"
WANE_BASE_RAW_CACHE="${POLYMATH_WANE_BASE_RAW_CACHE:-}"
WANE_BASE_RAW_CACHE_SHA256="${POLYMATH_WANE_BASE_RAW_CACHE_SHA256:-}"
WANE_BASE_RAW_OUTPUT="$OUTPUT_DIR/wane-full-base-raw.json"
GATE_CONFIG="${POLYMATH_GATE_CONFIG:-ml/training/configs/phase103-v007-visual-labels-validation-gate.json}"
BASE_CANARY_MANIFEST="${POLYMATH_BASE_CANARY_MANIFEST:-ml/training/configs/phase103-v007-wane-base-runpod.json}"
CANDIDATE_CANARY_MANIFEST="${POLYMATH_CANDIDATE_CANARY_MANIFEST:-ml/training/configs/phase103-v007-wane-candidate-runpod.json}"
VALIDATION_INTERVAL_STEPS="${POLYMATH_VALIDATION_INTERVAL_STEPS:-0}"
MAX_OPTIMIZER_STEPS="${POLYMATH_MAX_OPTIMIZER_STEPS:-0}"
HIGH_NOTE_ON_WEIGHT="${POLYMATH_HIGH_NOTE_ON_WEIGHT:-1.0}"
HIGH_NOTE_MIN_MIDI="${POLYMATH_HIGH_NOTE_MIN_MIDI:-60}"
TRAIN_LAST_LAYERS="${POLYMATH_TRAIN_LAST_LAYERS:-1}"

for absent in "$OUTPUT_DIR" "$REJECTION_RECEIPT" "$REJECTION_RECEIPT.sha256" "$STATUS_PATH" "$LOG_PATH" "$REPO_PATH"; do
  test ! -e "$absent"
done
test -x "$VENV_PATH/bin/python"

finish() {
  code=$?
  trap - EXIT
  printf '%s\n' "$code" > "$STATUS_PATH"
  sync
  exit "$code"
}
trap finish EXIT

printf 'RUNNING\n' > "$STATUS_PATH"
exec > >(tee "$LOG_PATH") 2>&1

for required in "$BASE_PATH" "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" "$WANE_AUDIO"; do
  test -f "$required"
done
printf '%s  %s\n' "$BASE_SHA256" "$BASE_PATH" | sha256sum --check --strict -
printf '%s  %s\n' "$TRAIN_SHA256" "$TRAIN_MANIFEST" | sha256sum --check --strict -
printf '%s  %s\n' "$VALIDATION_SHA256" "$VALIDATION_MANIFEST" | sha256sum --check --strict -
printf '%s  %s\n' "$WANE_AUDIO_SHA256" "$WANE_AUDIO" | sha256sum --check --strict -
if [[ -n "$FULL_BASELINE_EVALUATION" ]]; then
  test -n "$FULL_BASELINE_EVALUATION_SHA256"
  test -f "$FULL_BASELINE_EVALUATION"
  printf '%s  %s\n' \
    "$FULL_BASELINE_EVALUATION_SHA256" \
    "$FULL_BASELINE_EVALUATION" | sha256sum --check --strict -
fi
if [[ -n "$WANE_BASE_RAW_CACHE" ]]; then
  test -n "$WANE_BASE_RAW_CACHE_SHA256"
  test -f "$WANE_BASE_RAW_CACHE"
  printf '%s  %s\n' \
    "$WANE_BASE_RAW_CACHE_SHA256" \
    "$WANE_BASE_RAW_CACHE" | sha256sum --check --strict -
fi
if [[ -n "$TARGET_DEVELOPMENT_MANIFEST" ]]; then
  test -n "$TARGET_DEVELOPMENT_SHA256"
  test -f "$TARGET_DEVELOPMENT_MANIFEST"
  printf '%s  %s\n' \
    "$TARGET_DEVELOPMENT_SHA256" \
    "$TARGET_DEVELOPMENT_MANIFEST" | sha256sum --check --strict -
fi
if [[ -n "$FULL_SONG_TARGET_DEVELOPMENT_CONFIG" ]]; then
  test -z "$TARGET_DEVELOPMENT_MANIFEST"
  test -n "$FULL_SONG_TARGET_DEVELOPMENT_CONFIG_SHA256"
  test -n "$FULL_SONG_TARGET_DEVELOPMENT_ARTIFACT_DIR"
  test ! -e "$FULL_SONG_TARGET_DEVELOPMENT_ARTIFACT_DIR"
fi
if [[ -n "$BASELINE_TARGET_DEVELOPMENT_EVALUATION" ]]; then
  test -n "$TARGET_DEVELOPMENT_MANIFEST"
  test -n "$BASELINE_TARGET_DEVELOPMENT_EVALUATION_SHA256"
  test -f "$BASELINE_TARGET_DEVELOPMENT_EVALUATION"
  printf '%s  %s\n' \
    "$BASELINE_TARGET_DEVELOPMENT_EVALUATION_SHA256" \
    "$BASELINE_TARGET_DEVELOPMENT_EVALUATION" | sha256sum --check --strict -
fi

echo PHASE103_V007_STAGE_CHECKOUT
git clone --filter=blob:none https://github.com/ImmanuelKhooKluxKlan/polymath.git "$REPO_PATH"
cd "$REPO_PATH"
git checkout --detach "$REPO_COMMIT"
test "$(git rev-parse HEAD)" = "$REPO_COMMIT"
if [[ -n "$FULL_SONG_TARGET_DEVELOPMENT_CONFIG" ]]; then
  if [[ "$FULL_SONG_TARGET_DEVELOPMENT_CONFIG" != /* ]]; then
    FULL_SONG_TARGET_DEVELOPMENT_CONFIG="$REPO_PATH/$FULL_SONG_TARGET_DEVELOPMENT_CONFIG"
  fi
  test -f "$FULL_SONG_TARGET_DEVELOPMENT_CONFIG"
  printf '%s  %s\n' \
    "$FULL_SONG_TARGET_DEVELOPMENT_CONFIG_SHA256" \
    "$FULL_SONG_TARGET_DEVELOPMENT_CONFIG" | sha256sum --check --strict -
fi
PYTHON="$VENV_PATH/bin/python"
export BASE_PATH OUTPUT_DIR VALIDATION_MANIFEST EVALUATION_PATH CANDIDATE_ID

echo PHASE103_V007_STAGE_TRAIN_VISUAL_LABELS
TRAINING_COMMAND=(
  "$PYTHON" -m ml.training.train_muscriptor_piano \
  --train-manifest "$TRAIN_MANIFEST" \
  --validation-manifest "$VALIDATION_MANIFEST" \
  --base "$BASE_PATH" \
  --out "$OUTPUT_DIR" \
  --candidate-id "$CANDIDATE_ID" \
  --rejection-receipt "$REJECTION_RECEIPT" \
  --execute \
  --rights-acknowledgement I_HAVE_TRAINING_RIGHTS \
  --minimum-train-songs 20 \
  --train-last-layers "$TRAIN_LAST_LAYERS" \
  --epochs 1 \
  --learning-rate 5e-7 \
  --weight-decay 0.01 \
  --gradient-accumulation 8 \
  --gradient-clip-norm 1.0 \
  --validation-interval-steps "$VALIDATION_INTERVAL_STEPS" \
  --max-optimizer-steps "$MAX_OPTIMIZER_STEPS" \
  --timing-token-weight 1.15 \
  --note-on-token-weight 1.0 \
  --high-note-on-weight "$HIGH_NOTE_ON_WEIGHT" \
  --high-note-min-midi "$HIGH_NOTE_MIN_MIDI" \
  --note-off-token-weight 1.25 \
  --eos-token-weight 1.20 \
  --conditioning-mode unconditioned \
  --precision bf16 \
  --seed polymath-phase100-v003-unconditioned
)
if [[ -n "$TARGET_DEVELOPMENT_MANIFEST" ]]; then
  TRAINING_COMMAND+=(
    --target-development-manifest "$TARGET_DEVELOPMENT_MANIFEST"
    --target-development-minimum-100ms-f1-improvement \
      "${POLYMATH_TARGET_DEVELOPMENT_MINIMUM_100MS_F1_IMPROVEMENT:-0.0}"
    --target-development-minimum-100ms-recall-improvement \
      "${POLYMATH_TARGET_DEVELOPMENT_MINIMUM_100MS_RECALL_IMPROVEMENT:-0.0}"
    --target-development-maximum-250ms-f1-regression \
      "${POLYMATH_TARGET_DEVELOPMENT_MAXIMUM_250MS_F1_REGRESSION:-0.0}"
    --target-development-maximum-250ms-recall-regression \
      "${POLYMATH_TARGET_DEVELOPMENT_MAXIMUM_250MS_RECALL_REGRESSION:-0.0}"
  )
fi
if [[ -n "$FULL_SONG_TARGET_DEVELOPMENT_CONFIG" ]]; then
  TRAINING_COMMAND+=(
    --full-song-target-development-config \
      "$FULL_SONG_TARGET_DEVELOPMENT_CONFIG"
    --full-song-target-development-artifact-dir \
      "$FULL_SONG_TARGET_DEVELOPMENT_ARTIFACT_DIR"
    --target-development-minimum-100ms-f1-improvement \
      "${POLYMATH_TARGET_DEVELOPMENT_MINIMUM_100MS_F1_IMPROVEMENT:-0.0}"
    --target-development-minimum-100ms-recall-improvement \
      "${POLYMATH_TARGET_DEVELOPMENT_MINIMUM_100MS_RECALL_IMPROVEMENT:-0.0}"
    --target-development-maximum-250ms-f1-regression \
      "${POLYMATH_TARGET_DEVELOPMENT_MAXIMUM_250MS_F1_REGRESSION:-0.0}"
    --target-development-maximum-250ms-recall-regression \
      "${POLYMATH_TARGET_DEVELOPMENT_MAXIMUM_250MS_RECALL_REGRESSION:-0.0}"
  )
fi
if [[ -n "$FULL_BASELINE_EVALUATION" ]]; then
  TRAINING_COMMAND+=(
    --baseline-decoded-evaluation "$FULL_BASELINE_EVALUATION"
  )
fi
if [[ -n "$BASELINE_TARGET_DEVELOPMENT_EVALUATION" ]]; then
  TRAINING_COMMAND+=(
    --baseline-target-development-evaluation \
      "$BASELINE_TARGET_DEVELOPMENT_EVALUATION"
  )
fi
"${TRAINING_COMMAND[@]}"

echo PHASE103_V007_STAGE_FULL_UNCONDITIONED_EVALUATION
if [[ -n "$FULL_BASELINE_EVALUATION" ]]; then
  echo PHASE103_V007_REUSE_FROZEN_FULL_BASELINE
  "$PYTHON" -m ml.training.evaluate_checkpoint \
    --checkpoint "$OUTPUT_DIR/model.safetensors" \
    --validation-manifest "$VALIDATION_MANIFEST" \
    --unconditioned \
    --out "$CANDIDATE_SINGLE_EVALUATION_PATH"
  "$PYTHON" -m ml.training.compose_checkpoint_comparison \
    --baseline-evaluation "$FULL_BASELINE_EVALUATION" \
    --candidate-evaluation "$CANDIDATE_SINGLE_EVALUATION_PATH" \
    --out "$EVALUATION_PATH"
else
  "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path
from ml.training.evaluate_checkpoint import compare_checkpoints, save_comparison

base = Path(os.environ["BASE_PATH"])
candidate = Path(os.environ["OUTPUT_DIR"]) / "model.safetensors"
manifest = Path(os.environ["VALIDATION_MANIFEST"])
output = Path(os.environ["EVALUATION_PATH"])
result = compare_checkpoints(
    base, candidate, manifest,
    progress_callback=lambda message: print(message, flush=True),
    instruments=None,
)
save_comparison(result, output)
print(json.dumps({
    "output": str(output),
    "clips": result["clips"],
    "instrumentConstraint": result["instrumentConstraint"],
    "candidateMinusBaselineMicroF1": result["candidateMinusBaselineMicroF1"],
}, indent=2), flush=True)
PY
fi

echo PHASE103_V007_STAGE_CONTINUOUS_SONG_RESCORE
"$PYTHON" -m ml.training.rescore_song_timelines \
  --evaluation "$EVALUATION_PATH" \
  --validation-manifest "$VALIDATION_MANIFEST" \
  --comparison-side baseline \
  --out "$BASE_TIMELINE_PATH"
"$PYTHON" -m ml.training.rescore_song_timelines \
  --evaluation "$EVALUATION_PATH" \
  --validation-manifest "$VALIDATION_MANIFEST" \
  --comparison-side candidate \
  --out "$CANDIDATE_TIMELINE_PATH"

echo PHASE103_V007_STAGE_WANE_FULL_BASE
if [[ -n "$WANE_BASE_RAW_CACHE" ]]; then
  echo PHASE103_V007_REUSE_FROZEN_WANE_BASE
  test ! -e "$WANE_BASE_RAW_OUTPUT"
  cp -- "$WANE_BASE_RAW_CACHE" "$WANE_BASE_RAW_OUTPUT"
  printf '%s  %s\n' \
    "$WANE_BASE_RAW_CACHE_SHA256" \
    "$WANE_BASE_RAW_CACHE" > "$OUTPUT_DIR/wane-full-base-raw.reused.sha256"
else
  "$PYTHON" -m ml.training.transcribe_batch \
    --manifest "$BASE_CANARY_MANIFEST" \
    --model "$BASE_PATH" \
    --model-id phase46-v017 \
    --provider "Polymath immutable experiment base"
fi

echo PHASE103_V007_STAGE_WANE_FULL_CANDIDATE
"$PYTHON" -m ml.training.transcribe_batch \
  --manifest "$CANDIDATE_CANARY_MANIFEST" \
  --model "$OUTPUT_DIR/model.safetensors" \
  --model-id "$CANDIDATE_ID" \
  --provider "Polymath independent-visual-label experiment"

echo PHASE103_V007_STAGE_FAIL_CLOSED_ROUTE_GATE
"$PYTHON" -m ml.training.apply_route_matched_validation_gate \
  --comparison "$EVALUATION_PATH" \
  --baseline-timeline "$BASE_TIMELINE_PATH" \
  --candidate-timeline "$CANDIDATE_TIMELINE_PATH" \
  --training-metadata "$OUTPUT_DIR/training-metadata.json" \
  --gate "$GATE_CONFIG" \
  --out "$GATE_RESULT_PATH" \
  --fail-on-rejection

echo PHASE103_STAGE_SOURCE_AWARE_PIANO_PROCESSING_GATE
bash scripts/training/evaluate_phase103_wane_production_route.sh "$CANDIDATE_ID"
"$PYTHON" - "$OUTPUT_DIR/wane-production-route-evaluation/evaluation.json" <<'PY'
import json
import sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text())
gates = report.get("promotionGates")
if report.get("decision") != "PROMOTE" or not isinstance(gates, dict) or not gates or not all(value is True for value in gates.values()):
    raise SystemExit("Source-aware piano-processing checks rejected the candidate; no production promotion is authorized.")
print("Source-aware piano-processing checks passed. Other-song transfer, listening, and rights review remain required.", flush=True)
PY

echo PHASE103_V007_COMPLETE
