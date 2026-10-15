#!/usr/bin/env bash
set -Eeuo pipefail

DATASET_DIR="/runpod-volume/training/phase-103-wane-visual-labels-v002"
STATUS_PATH="$DATASET_DIR/phase103-v007-visual-labels-status.txt"
LOG_PATH="$DATASET_DIR/phase103-v007-visual-labels.log"
REJECTION_RECEIPT="$DATASET_DIR/phase103-v007-visual-labels-rejection.json"
REPO_PATH="/workspace/polymath-phase103-v007"
VENV_PATH="/workspace/polymath-phase100-v006-venv"
BASE_PATH="/runpod-volume/models/muscriptor-tester/phase46-v017/model.safetensors"
OUTPUT_DIR="/runpod-volume/models/muscriptor-tester/phase103-v007-visual-labels"
TRAIN_MANIFEST="$DATASET_DIR/prepared-train.jsonl"
VALIDATION_MANIFEST="$DATASET_DIR/prepared-validation.jsonl"
WANE_AUDIO="/runpod-volume/jobs/phase45-foundation-eval-20260916/wane-gbt-original.wav"
REPO_COMMIT="${POLYMATH_REPO_COMMIT:?Set POLYMATH_REPO_COMMIT to the immutable commit containing this experiment}"
BASE_SHA256="a078a3a3dfdf35b2641de676ab9026b3475af6c009e4042a0d704b2cdd38afe0"
TRAIN_SHA256="05cc023be3362dc4c3f55b51a775abe071ce4f16dcfe9f4b76a7547aaf8b0932"
VALIDATION_SHA256="c48aa0b292f4baaa8b63412e2bacd0c56572ed659010289f790a4eff24ccb117"
WANE_AUDIO_SHA256="c1cb84e5b11ae5314c297e96f82469e88922add794d3f9c111122ecb8278b142"
EVALUATION_PATH="$OUTPUT_DIR/evaluation-unconditioned-phase103-v007.json"
BASE_TIMELINE_PATH="$OUTPUT_DIR/evaluation-unconditioned-baseline-continuous.json"
CANDIDATE_TIMELINE_PATH="$OUTPUT_DIR/evaluation-unconditioned-candidate-continuous.json"
GATE_RESULT_PATH="$OUTPUT_DIR/route-matched-validation-gate.json"

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

echo PHASE103_V007_STAGE_CHECKOUT
git clone --filter=blob:none https://github.com/ImmanuelKhooKluxKlan/polymath.git "$REPO_PATH"
cd "$REPO_PATH"
git checkout --detach "$REPO_COMMIT"
test "$(git rev-parse HEAD)" = "$REPO_COMMIT"
PYTHON="$VENV_PATH/bin/python"

echo PHASE103_V007_STAGE_TRAIN_VISUAL_LABELS
"$PYTHON" -m ml.training.train_muscriptor_piano \
  --train-manifest "$TRAIN_MANIFEST" \
  --validation-manifest "$VALIDATION_MANIFEST" \
  --base "$BASE_PATH" \
  --out "$OUTPUT_DIR" \
  --rejection-receipt "$REJECTION_RECEIPT" \
  --execute \
  --rights-acknowledgement I_HAVE_TRAINING_RIGHTS \
  --minimum-train-songs 20 \
  --train-last-layers 1 \
  --epochs 1 \
  --learning-rate 5e-7 \
  --weight-decay 0.01 \
  --gradient-accumulation 8 \
  --gradient-clip-norm 1.0 \
  --timing-token-weight 1.15 \
  --note-on-token-weight 1.0 \
  --note-off-token-weight 1.25 \
  --eos-token-weight 1.20 \
  --conditioning-mode unconditioned \
  --precision bf16 \
  --seed polymath-phase100-v003-unconditioned

echo PHASE103_V007_STAGE_FULL_UNCONDITIONED_EVALUATION
"$PYTHON" - <<'PY'
import json
from pathlib import Path
from ml.training.evaluate_checkpoint import compare_checkpoints, save_comparison

base = Path("/runpod-volume/models/muscriptor-tester/phase46-v017/model.safetensors")
candidate = Path("/runpod-volume/models/muscriptor-tester/phase103-v007-visual-labels/model.safetensors")
manifest = Path("/runpod-volume/training/phase-103-wane-visual-labels-v002/prepared-validation.jsonl")
output = Path("/runpod-volume/models/muscriptor-tester/phase103-v007-visual-labels/evaluation-unconditioned-phase103-v007.json")
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
"$PYTHON" -m ml.training.transcribe_batch \
  --manifest ml/training/configs/phase103-v007-wane-base-runpod.json \
  --model "$BASE_PATH" \
  --model-id phase46-v017 \
  --provider "Polymath immutable experiment base"

echo PHASE103_V007_STAGE_WANE_FULL_CANDIDATE
"$PYTHON" -m ml.training.transcribe_batch \
  --manifest ml/training/configs/phase103-v007-wane-candidate-runpod.json \
  --model "$OUTPUT_DIR/model.safetensors" \
  --model-id phase103-v007-visual-labels \
  --provider "Polymath independent-visual-label experiment"

echo PHASE103_V007_STAGE_FAIL_CLOSED_ROUTE_GATE
"$PYTHON" -m ml.training.apply_route_matched_validation_gate \
  --comparison "$EVALUATION_PATH" \
  --baseline-timeline "$BASE_TIMELINE_PATH" \
  --candidate-timeline "$CANDIDATE_TIMELINE_PATH" \
  --training-metadata "$OUTPUT_DIR/training-metadata.json" \
  --gate ml/training/configs/phase103-v007-visual-labels-validation-gate.json \
  --out "$GATE_RESULT_PATH" \
  --fail-on-rejection

echo PHASE103_V007_COMPLETE
