#!/usr/bin/env bash
set -Eeuo pipefail

STATUS_PATH="/runpod-volume/training/phase-100-wane-p025-v001/phase100-v006-unconditioned-p025-status.txt"
LOG_PATH="/runpod-volume/training/phase-100-wane-p025-v001/phase100-v006-unconditioned-p025.log"
REJECTION_RECEIPT="/runpod-volume/training/phase-100-wane-p025-v001/phase100-v006-unconditioned-p025-rejection.json"
REPO_PATH="/workspace/polymath-phase100-v006"
VENV_PATH="/workspace/polymath-phase100-v006-venv"
BASE_PATH="/runpod-volume/models/muscriptor-tester/phase46-v017/model.safetensors"
OUTPUT_DIR="/runpod-volume/models/muscriptor-tester/phase100-v006-unconditioned-p025"
TRAIN_MANIFEST="/runpod-volume/training/phase-100-wane-p025-v001/prepared-train.jsonl"
VALIDATION_MANIFEST="/runpod-volume/training/phase-100-wane-p025-v001/prepared-validation.jsonl"
CANARY_MANIFEST="/runpod-volume/training/phase-100-wane-canary-v001/phase100-v006-unconditioned-p025-manifest.json"
CANARY_AUDIO="/runpod-volume/training/phase-100-wane-canary-v001/wane-opening-10.1896-32.0.wav"
REPO_COMMIT="${POLYMATH_REPO_COMMIT:?Set POLYMATH_REPO_COMMIT to the immutable commit containing this experiment}"
BASE_SHA256="a078a3a3dfdf35b2641de676ab9026b3475af6c009e4042a0d704b2cdd38afe0"
TRAIN_SHA256="22348f7df462d603cddb5e1d5a14a69d6b045e863adce2a6e895e844ee3029ea"
VALIDATION_SHA256="c48aa0b292f4baaa8b63412e2bacd0c56572ed659010289f790a4eff24ccb117"
CANARY_MANIFEST_SHA256="ad6cc282011e86c0480e6be0c60f730b1009d349a3010c4290ba9bd166b86437"
CANARY_AUDIO_SHA256="919d37b0481805ef0f5a0e31a5b07c500492160db5d5abe57afb9bfc58b1e3dd"
EVALUATION_PATH="$OUTPUT_DIR/evaluation-unconditioned-phase-100-wane-p025-v001.json"
BASE_TIMELINE_PATH="$OUTPUT_DIR/evaluation-unconditioned-baseline-continuous.json"
CANDIDATE_TIMELINE_PATH="$OUTPUT_DIR/evaluation-unconditioned-candidate-continuous.json"
GATE_RESULT_PATH="$OUTPUT_DIR/route-matched-validation-gate.json"

for absent in "$OUTPUT_DIR" "$REJECTION_RECEIPT" "$REJECTION_RECEIPT.sha256" "$STATUS_PATH" "$LOG_PATH" "$REPO_PATH" "$VENV_PATH"; do
  test ! -e "$absent"
done

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

for required in "$BASE_PATH" "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" "$CANARY_MANIFEST" "$CANARY_AUDIO"; do
  test -f "$required"
done
printf '%s  %s\n' "$BASE_SHA256" "$BASE_PATH" | sha256sum --check --strict -
printf '%s  %s\n' "$TRAIN_SHA256" "$TRAIN_MANIFEST" | sha256sum --check --strict -
printf '%s  %s\n' "$VALIDATION_SHA256" "$VALIDATION_MANIFEST" | sha256sum --check --strict -
printf '%s  %s\n' "$CANARY_MANIFEST_SHA256" "$CANARY_MANIFEST" | sha256sum --check --strict -
printf '%s  %s\n' "$CANARY_AUDIO_SHA256" "$CANARY_AUDIO" | sha256sum --check --strict -

echo PHASE100_V006_STAGE_BOOTSTRAP
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  git ffmpeg libsndfile1 ca-certificates python3-venv
git clone --filter=blob:none https://github.com/ImmanuelKhooKluxKlan/polymath.git "$REPO_PATH"
cd "$REPO_PATH"
git checkout --detach "$REPO_COMMIT"
test "$(git rev-parse HEAD)" = "$REPO_COMMIT"
python -m venv --system-site-packages "$VENV_PATH"
PYTHON="$VENV_PATH/bin/python"
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install --no-cache-dir -r serverless/muscriptor/requirements.txt

echo PHASE100_V006_STAGE_TRAIN_UNCONDITIONED_P025
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

echo PHASE100_V006_STAGE_FULL_UNCONDITIONED_EVALUATION
"$PYTHON" - <<'PY'
import json
from pathlib import Path

from ml.training.evaluate_checkpoint import compare_checkpoints, save_comparison

base = Path("/runpod-volume/models/muscriptor-tester/phase46-v017/model.safetensors")
candidate = Path("/runpod-volume/models/muscriptor-tester/phase100-v006-unconditioned-p025/model.safetensors")
manifest = Path("/runpod-volume/training/phase-100-wane-p025-v001/prepared-validation.jsonl")
output = Path("/runpod-volume/models/muscriptor-tester/phase100-v006-unconditioned-p025/evaluation-unconditioned-phase-100-wane-p025-v001.json")

result = compare_checkpoints(
    base,
    candidate,
    manifest,
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

echo PHASE100_V006_STAGE_CONTINUOUS_SONG_RESCORE
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

echo PHASE100_V006_STAGE_FAIL_CLOSED_ROUTE_GATE
"$PYTHON" -m ml.training.apply_route_matched_validation_gate \
  --comparison "$EVALUATION_PATH" \
  --baseline-timeline "$BASE_TIMELINE_PATH" \
  --candidate-timeline "$CANDIDATE_TIMELINE_PATH" \
  --training-metadata "$OUTPUT_DIR/training-metadata.json" \
  --gate "ml/training/configs/phase100-v006-unconditioned-p025-validation-gate.json" \
  --out "$GATE_RESULT_PATH" \
  --fail-on-rejection

echo PHASE100_V006_STAGE_WANE_UNCONDITIONED_CANARY
"$PYTHON" -m ml.training.transcribe_batch \
  --manifest "$CANARY_MANIFEST" \
  --model "$OUTPUT_DIR/model.safetensors" \
  --model-id phase100-v006-unconditioned-p025 \
  --provider "Polymath route-matched p025 curriculum experiment"

echo PHASE100_V006_COMPLETE
