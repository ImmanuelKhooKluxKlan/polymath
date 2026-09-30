#!/usr/bin/env bash
set -Eeuo pipefail

STATUS_PATH="/runpod-volume/training/phase-100-wane-p050-v001/phase100-v003-unconditioned-status.txt"
LOG_PATH="/runpod-volume/training/phase-100-wane-p050-v001/phase100-v003-unconditioned.log"
REPO_PATH="/workspace/polymath-phase100-v003"
BASE_PATH="/runpod-volume/models/muscriptor-tester/phase46-v017/model.safetensors"
OUTPUT_DIR="/runpod-volume/models/muscriptor-tester/phase100-v003-unconditioned"
TRAIN_MANIFEST="/runpod-volume/training/phase-100-wane-p050-v001/prepared-train.jsonl"
VALIDATION_MANIFEST="/runpod-volume/training/phase-100-wane-p050-v001/prepared-validation.jsonl"
CANARY_MANIFEST="/runpod-volume/training/phase-100-wane-canary-v001/phase100-v003-unconditioned-manifest.json"
REPO_COMMIT="e7b1b4e324aadf0031afbcecf4a31360fa437990"

finish() {
  code=$?
  trap - EXIT
  printf '%s\n' "$code" > "$STATUS_PATH"
  sync
  exit "$code"
}
trap finish EXIT

printf 'RUNNING\n' > "$STATUS_PATH"
exec > >(tee -a "$LOG_PATH") 2>&1

for required in "$BASE_PATH" "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" "$CANARY_MANIFEST"; do
  test -f "$required"
done
test ! -e "$OUTPUT_DIR"
test ! -e "$REPO_PATH"

echo PHASE100_V003_STAGE_BOOTSTRAP
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  git ffmpeg libsndfile1 ca-certificates
git clone --filter=blob:none https://github.com/ImmanuelKhooKluxKlan/polymath.git "$REPO_PATH"
cd "$REPO_PATH"
git checkout --detach "$REPO_COMMIT"
test "$(git rev-parse HEAD)" = "$REPO_COMMIT"
python -m pip install --no-cache-dir -r serverless/muscriptor/requirements.txt

echo PHASE100_V003_STAGE_TRAIN_UNCONDITIONED
python -m ml.training.train_muscriptor_piano \
  --train-manifest "$TRAIN_MANIFEST" \
  --validation-manifest "$VALIDATION_MANIFEST" \
  --base "$BASE_PATH" \
  --out "$OUTPUT_DIR" \
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

echo PHASE100_V003_STAGE_FULL_UNCONDITIONED_EVALUATION
python - <<'PY'
import json
from pathlib import Path

from ml.training.evaluate_checkpoint import compare_checkpoints, save_comparison

base = Path("/runpod-volume/models/muscriptor-tester/phase46-v017/model.safetensors")
candidate = Path("/runpod-volume/models/muscriptor-tester/phase100-v003-unconditioned/model.safetensors")
manifest = Path("/runpod-volume/training/phase-100-wane-p050-v001/prepared-validation.jsonl")
output = Path("/runpod-volume/models/muscriptor-tester/phase100-v003-unconditioned/evaluation-unconditioned-phase-100-wane-p050-v001.json")

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

echo PHASE100_V003_STAGE_WANE_UNCONDITIONED_CANARY
python -m ml.training.transcribe_batch \
  --manifest "$CANARY_MANIFEST" \
  --model "$OUTPUT_DIR/model.safetensors" \
  --model-id phase100-v003-unconditioned \
  --provider "Polymath route-matched research candidate"

echo PHASE100_V003_COMPLETE
