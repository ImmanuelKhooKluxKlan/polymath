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
CANARY_AUDIO="/runpod-volume/training/phase-100-wane-canary-v001/wane-opening-10.1896-32.0.wav"
REPO_COMMIT="${POLYMATH_REPO_COMMIT:?Set POLYMATH_REPO_COMMIT to the immutable commit containing this gate}"
BASE_SHA256="a078a3a3dfdf35b2641de676ab9026b3475af6c009e4042a0d704b2cdd38afe0"
TRAIN_SHA256="733bf64496c858fd88d5d92f98aa447dd0234008328501387e9b1a55df31f1d6"
VALIDATION_SHA256="c48aa0b292f4baaa8b63412e2bacd0c56572ed659010289f790a4eff24ccb117"
CANARY_MANIFEST_SHA256="38e26eee5f00dc92a1ae96a84d859486c3e7c2f11a4bf2c449d50e0b4aaad3d0"
CANARY_AUDIO_SHA256="919d37b0481805ef0f5a0e31a5b07c500492160db5d5abe57afb9bfc58b1e3dd"
EVALUATION_PATH="$OUTPUT_DIR/evaluation-unconditioned-phase-100-wane-p050-v001.json"
BASE_TIMELINE_PATH="$OUTPUT_DIR/evaluation-unconditioned-baseline-continuous.json"
CANDIDATE_TIMELINE_PATH="$OUTPUT_DIR/evaluation-unconditioned-candidate-continuous.json"
GATE_RESULT_PATH="$OUTPUT_DIR/route-matched-validation-gate.json"

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

for required in "$BASE_PATH" "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" "$CANARY_MANIFEST" "$CANARY_AUDIO"; do
  test -f "$required"
done
printf '%s  %s\n' "$BASE_SHA256" "$BASE_PATH" | sha256sum --check --strict -
printf '%s  %s\n' "$TRAIN_SHA256" "$TRAIN_MANIFEST" | sha256sum --check --strict -
printf '%s  %s\n' "$VALIDATION_SHA256" "$VALIDATION_MANIFEST" | sha256sum --check --strict -
printf '%s  %s\n' "$CANARY_MANIFEST_SHA256" "$CANARY_MANIFEST" | sha256sum --check --strict -
printf '%s  %s\n' "$CANARY_AUDIO_SHA256" "$CANARY_AUDIO" | sha256sum --check --strict -
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

echo PHASE100_V003_STAGE_CONTINUOUS_SONG_RESCORE
python -m ml.training.rescore_song_timelines \
  --evaluation "$EVALUATION_PATH" \
  --validation-manifest "$VALIDATION_MANIFEST" \
  --comparison-side baseline \
  --out "$BASE_TIMELINE_PATH"
python -m ml.training.rescore_song_timelines \
  --evaluation "$EVALUATION_PATH" \
  --validation-manifest "$VALIDATION_MANIFEST" \
  --comparison-side candidate \
  --out "$CANDIDATE_TIMELINE_PATH"

echo PHASE100_V003_STAGE_FAIL_CLOSED_ROUTE_GATE
python -m ml.training.apply_route_matched_validation_gate \
  --comparison "$EVALUATION_PATH" \
  --baseline-timeline "$BASE_TIMELINE_PATH" \
  --candidate-timeline "$CANDIDATE_TIMELINE_PATH" \
  --training-metadata "$OUTPUT_DIR/training-metadata.json" \
  --gate "ml/training/configs/phase100-v003-unconditioned-validation-gate.json" \
  --out "$GATE_RESULT_PATH" \
  --fail-on-rejection

echo PHASE100_V003_STAGE_WANE_UNCONDITIONED_CANARY
python -m ml.training.transcribe_batch \
  --manifest "$CANARY_MANIFEST" \
  --model "$OUTPUT_DIR/model.safetensors" \
  --model-id phase100-v003-unconditioned \
  --provider "Polymath route-matched research candidate"

echo PHASE100_V003_COMPLETE
