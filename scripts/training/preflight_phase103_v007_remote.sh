#!/usr/bin/env bash
set -Eeuo pipefail

BASE_PATH="/runpod-volume/models/muscriptor-tester/phase46-v017/model.safetensors"
TRAIN_MANIFEST="/runpod-volume/training/phase-103-wane-visual-labels-v002/prepared-train.jsonl"
VALIDATION_MANIFEST="/runpod-volume/training/phase-103-wane-visual-labels-v002/prepared-validation.jsonl"
WANE_AUDIO="/runpod-volume/jobs/phase45-foundation-eval-20260916/wane-gbt-original.wav"
OUTPUT_DIR="/runpod-volume/models/muscriptor-tester/phase103-v007-visual-labels"
REJECTION_RECEIPT="/runpod-volume/training/phase-103-wane-visual-labels-v002/phase103-v007-visual-labels-rejection.json"
STATUS_PATH="/runpod-volume/training/phase-103-wane-visual-labels-v002/phase103-v007-visual-labels-status.txt"
LOG_PATH="/runpod-volume/training/phase-103-wane-visual-labels-v002/phase103-v007-visual-labels.log"

BASE_SHA256="a078a3a3dfdf35b2641de676ab9026b3475af6c009e4042a0d704b2cdd38afe0"
TRAIN_SHA256="05cc023be3362dc4c3f55b51a775abe071ce4f16dcfe9f4b76a7547aaf8b0932"
VALIDATION_SHA256="c48aa0b292f4baaa8b63412e2bacd0c56572ed659010289f790a4eff24ccb117"
WANE_AUDIO_SHA256="c1cb84e5b11ae5314c297e96f82469e88922add794d3f9c111122ecb8278b142"

echo PHASE103_V007_PREFLIGHT_GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo PHASE103_V007_PREFLIGHT_STORAGE
df -h /runpod-volume /

echo PHASE103_V007_PREFLIGHT_REQUIRED_FILES
for required in "$BASE_PATH" "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" "$WANE_AUDIO"; do
  test -f "$required"
  stat -c '%n %s bytes' "$required"
done

echo PHASE103_V007_PREFLIGHT_HASHES
printf '%s  %s\n' "$BASE_SHA256" "$BASE_PATH" | sha256sum --check --strict -
printf '%s  %s\n' "$TRAIN_SHA256" "$TRAIN_MANIFEST" | sha256sum --check --strict -
printf '%s  %s\n' "$VALIDATION_SHA256" "$VALIDATION_MANIFEST" | sha256sum --check --strict -
printf '%s  %s\n' "$WANE_AUDIO_SHA256" "$WANE_AUDIO" | sha256sum --check --strict -

for absent in "$OUTPUT_DIR" "$REJECTION_RECEIPT" "$REJECTION_RECEIPT.sha256" "$STATUS_PATH" "$LOG_PATH"; do
  test ! -e "$absent"
done
echo PHASE103_V007_PREFLIGHT_PASS
