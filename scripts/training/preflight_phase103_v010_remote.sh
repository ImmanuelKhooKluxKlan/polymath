#!/usr/bin/env bash
set -Eeuo pipefail

BASE_PATH="/runpod-volume/models/muscriptor-tester/phase46-v017/model.safetensors"
DATASET_DIR="/runpod-volume/training/phase-103-wane-visual-labels-v003"
TRAIN_MANIFEST="$DATASET_DIR/prepared-train.jsonl"
VALIDATION_MANIFEST="$DATASET_DIR/prepared-validation.jsonl"
WANE_AUDIO="/runpod-volume/jobs/phase45-foundation-eval-20260916/wane-gbt-original.wav"
OUTPUT_DIR="/runpod-volume/models/muscriptor-tester/phase103-v010-safe-step020"

BASE_SHA256="a078a3a3dfdf35b2641de676ab9026b3475af6c009e4042a0d704b2cdd38afe0"
TRAIN_SHA256="e20a0eb3d54a45c438aa70a86e78c5af6fc571eb9cb7d3a7bdb3a0fed39a92d3"
VALIDATION_SHA256="c48aa0b292f4baaa8b63412e2bacd0c56572ed659010289f790a4eff24ccb117"
WANE_AUDIO_SHA256="c1cb84e5b11ae5314c297e96f82469e88922add794d3f9c111122ecb8278b142"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for required in "$BASE_PATH" "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" "$WANE_AUDIO"; do
  test -f "$required"
  stat -c '%n %s bytes' "$required"
done
printf '%s  %s\n' "$BASE_SHA256" "$BASE_PATH" | sha256sum --check --strict -
printf '%s  %s\n' "$TRAIN_SHA256" "$TRAIN_MANIFEST" | sha256sum --check --strict -
printf '%s  %s\n' "$VALIDATION_SHA256" "$VALIDATION_MANIFEST" | sha256sum --check --strict -
printf '%s  %s\n' "$WANE_AUDIO_SHA256" "$WANE_AUDIO" | sha256sum --check --strict -
for absent in "$OUTPUT_DIR" "$DATASET_DIR/phase103-v010-safe-step020-rejection.json" "$DATASET_DIR/phase103-v010-safe-step020-status.txt" "$DATASET_DIR/phase103-v010-safe-step020.log"; do
  test ! -e "$absent"
done
echo PHASE103_V010_PREFLIGHT_PASS
