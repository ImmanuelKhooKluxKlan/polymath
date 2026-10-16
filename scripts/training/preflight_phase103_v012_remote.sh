#!/usr/bin/env bash
set -Eeuo pipefail

BASE_PATH="/runpod-volume/models/muscriptor-tester/phase46-v017/model.safetensors"
DATASET_DIR="/runpod-volume/training/phase-103-wane-visual-labels-v004"
TRAIN_MANIFEST="$DATASET_DIR/prepared-train.jsonl"
VALIDATION_MANIFEST="/runpod-volume/training/phase-103-wane-visual-labels-v003/prepared-validation.jsonl"
SPLIT_RECEIPT="$DATASET_DIR/development-split-receipt.json"
WANE_AUDIO="/runpod-volume/jobs/phase45-foundation-eval-20260916/wane-gbt-original.wav"
OUTPUT_DIR="/runpod-volume/models/muscriptor-tester/phase103-v012-target-dev-high-note115"
ARTIFACT_DIR="$DATASET_DIR/phase103-v012-target-dev-high-note115-full-song-development"

BASE_SHA256="a078a3a3dfdf35b2641de676ab9026b3475af6c009e4042a0d704b2cdd38afe0"
TRAIN_SHA256="99373fac758a9c5cdf7b12bd591bdcc3d98d0ce73fbd91aefe727455b7ddb3fa"
VALIDATION_SHA256="c48aa0b292f4baaa8b63412e2bacd0c56572ed659010289f790a4eff24ccb117"
SPLIT_RECEIPT_SHA256="ece35a594d8c5c3c6a123830cbfce0407e7fd0657df97825e70b4db61c5bbed7"
WANE_AUDIO_SHA256="c1cb84e5b11ae5314c297e96f82469e88922add794d3f9c111122ecb8278b142"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for required in "$BASE_PATH" "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" "$SPLIT_RECEIPT" "$WANE_AUDIO"; do
  test -f "$required"
  stat -c '%n %s bytes' "$required"
done
printf '%s  %s\n' "$BASE_SHA256" "$BASE_PATH" | sha256sum --check --strict -
printf '%s  %s\n' "$TRAIN_SHA256" "$TRAIN_MANIFEST" | sha256sum --check --strict -
printf '%s  %s\n' "$VALIDATION_SHA256" "$VALIDATION_MANIFEST" | sha256sum --check --strict -
printf '%s  %s\n' "$SPLIT_RECEIPT_SHA256" "$SPLIT_RECEIPT" | sha256sum --check --strict -
printf '%s  %s\n' "$WANE_AUDIO_SHA256" "$WANE_AUDIO" | sha256sum --check --strict -
for absent in "$OUTPUT_DIR" "$ARTIFACT_DIR" "$DATASET_DIR/phase103-v012-target-dev-high-note115-rejection.json" "$DATASET_DIR/phase103-v012-target-dev-high-note115-status.txt" "$DATASET_DIR/phase103-v012-target-dev-high-note115.log"; do
  test ! -e "$absent"
done
echo PHASE103_V012_PREFLIGHT_PASS
