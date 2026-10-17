#!/usr/bin/env bash
set -Eeuo pipefail

BASE_PATH="/runpod-volume/models/muscriptor-tester/phase46-v017/model.safetensors"
DATASET_DIR="/runpod-volume/training/phase-103-wane-visual-labels-v005-p000-ablation"
CORE_DIR="/runpod-volume/training/phase-103-wane-visual-labels-v005-core"
TRAIN_MANIFEST="$DATASET_DIR/prepared-train.jsonl"
VALIDATION_MANIFEST="/runpod-volume/training/phase-103-wane-visual-labels-v003/prepared-validation.jsonl"
SPLIT_RECEIPT="$CORE_DIR/development-split-receipt.json"
ABLATION_RECEIPT="$DATASET_DIR/variant-weight-ablation-receipt.json"
WANE_AUDIO="/runpod-volume/jobs/phase45-foundation-eval-20260916/wane-gbt-original.wav"
OUTPUT_DIR="/runpod-volume/models/muscriptor-tester/phase103-v014-p000-ablation-control"
ARTIFACT_DIR="$DATASET_DIR/phase103-v014-p000-ablation-control-full-song-development"

BASE_SHA256="a078a3a3dfdf35b2641de676ab9026b3475af6c009e4042a0d704b2cdd38afe0"
TRAIN_SHA256="1e7f6cb9fe8d5a4f4f8a788af14d7988cc4ded55c6b655da874f89e27259a463"
VALIDATION_SHA256="c48aa0b292f4baaa8b63412e2bacd0c56572ed659010289f790a4eff24ccb117"
SPLIT_RECEIPT_SHA256="c9cb40dd5443c066ef881553a3a2d89ed7cf74b438515150ffd003a9e6d81ff8"
ABLATION_RECEIPT_SHA256="7627622c765973e14e4dc5fa72f8f2e586905d8d650fec23cdb2e1e9d463a0cc"
WANE_AUDIO_SHA256="c1cb84e5b11ae5314c297e96f82469e88922add794d3f9c111122ecb8278b142"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for required in "$BASE_PATH" "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" "$SPLIT_RECEIPT" "$ABLATION_RECEIPT" "$WANE_AUDIO"; do
  test -f "$required"
  stat -c '%n %s bytes' "$required"
done
printf '%s  %s\n' "$BASE_SHA256" "$BASE_PATH" | sha256sum --check --strict -
printf '%s  %s\n' "$TRAIN_SHA256" "$TRAIN_MANIFEST" | sha256sum --check --strict -
printf '%s  %s\n' "$VALIDATION_SHA256" "$VALIDATION_MANIFEST" | sha256sum --check --strict -
printf '%s  %s\n' "$SPLIT_RECEIPT_SHA256" "$SPLIT_RECEIPT" | sha256sum --check --strict -
printf '%s  %s\n' "$ABLATION_RECEIPT_SHA256" "$ABLATION_RECEIPT" | sha256sum --check --strict -
printf '%s  %s\n' "$WANE_AUDIO_SHA256" "$WANE_AUDIO" | sha256sum --check --strict -
for absent in "$OUTPUT_DIR" "$ARTIFACT_DIR" "$DATASET_DIR/phase103-v014-p000-ablation-control-rejection.json" "$DATASET_DIR/phase103-v014-p000-ablation-control-status.txt" "$DATASET_DIR/phase103-v014-p000-ablation-control.log" "/workspace/polymath-phase103-v014"; do
  test ! -e "$absent"
done
echo PHASE103_V014_PREFLIGHT_PASS
