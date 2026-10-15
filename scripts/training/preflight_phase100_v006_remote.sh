#!/usr/bin/env bash
set -Eeuo pipefail

BASE_PATH="/runpod-volume/models/muscriptor-tester/phase46-v017/model.safetensors"
TRAIN_MANIFEST="/runpod-volume/training/phase-100-wane-p025-v001/prepared-train.jsonl"
VALIDATION_MANIFEST="/runpod-volume/training/phase-100-wane-p025-v001/prepared-validation.jsonl"
CANARY_MANIFEST="/runpod-volume/training/phase-100-wane-canary-v001/phase100-v006-unconditioned-p025-manifest.json"
CANARY_AUDIO="/runpod-volume/training/phase-100-wane-canary-v001/wane-opening-10.1896-32.0.wav"
OUTPUT_DIR="/runpod-volume/models/muscriptor-tester/phase100-v006-unconditioned-p025"
REJECTION_RECEIPT="/runpod-volume/training/phase-100-wane-p025-v001/phase100-v006-unconditioned-p025-rejection.json"
STATUS_PATH="/runpod-volume/training/phase-100-wane-p025-v001/phase100-v006-unconditioned-p025-status.txt"
LOG_PATH="/runpod-volume/training/phase-100-wane-p025-v001/phase100-v006-unconditioned-p025.log"

BASE_SHA256="a078a3a3dfdf35b2641de676ab9026b3475af6c009e4042a0d704b2cdd38afe0"
TRAIN_SHA256="22348f7df462d603cddb5e1d5a14a69d6b045e863adce2a6e895e844ee3029ea"
VALIDATION_SHA256="c48aa0b292f4baaa8b63412e2bacd0c56572ed659010289f790a4eff24ccb117"
CANARY_MANIFEST_SHA256="ad6cc282011e86c0480e6be0c60f730b1009d349a3010c4290ba9bd166b86437"
CANARY_AUDIO_SHA256="919d37b0481805ef0f5a0e31a5b07c500492160db5d5abe57afb9bfc58b1e3dd"

echo PHASE100_V006_PREFLIGHT_GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo PHASE100_V006_PREFLIGHT_STORAGE
df -h /runpod-volume /

echo PHASE100_V006_PREFLIGHT_REQUIRED_FILES
for required in "$BASE_PATH" "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" "$CANARY_MANIFEST" "$CANARY_AUDIO"; do
  test -f "$required"
  stat -c '%n %s bytes' "$required"
done

echo PHASE100_V006_PREFLIGHT_HASHES
printf '%s  %s\n' "$BASE_SHA256" "$BASE_PATH" | sha256sum --check --strict -
printf '%s  %s\n' "$TRAIN_SHA256" "$TRAIN_MANIFEST" | sha256sum --check --strict -
printf '%s  %s\n' "$VALIDATION_SHA256" "$VALIDATION_MANIFEST" | sha256sum --check --strict -
printf '%s  %s\n' "$CANARY_MANIFEST_SHA256" "$CANARY_MANIFEST" | sha256sum --check --strict -
printf '%s  %s\n' "$CANARY_AUDIO_SHA256" "$CANARY_AUDIO" | sha256sum --check --strict -

for absent in "$OUTPUT_DIR" "$REJECTION_RECEIPT" "$REJECTION_RECEIPT.sha256" "$STATUS_PATH" "$LOG_PATH"; do
  test ! -e "$absent"
done
echo PHASE100_V006_PREFLIGHT_PASS
