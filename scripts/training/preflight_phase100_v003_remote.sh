#!/usr/bin/env bash
set -Eeuo pipefail

BASE_PATH="/runpod-volume/models/muscriptor-tester/phase46-v017/model.safetensors"
TRAIN_MANIFEST="/runpod-volume/training/phase-100-wane-p050-v001/prepared-train.jsonl"
VALIDATION_MANIFEST="/runpod-volume/training/phase-100-wane-p050-v001/prepared-validation.jsonl"
CANARY_MANIFEST="/runpod-volume/training/phase-100-wane-canary-v001/phase100-v003-unconditioned-manifest.json"
CANARY_AUDIO="/runpod-volume/training/phase-100-wane-canary-v001/wane-opening-10.1896-32.0.wav"
OUTPUT_DIR="/runpod-volume/models/muscriptor-tester/phase100-v003-unconditioned"

BASE_SHA256="a078a3a3dfdf35b2641de676ab9026b3475af6c009e4042a0d704b2cdd38afe0"
TRAIN_SHA256="733bf64496c858fd88d5d92f98aa447dd0234008328501387e9b1a55df31f1d6"
VALIDATION_SHA256="c48aa0b292f4baaa8b63412e2bacd0c56572ed659010289f790a4eff24ccb117"
CANARY_MANIFEST_SHA256="38e26eee5f00dc92a1ae96a84d859486c3e7c2f11a4bf2c449d50e0b4aaad3d0"
CANARY_AUDIO_SHA256="919d37b0481805ef0f5a0e31a5b07c500492160db5d5abe57afb9bfc58b1e3dd"

echo PHASE100_V003_PREFLIGHT_GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo PHASE100_V003_PREFLIGHT_STORAGE
df -h /runpod-volume /

echo PHASE100_V003_PREFLIGHT_REQUIRED_FILES
for required in "$BASE_PATH" "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" "$CANARY_MANIFEST" "$CANARY_AUDIO"; do
  test -f "$required"
  stat -c '%n %s bytes' "$required"
done

echo PHASE100_V003_PREFLIGHT_HASHES
printf '%s  %s\n' "$BASE_SHA256" "$BASE_PATH" | sha256sum --check --strict -
printf '%s  %s\n' "$TRAIN_SHA256" "$TRAIN_MANIFEST" | sha256sum --check --strict -
printf '%s  %s\n' "$VALIDATION_SHA256" "$VALIDATION_MANIFEST" | sha256sum --check --strict -
printf '%s  %s\n' "$CANARY_MANIFEST_SHA256" "$CANARY_MANIFEST" | sha256sum --check --strict -
printf '%s  %s\n' "$CANARY_AUDIO_SHA256" "$CANARY_AUDIO" | sha256sum --check --strict -

test ! -e "$OUTPUT_DIR"
echo PHASE100_V003_PREFLIGHT_PASS
