#!/usr/bin/env bash
set -Eeuo pipefail

export POLYMATH_CANDIDATE_ID="phase103-v018-p000-weight050"
export POLYMATH_DATASET_ID="phase-103-wane-p000-weight050-v008"
export POLYMATH_VALIDATION_MANIFEST="/runpod-volume/training/phase-103-wane-visual-labels-v003/prepared-validation.jsonl"
export POLYMATH_REPO_PATH="/workspace/polymath-phase103-v018"
export POLYMATH_TRAIN_SHA256="e15eeaa6c240f6fa2485d6176e4b9bdd74b2bb6694bcf40918f483255272e298"
export POLYMATH_GATE_CONFIG="ml/training/configs/phase103-v018-p000-weight050-validation-gate.json"
export POLYMATH_BASE_CANARY_MANIFEST="ml/training/configs/phase103-v018-wane-base-runpod.json"
export POLYMATH_CANDIDATE_CANARY_MANIFEST="ml/training/configs/phase103-v018-wane-candidate-runpod.json"
export POLYMATH_TRAIN_LAST_LAYERS="1"
export POLYMATH_VALIDATION_INTERVAL_STEPS="5"
export POLYMATH_MAX_OPTIMIZER_STEPS="40"
export POLYMATH_HIGH_NOTE_ON_WEIGHT="1.0"
export POLYMATH_HIGH_NOTE_MIN_MIDI="72"
export POLYMATH_FULL_BASELINE_EVALUATION="/runpod-volume/training/phase-103-wane-visual-labels-v003/baseline-evaluation-phase46-v017-unconditioned.json"
export POLYMATH_FULL_BASELINE_EVALUATION_SHA256="e4e5231d919e749e6cf08e0d772d5a2142fff323cf7e11d2dbf409c50bb0bd54"
export POLYMATH_WANE_BASE_RAW_CACHE="/runpod-volume/training/phase-103-wane-visual-labels-v003/baseline-wane-full-raw-phase46-v017.json"
export POLYMATH_WANE_BASE_RAW_CACHE_SHA256="b7c615b1738390e27889f7ad234e67ade67d981abb9f082c1f5e4f11fb64f6e2"
export POLYMATH_FULL_SONG_TARGET_DEVELOPMENT_CONFIG="ml/training/configs/phase103-v016-full-song-target-development.json"
export POLYMATH_FULL_SONG_TARGET_DEVELOPMENT_CONFIG_SHA256="566d96e074f0f7c003258d127ce3a3893b48f2f4122c3847b37c0af0020e5f42"
export POLYMATH_FULL_SONG_TARGET_DEVELOPMENT_ARTIFACT_DIR="/runpod-volume/training/phase-103-wane-p000-weight050-v008/phase103-v018-p000-weight050-full-song-development"
export POLYMATH_TARGET_DEVELOPMENT_MINIMUM_100MS_F1_IMPROVEMENT="0.002"
export POLYMATH_TARGET_DEVELOPMENT_MINIMUM_100MS_RECALL_IMPROVEMENT="0.0"
export POLYMATH_TARGET_DEVELOPMENT_MAXIMUM_250MS_F1_REGRESSION="0.0"
export POLYMATH_TARGET_DEVELOPMENT_MAXIMUM_250MS_RECALL_REGRESSION="0.0"

printf '%s  %s\n' "cd005750e1434eb31f0d2bb3d0aded01e6df7359e9735adba129274241d60b61" "/runpod-volume/training/phase-103-wane-p000-weight050-v008/variant-weight-ablation-receipt.json" | sha256sum --check --strict -
printf '%s  %s\n' "6fc65acce8e802775eb5f82bb6c1fe7c53626b1ff1719a0929afaaa28ca025aa" "/runpod-volume/training/phase-103-wane-coherent-audio-v006/audio-rebuild-receipt.json" | sha256sum --check --strict -
printf '%s  %s\n' "d558cab2304a4f2c8122945f44cba5c5969e2b1dca4846582005c1c268beee81" "/runpod-volume/training/phase-103-decoder-baseline-audit-v001/audit.json" | sha256sum --check --strict -
exec bash "$(dirname "$0")/run_phase103_v007_visual_labels.sh"
