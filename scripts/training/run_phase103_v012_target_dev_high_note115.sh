#!/usr/bin/env bash
set -Eeuo pipefail

export POLYMATH_CANDIDATE_ID="phase103-v012-target-dev-high-note115"
export POLYMATH_DATASET_ID="phase-103-wane-visual-labels-v004"
export POLYMATH_VALIDATION_MANIFEST="/runpod-volume/training/phase-103-wane-visual-labels-v003/prepared-validation.jsonl"
export POLYMATH_REPO_PATH="/workspace/polymath-phase103-v012"
export POLYMATH_TRAIN_SHA256="99373fac758a9c5cdf7b12bd591bdcc3d98d0ce73fbd91aefe727455b7ddb3fa"
export POLYMATH_GATE_CONFIG="ml/training/configs/phase103-v012-target-dev-high-note115-validation-gate.json"
export POLYMATH_BASE_CANARY_MANIFEST="ml/training/configs/phase103-v012-wane-base-runpod.json"
export POLYMATH_CANDIDATE_CANARY_MANIFEST="ml/training/configs/phase103-v012-wane-candidate-runpod.json"
export POLYMATH_VALIDATION_INTERVAL_STEPS="5"
export POLYMATH_MAX_OPTIMIZER_STEPS="40"
export POLYMATH_HIGH_NOTE_ON_WEIGHT="1.15"
export POLYMATH_HIGH_NOTE_MIN_MIDI="72"

export POLYMATH_FULL_BASELINE_EVALUATION="/runpod-volume/training/phase-103-wane-visual-labels-v003/baseline-evaluation-phase46-v017-unconditioned.json"
export POLYMATH_FULL_BASELINE_EVALUATION_SHA256="e4e5231d919e749e6cf08e0d772d5a2142fff323cf7e11d2dbf409c50bb0bd54"
export POLYMATH_WANE_BASE_RAW_CACHE="/runpod-volume/training/phase-103-wane-visual-labels-v003/baseline-wane-full-raw-phase46-v017.json"
export POLYMATH_WANE_BASE_RAW_CACHE_SHA256="b7c615b1738390e27889f7ad234e67ade67d981abb9f082c1f5e4f11fb64f6e2"

export POLYMATH_FULL_SONG_TARGET_DEVELOPMENT_CONFIG="ml/training/configs/phase103-v012-full-song-target-development.json"
export POLYMATH_FULL_SONG_TARGET_DEVELOPMENT_CONFIG_SHA256="61a7618a8803507f34945c429ea249249623efe1e1b0372d9e9a947f459a877a"
export POLYMATH_FULL_SONG_TARGET_DEVELOPMENT_ARTIFACT_DIR="/runpod-volume/training/phase-103-wane-visual-labels-v004/phase103-v012-target-dev-high-note115-full-song-development"
export POLYMATH_TARGET_DEVELOPMENT_MINIMUM_100MS_F1_IMPROVEMENT="0.002"
export POLYMATH_TARGET_DEVELOPMENT_MINIMUM_100MS_RECALL_IMPROVEMENT="0.0"
export POLYMATH_TARGET_DEVELOPMENT_MAXIMUM_250MS_F1_REGRESSION="0.0"
export POLYMATH_TARGET_DEVELOPMENT_MAXIMUM_250MS_RECALL_REGRESSION="0.0"

exec bash "$(dirname "$0")/run_phase103_v007_visual_labels.sh"
