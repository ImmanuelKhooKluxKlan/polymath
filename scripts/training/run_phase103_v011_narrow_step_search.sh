#!/usr/bin/env bash
set -Eeuo pipefail

export POLYMATH_CANDIDATE_ID="phase103-v011-narrow-step-search"
export POLYMATH_DATASET_ID="phase-103-wane-visual-labels-v003"
export POLYMATH_REPO_PATH="/workspace/polymath-phase103-v011"
export POLYMATH_TRAIN_SHA256="e20a0eb3d54a45c438aa70a86e78c5af6fc571eb9cb7d3a7bdb3a0fed39a92d3"
export POLYMATH_GATE_CONFIG="ml/training/configs/phase103-v011-narrow-step-search-validation-gate.json"
export POLYMATH_BASE_CANARY_MANIFEST="ml/training/configs/phase103-v011-wane-base-runpod.json"
export POLYMATH_CANDIDATE_CANARY_MANIFEST="ml/training/configs/phase103-v011-wane-candidate-runpod.json"
export POLYMATH_VALIDATION_INTERVAL_STEPS="5"
export POLYMATH_MAX_OPTIMIZER_STEPS="40"

exec bash "$(dirname "$0")/run_phase103_v007_visual_labels.sh"
