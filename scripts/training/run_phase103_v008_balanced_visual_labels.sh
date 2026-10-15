#!/usr/bin/env bash
set -Eeuo pipefail

export POLYMATH_CANDIDATE_ID="phase103-v008-balanced-visual-labels"
export POLYMATH_DATASET_ID="phase-103-wane-visual-labels-v003"
export POLYMATH_REPO_PATH="/workspace/polymath-phase103-v008"
export POLYMATH_TRAIN_SHA256="e20a0eb3d54a45c438aa70a86e78c5af6fc571eb9cb7d3a7bdb3a0fed39a92d3"
export POLYMATH_GATE_CONFIG="ml/training/configs/phase103-v008-balanced-visual-labels-validation-gate.json"
export POLYMATH_BASE_CANARY_MANIFEST="ml/training/configs/phase103-v008-wane-base-runpod.json"
export POLYMATH_CANDIDATE_CANARY_MANIFEST="ml/training/configs/phase103-v008-wane-candidate-runpod.json"

exec bash "$(dirname "$0")/run_phase103_v007_visual_labels.sh"
