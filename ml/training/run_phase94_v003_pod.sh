#!/usr/bin/env bash
set -euo pipefail

repo=/root/polymath-phase93
python=/root/phase93-venv/bin/python
prior_run=/workspace/training-runs/phase94-v002-note-on-135
run=/workspace/training-runs/phase94-v003-decoded-selection-control
overlap_run=/workspace/training-runs/phase94-overlap-boundary-decoder-v002
output=/root/phase94-v003-decoded-selection-control
dataset=/workspace/training/phase-94-maestro-individual-composer-v001
original=/workspace/models/original/model.safetensors
baseline=/workspace/training-runs/phase94-original-baseline-v001/original-full-validation.json
overlap=/workspace/training/phase94-overlap-validation-v002

prior_pid=$(cat "$prior_run/evaluation.pid")
while kill -0 "$prior_pid" 2>/dev/null; do
  sleep 10
done
test -s "$prior_run/control-vs-candidate-full-validation.json"

test ! -e "$output"
test -s "$dataset/prepared-train.jsonl"
test -s "$dataset/prepared-validation.jsonl"
test -s "$original"
test -s "$baseline"
test -s "$overlap/prepared-validation-overlap.jsonl"
mkdir -p "$run" "$overlap_run"
test ! -e "$run/training-complete.log"
test ! -e "$run/candidate-full-validation.json"
test ! -e "$overlap_run/original-overlap-evaluation.json"

cd "$repo"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run/started-at.txt"
"$python" -u -m ml.training.train_muscriptor_piano \
  --train-manifest "$dataset/prepared-train.jsonl" \
  --validation-manifest "$dataset/prepared-validation.jsonl" \
  --base "$original" \
  --out "$output" \
  --baseline-decoded-evaluation "$baseline" \
  --execute \
  --rights-acknowledgement I_HAVE_TRAINING_RIGHTS \
  --minimum-train-songs 20 \
  --train-last-layers 1 \
  --epochs 1 \
  --learning-rate 1e-7 \
  --weight-decay 0.01 \
  --gradient-accumulation 8 \
  --gradient-clip-norm 1.0 \
  --validation-interval-steps 90 \
  --timing-token-weight 1.35 \
  --note-on-token-weight 1.0 \
  --note-off-token-weight 1.25 \
  --eos-token-weight 1.20 \
  --conditioning-mode instrument \
  --precision bf16 \
  --seed polymath-piano-phase1-v001 \
  --decoded-gate-clips-per-song 6 \
  --decoded-gate-max-aggregate-f1-regression 0.001 \
  --decoded-gate-max-aggregate-recall-regression 0.002 \
  --decoded-gate-max-per-song-f1-regression 0.010 \
  > "$run/training-complete.log" 2>&1

sha256sum \
  "$output/model.safetensors" \
  "$output/training-metadata.json" \
  "$run/training-complete.log" \
  > "$run/training-artifact-sha256.txt"

"$python" -u -m ml.training.evaluate_checkpoint \
  --checkpoint "$output/model.safetensors" \
  --validation-manifest "$dataset/prepared-validation.jsonl" \
  --out "$run/candidate-full-validation.json" \
  > "$run/candidate-full-validation.log" 2>&1
sha256sum \
  "$run/candidate-full-validation.json" \
  "$run/candidate-full-validation.log" \
  > "$run/evaluation-artifact-sha256.txt"

"$python" -u -m ml.training.evaluate_checkpoint \
  --checkpoint "$original" \
  --validation-manifest "$overlap/prepared-validation-overlap.jsonl" \
  --out "$overlap_run/original-overlap-evaluation.json" \
  > "$overlap_run/original-overlap-evaluation.log" 2>&1
sha256sum \
  "$overlap_run/original-overlap-evaluation.json" \
  "$overlap_run/original-overlap-evaluation.log" \
  > "$overlap_run/evaluation-artifact-sha256.txt"

date -u +%Y-%m-%dT%H:%M:%SZ > "$run/completed-at.txt"
