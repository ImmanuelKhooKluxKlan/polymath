# Polymath piano supervised-learning workspace

This directory prepares trustworthy training examples and includes a conservative
MuScriptor-compatible individual-instrument teacher-forcing adapter. The published package contains
inference architecture and decoding, but not its original trainer or production
data loader, so our adapter is deliberately gated, partial, and candidate-only.
It must not run until the reviewed dataset is large enough.

## Why this system exists

The ideal MIDI and source video are often different clocks:

- the screen recording can start several seconds early or late;
- MIDI can be globally faster/slower;
- a performance can pause, edit, or drift locally;
- the model output contains extra instruments, missing notes, octave errors, and
  duplicate/stutter notes.

The alignment engine uses the model output only to find musical coordinates.
The desired MIDI supplies the piano labels. Labels are warped onto the original
source-video timeline, and nothing outside that timeline is allowed into the
dataset.

## End-to-end flow

1. Run a song through Admin → Piano Model Lab, or upload an existing raw
   MuScriptor MIDI/JSON.
2. Upload the desired piano MIDI/JSON to the Supervised Learning workbench.
3. Inspect the global line, local nonlinear map, 5-second confidence blocks,
   tempo/pause segments, and pitch/timing statistics.
4. Add manual coordinate anchors around real matching moments when the automatic
   map is wrong. Accept or reject every uncertain block.
5. Download the full `polymath-supervision-package-v1` JSON.
6. Create a private training index from `training-index.example.json`. Record
   actual rights/provenance; `allowedForTraining` must be true.
7. Build manifests:

```powershell
python ml/training/dataset_builder.py `
  --index "C:\private-training\training-index.json" `
  --out "C:\private-training\dataset-v001"
```

8. Locate bundled FFmpeg and render the exact 5-second, mono 16 kHz audio:

```powershell
$trainingFfmpeg = node -e "process.stdout.write(require('./server/node_modules/ffmpeg-static'))"
python ml/training/prepare_audio_clips.py `
  --manifest "C:\private-training\dataset-v001\train.jsonl" `
  --out "C:\private-training\dataset-v001" `
  --ffmpeg $trainingFfmpeg
```

Repeat the clip-preparation command for validation and test manifests.

9. Freeze the test split before model tuning. A song belongs to only one split;
   clips from the same song never leak across train/validation/test.
10. Audit the prepared manifests without touching weights:

```powershell
python -m ml.training.train_muscriptor_piano `
  --train-manifest "C:\private-training\dataset-v001\prepared-train.jsonl" `
  --validation-manifest "C:\private-training\dataset-v001\prepared-validation.jsonl" `
  --base "C:\models\original\model.safetensors" `
  --out "C:\models\muscriptor-tester\v002"
```

11. Only on RunPod, after rights and review gates pass, add `--execute` and
    `--rights-acknowledgement I_HAVE_TRAINING_RIGHTS`. The trainer refuses fewer
    than 20 training songs by default, updates only the final transformer block
    plus output head, validates after every epoch, and never overwrites a nonempty
    output directory. Do not overwrite `original/`.
12. Evaluate every candidate against the frozen baseline:

```powershell
python ml/training/evaluate_predictions.py `
  --reference "C:\labels\reviewed-labels.json" `
  --predicted "C:\predictions\candidate-v002.json" `
  --onset-tolerance 0.05
```

The detailed evaluator now reports each instrument independently and separates:

- ignored notes and spurious notes;
- wrong-instrument, octave, near-pitch, and near-timing substitutions;
- repeated-key retriggers within 75 ms;
- severely cut-off and overlong notes;
- onset-only, onset+offset, and 20 ms frame scores;
- complete, partial, and missed chords;
- low, middle, and high pitch-band accuracy;
- the worst five-second error windows.

The dataset builder can include explicitly reviewed zero-note windows as weighted
negative examples. Neutral or rejected windows can never become silence labels.
Training clips may overlap (for example `--training-hop-seconds 2.5`) so sustained
notes are seen at more than one artificial boundary, while validation remains on
the non-overlapping hop. Same-instrument/same-pitch overlaps are normalized using
MuScriptor's public tokenizer rule: the earlier note ends at the next strike.

The loss keeps the mean weight of every clip stable but gives slightly more
importance to timing shifts, note-off events, and EOS. This is an experimental
hypothesis; decoded frozen-song results, not training loss, decide whether a new
checkpoint survives.

## Critical rules

- A green window may train automatically.
- An amber window must be heard/inspected before acceptance.
- A red window must be corrected or rejected.
- A rejected window is not a silent negative example: every 5-second clip that
  overlaps it is excluded entirely.
- Manual acceptance and anchors are written into the package for auditability.
- The source audio/video duration is the hard outer clock.
- Keep `models/original/` immutable. Write candidates to
  `models/muscriptor-tester/v002`, `v003`, and so on.
- Commercial deployment remains blocked until the checkpoint licence is clean.

## Files

- `dataset_builder.py`: validates supervision/rights, splits by song, and creates
  5-second JSONL clip manifests.
- `prepare_audio_clips.py`: uses FFmpeg to render mono 16 kHz WAV clips.
- `evaluate_predictions.py`: instrument-aware pattern/error diagnostics and note metrics.
- `muscriptor_tokens.py`: MT3-like individual-instrument targets, overlap normalization, and ties.
- `train_muscriptor_piano.py`: audit-first partial checkpoint fine-tuning adapter.
- `configs/piano_v001.json`: base revision, data contract and promotion gates.
- `training-index.example.json`: private dataset index template.
- `tests/`: deterministic validation, leakage and metric tests.
