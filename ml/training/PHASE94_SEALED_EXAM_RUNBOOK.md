# Phase 94 single-use sealed exam runbook

## Stop before step 1

Do not download, inspect, parse, list by name, or play any selected test asset
until the owner has submitted an A/B/TIE verdict for the opened Kiss Me pack.
The frozen plan is:

```text
ml/training/configs/phase94-v003-single-use-sealed-exam-v007.json
SHA-256 cb8ec5000cc63c726354f3cdcf8dc28da08aee177492400dfa3c85c05c7801aa
```

Current preflight: eight test songs, four test people, zero person overlap,
zero selected source pairs present, zero prepared test records, and zero labels
read. The test is single-use. A failed result becomes opened development data
and may never be presented as another sealed examination.

## 1. Record the blind choice before revealing the map

Substitute only `A`, `B`, or `TIE` after receiving the owner's answer:

```powershell
python -m ml.training.finalize_overlap_transfer_verdict `
  --freeze ml/training/evidence/phase94-original-overlap-opened-kiss-me-v005/FREEZE.json `
  --pre-listening ml/training/evidence/phase94-original-overlap-opened-kiss-me-v005/PRE_LISTENING.json `
  --structural-freeze ml/training/evidence/phase94-opened-transfer-structural-audit-v006/FREEZE.json `
  --structural-result ml/training/evidence/phase94-opened-transfer-structural-audit-v006/RESULT.json `
  --pack "D:\Polymath Training\phase-94-opened-transfer-kiss-me-v001-2026-09-21\blind-listening" `
  --choice A `
  --reviewer owner `
  --output-dir ml/training/evidence/phase94-opened-kiss-me-blind-verdict-v001
```

Stop permanently if this writes `FAIL`. Do not open the sealed eight-song test.

## 2. Authorize audio only

```powershell
$plan = "ml/training/configs/phase94-v003-single-use-sealed-exam-v007.json"
$custody = "D:\Polymath Training\phase-94-v003-sealed-exam-v007\custody"
$verdict = "ml/training/evidence/phase94-opened-kiss-me-blind-verdict-v001/RESULT.json"

python -m ml.training.sealed_exam_custody --repo-root . begin-audio `
  --plan $plan `
  --blind-result $verdict `
  --custody-dir $custody
```

This creates receipt 01. It must say `audio-only`, `labelsRead: false`, and
`productionPromotionAllowed: false`.

## 3. Download only the selected audio

```powershell
$source = "D:\Polymath Training\phase-94-maestro-individual-composer-curriculum-v003-2026-09-19\source"

python -m ml.training.materialize_sealed_assets --repo-root . `
  --plan $plan `
  --custody-dir $custody `
  --dataset-root $source `
  --kind audio
```

The downloader uses an exact eight-file allow-list. It fails if any selected
MIDI appears. Receipt `01A-AUDIO-MATERIALIZED.json` hashes every audio file.

## 4. Build label-free primary and shifted passes

```powershell
$pack = "D:\Polymath Training\phase-94-v003-sealed-exam-v007\inference-pack"

python -m ml.training.prepare_sealed_audio_passes --repo-root . `
  --plan $plan `
  --custody-dir $custody `
  --dataset-root $source `
  --out-root $pack `
  --remote-root /runpod-volume/training/phase94-v003-sealed-exam-v007
```

Expected properties:

- `split: test`
- `labelsPresent: false`
- `notes: []`
- primary windows at 0, 5, 10, ... seconds
- overlap windows at 2.5, 7.5, 12.5, ... seconds
- identical source-audio SHA-256 on both passes

## 5. Upload and decode exactly once

Upload the complete inference pack to the frozen RunPod network-volume path.
On the research worker, use the original checkpoint whose SHA-256 is:

```text
ac4eb6ea87dfc26b6ca6b954c6b967ab87ad4c7d08e078b25214f13ed051f397
```

Run the frozen evaluator separately for the primary and overlap manifests:

```bash
python -m ml.training.evaluate_checkpoint \
  --checkpoint /workspace/models/original/model.safetensors \
  --validation-manifest /runpod-volume/training/phase94-v003-sealed-exam-v007/primary/manifest.jsonl \
  --out /runpod-volume/training/phase94-v003-sealed-exam-v007/primary-evaluation.json

python -m ml.training.evaluate_checkpoint \
  --checkpoint /workspace/models/original/model.safetensors \
  --validation-manifest /runpod-volume/training/phase94-v003-sealed-exam-v007/overlap/manifest.jsonl \
  --out /runpod-volume/training/phase94-v003-sealed-exam-v007/overlap-evaluation.json
```

Both outputs must include the checkpoint SHA-256. Download the two immutable
evaluations and stop the Pod after verifying file hashes.

## 6. Apply the frozen inference policy

```powershell
python -m ml.training.apply_fixed_overlap_policy `
  --primary-evaluation "D:\Polymath Training\phase-94-v003-sealed-exam-v007\primary-evaluation.json" `
  --overlap-evaluation "D:\Polymath Training\phase-94-v003-sealed-exam-v007\overlap-evaluation.json" `
  --primary-manifest "$pack\primary\manifest.jsonl" `
  --overlap-manifest "$pack\overlap\manifest.jsonl" `
  --radius-seconds 0.6 `
  --onset-match-tolerance-seconds 0.1 `
  --out "D:\Polymath Training\phase-94-v003-sealed-exam-v007\candidate.json"
```

No alternative radius may be generated or compared.

## 7. Lock predictions before labels

```powershell
python -m ml.training.sealed_exam_custody --repo-root . lock-predictions `
  --plan $plan `
  --custody-dir $custody `
  --primary-manifest "$pack\primary\manifest.jsonl" `
  --overlap-manifest "$pack\overlap\manifest.jsonl" `
  --primary-evaluation "D:\Polymath Training\phase-94-v003-sealed-exam-v007\primary-evaluation.json" `
  --overlap-evaluation "D:\Polymath Training\phase-94-v003-sealed-exam-v007\overlap-evaluation.json" `
  --candidate-output "D:\Polymath Training\phase-94-v003-sealed-exam-v007\candidate.json"
```

Receipt 02 is the point of no return. Any later byte change prevents scoring.

## 8. Authorize and fetch MIDI labels

```powershell
python -m ml.training.sealed_exam_custody --repo-root . authorize-labels `
  --plan $plan `
  --custody-dir $custody

python -m ml.training.materialize_sealed_assets --repo-root . `
  --plan $plan `
  --custody-dir $custody `
  --dataset-root $source `
  --kind labels
```

Receipt 03 must predate receipt 03A. The materializer rehashes all locked
predictions immediately before downloading any MIDI.

## 9. Build the identity-preserving reference manifest

```powershell
$reference = "D:\Polymath Training\phase-94-v003-sealed-exam-v007\reference-test.jsonl"
$referenceReceipt = "D:\Polymath Training\phase-94-v003-sealed-exam-v007\reference-receipt.json"

python -m ml.training.prepare_sealed_reference_manifest --repo-root . `
  --plan $plan `
  --custody-dir $custody `
  --primary-manifest "$pack\primary\manifest.jsonl" `
  --out $reference `
  --receipt $referenceReceipt
```

The receipt must report `identityPreserved: true` and eight songs.

## 10. Score baseline and candidate against the same labels

```powershell
$baselineScore = "D:\Polymath Training\phase-94-v003-sealed-exam-v007\baseline-score.json"
$candidateScore = "D:\Polymath Training\phase-94-v003-sealed-exam-v007\candidate-score.json"
$gateResult = "D:\Polymath Training\phase-94-v003-sealed-exam-v007\gate-result.json"

python -m ml.training.rescore_song_timelines `
  --evaluation "D:\Polymath Training\phase-94-v003-sealed-exam-v007\primary-evaluation.json" `
  --validation-manifest $reference `
  --out $baselineScore

python -m ml.training.score_fixed_overlap_inference `
  --prediction "D:\Polymath Training\phase-94-v003-sealed-exam-v007\candidate.json" `
  --reference-manifest $reference `
  --out $candidateScore

python -m ml.training.apply_full_validation_gate `
  --baseline $baselineScore `
  --candidate $candidateScore `
  --gate ml/training/configs/phase94-v003-sealed-test-gate-v001.json `
  --out $gateResult
```

## 11. Consume the test and write the final receipt

```powershell
python -m ml.training.sealed_exam_custody --repo-root . finalize `
  --plan $plan `
  --custody-dir $custody `
  --baseline-score $baselineScore `
  --candidate-score $candidateScore `
  --gate-result $gateResult `
  --reference-receipt $referenceReceipt
```

A research PASS requires all safety checks, candidate F1 above baseline,
100 ms micro-F1 at least 0.900, and the song-bootstrap lower 95% bound at
least 0.900. PASS still does not authorize public production: MAESTRO and the
foundation weights are not cleared for Polymath's commercial use.
