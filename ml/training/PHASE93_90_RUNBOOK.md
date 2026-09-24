# Phase 93 direct-piano 90% runbook

## Post-run erratum — diagnostic only

Phase 93 cannot certify the 90% claim. A post-freeze audit found that the first
splitter compared complete MAESTRO credit strings, so `Robert Schumann / Franz
Liszt` was treated as different from `Franz Liszt`. At the individual-person
level, Liszt, Schumann, and Rachmaninoff crossed train/validation, while Liszt,
Schubert, and Bach crossed train/test. The recordings remained disjoint, but the
stated composer-disjoint contract was not true.

The complete three-epoch run is retained as diagnostic evidence only. Every
epoch lowered teacher-forced loss, every epoch failed the decoded-note safety
gate, and the trainer correctly wrote no checkpoint. The immutable Pod log has
SHA-256 `5891b14442601e5682dc6763bc9969479612dcfb4c0aa7f5aaa36e9a629e20f3`.
The Phase 93 sealed files remain unopened.

Phase 94 replaces this split with connected individual-person composer groups,
excludes all 42 previously opened recordings, and creates a new sealed test.

## Scope

This runbook covers **clean or piano-dominant audio to literal piano-note transcription**. It does not certify the harder full-song-to-Pianella arrangement route. Mixing those two tasks would make the score misleading.

The milestone is:

```text
micro exact-pitch + onset F1 at 100 ms >= 0.900
```

on the sealed, composer-disjoint test set. F1 is used because it requires both precision and recall:

```text
precision = matched notes / predicted notes
recall    = matched notes / reference notes
F1        = 2 * precision * recall / (precision + recall)
```

Deleting difficult notes can increase precision but lowers recall. Adding many guesses can increase recall but lowers precision. Neither shortcut can fake a strong F1.

## Frozen data contract

| Role | Songs | Composers | Clips | May update weights? |
|---|---:|---:|---:|---|
| Train | 30 | 24 | 2,667 | Yes |
| Validation | 6 | 6 | 208 | No; model/epoch selection only |
| Sealed test | 8 | 7 | Not materialized | No; one final examination |

- Composer overlap across the three roles: `0`.
- Opened regression songs are diagnostic only and cannot certify 90%.
- The sealed test audio and MIDI remain absent until the checkpoint and thresholds are frozen.
- Every audio clip is mono 16 kHz PCM and five seconds long.
- The training hop is 2.5 seconds; validation uses non-overlapping five-second windows.
- All clips from one song stay in one split.

The opened distributions are comparable rather than trivially easy: training averages 54.09 notes per clip (P50 48, P90 98, P99 132; MIDI 22-105), while validation averages 57.75 (P50 50, P90 101, P99 139; MIDI 21-104). Validation is slightly denser, so a win cannot come from evaluating an easier texture distribution.

Frozen artifact hashes:

```text
selection manifest: 53eb74f9cf990869127adbcf26f01219ff20ec601409536d48978fdb6cf48252
training manifest:  a36865e72e21071be7b623a713570f29116248d3a8a1beb4429ce8cdc7c53da3
validation manifest:c71d9c1e1b21cff28ca3e214c79b61d370fb6cd6c94b7c2d7ef3b817c6bb9f58
```

## Candidate 1: conservative foundation repair

Candidate: `phase93-v001`

- Start from immutable `original`, not v017.
- Train the final transformer block and output head only.
- Learning rate: `1e-7`.
- Epochs: `3`.
- Gradient accumulation: `8`.
- Timing-token weight: `1.35`.
- Note-off-token weight: `1.25`.
- End-of-sequence weight: `1.20`.
- Instrument conditioning: `acoustic_piano`.
- Precision: BF16.

An epoch is eligible only when teacher-forced validation loss improves **and** its decoded multi-song safety panel passes. The trainer keeps the best eligible epoch and writes nothing if all epochs fail.

## Acceptance gates

### Gate 1: package integrity

- Local audit contains exactly 30 training songs and 6 validation songs.
- Only `acoustic_piano` conditioning is present.
- Every remote object has the same byte length as its local source.
- Downloaded remote manifests reproduce the frozen hashes.
- The output checkpoint directory does not already exist.

### Gate 2: in-training safety panel

- Teacher-forced validation loss must beat the original.
- Aggregate decoded F1 regression may not exceed `0.001`.
- Aggregate decoded recall regression may not exceed `0.002`.
- No validation song may lose more than `0.010` F1.

This gate prevents saving an epoch that learns token probabilities but decodes worse music.

### Gate 3: full unseen-composer validation

Decode all 208 validation clips with both `original` and `phase93-v001` using identical preprocessing and acoustic-piano conditioning. Report:

- micro and macro exact-pitch/onset F1 at 50, 100, and 250 ms;
- precision and recall separately;
- all six per-song scores;
- note-count ratio;
- unmatched insertions and deletions;
- duration error, severe cutoffs, and rapid retriggers after clip-boundary stitching.

The candidate must improve 100 ms F1 without hiding a material per-song regression.

### Gate 4: opened regression panel

Run the same candidate on Schubert, Chopin, Rachmaninoff, and Debussy recordings used by the frozen baseline. This answers whether the candidate transfers outside its new validation set. These songs may reject a candidate, but they cannot certify it.

### Gate 5: freeze before test

Freeze:

- checkpoint/config/metadata hashes;
- source commit and worker image;
- model conditioning and preprocessing;
- arranger profile, if an arranger is evaluated separately;
- all pass/fail thresholds;
- blind-listening render recipe and A/B mapping commitment.

Only then may the sealed eight-song files be materialized.

### Gate 6: sealed examination

The headline passes only if sealed micro exact-pitch/onset F1 at 100 ms is at least `0.900`. Also reject the candidate for a serious worst-song, cutoff, retrigger, or blind-listening regression even when aggregate F1 passes.

## One-question-at-a-time diagnosis

| Observation | Meaning | Next controlled experiment |
|---|---|---|
| Loss does not improve | Optimization is ineffective or labels/input are wrong | Audit tokenization and audio/label timebase before changing architecture |
| Loss improves; 50/100/250 ms all flat | Better calibration did not change decoding | Inspect token margins/decoding threshold; do not add arranger rules |
| 250 ms rises; 100 ms stays flat | Notes exist but onsets are late/early | Train or calibrate timing tokens only; slice signed onset residuals |
| Recall rises; precision falls | Candidate over-detects | Analyze density/register false positives and reduce only the responsible branch |
| Precision rises; recall falls | Candidate becomes too conservative | Analyze missing pitches by density/register/dynamic level; recover supported events |
| Onsets improve; cutoffs worsen | Note-off learning is insufficient | Increase note-off supervision or duration examples without altering note-on selection |
| Only dense passages regress | Capacity or label-length pressure | Stratify dense clips from training songs and test one density-balanced curriculum |
| Only soft/low notes regress | Acoustic imbalance | Add licensed, deterministic gain/EQ augmentation to training only |
| One composer dominates the gain | Memorization/style bias | Reject broad claim; expand composer-balanced data |

## Pre-registered follow-up branches

Do not launch all branches. Pick the branch whose prerequisite observation is actually present.

1. **Underfit branch:** if every eligible epoch remains safe and validation loss is still falling but decoded gain is too small, train one new candidate from `original` with a modestly larger learning rate. Keep data and all other settings fixed.
2. **Timing branch:** if 250 ms improves materially more than 100 ms, keep the checkpoint fixed and test a learned, song-held-out onset correction. Naive spectral-flux snapping is already rejected.
3. **Recall branch:** if exact recall is the limiting metric and source evidence contains the missed pitch, train a conservative recovery selector using complete-song leave-one-out evaluation.
4. **Duration branch:** if pitch/onset passes but notes sound chopped, tune note-off/duration behavior separately. Do not trade factual note accuracy for longer sustain.
5. **Data branch:** if gains are composer- or texture-specific, add more properly licensed composers/recordings before increasing model complexity.

Every branch starts from `original` unless the previous candidate has already passed all transfer gates. Changing one principal variable per candidate keeps cause and effect interpretable.

## Production boundary

MAESTRO is CC BY-NC-SA 4.0 and the upstream checkpoint has its own restrictions. Phase 93 is a research proof, not authorization for commercial deployment. A technically successful result must be reproduced on first-party or commercially licensed data and weights before it can serve public paid traffic.

## Current state

`phase93-v001` was rejected after all three epochs failed decoded-note safety.
No candidate checkpoint was written, production remains unchanged, and the
Phase 93 sealed test remains unopened. Continue only with the corrected Phase 94
person-disjoint benchmark.
