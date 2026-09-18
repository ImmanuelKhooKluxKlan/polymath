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

When two full performances have measured durations that rule out extreme tempo
ratios, lock those clock bounds before opening any candidate output. This keeps
repeated pop-song phrases from winning an implausible RANSAC line:

```powershell
npm run align:notes -- `
  --reference "C:\private-training\desired.json" `
  --observed "C:\private-training\raw-original.json" `
  --source-duration 238.56 `
  --minimum-scale 0.92 `
  --maximum-scale 1.12 `
  --ransac-iterations 10000 `
  --out "C:\private-training\alignment-frozen"
```

The bounds constrain only the reference-to-source clock. Choose them from
duration and provenance evidence—not from whichever values improve a candidate
score—and freeze the resulting anchors before baseline/candidate evaluation.

## End-to-end flow

For exact synthetic multitrack supervision, import the official CC BY 4.0
BabySlakh prototype before the ordinary dataset-builder step. Per-stem MIDI is
the same MIDI used to synthesize each stem, so the importer marks its fixed
five-second windows as trusted without consulting model predictions. Duplicate
composition identities are removed before song-level splits. The arranger
teacher mode produces a route-specific, playable piano reduction; it does not
replace a final untouched real-song listening test.

```powershell
python -m ml.training.import_slakh `
  --dataset-root "C:\private-training\BabySlakh" `
  --output-root "C:\private-training\babyslakh-piano-v001" `
  --target-mode arranger-teacher `
  --profile "server\models\piano-arranger\pianella-supervised-v006.json" `
  --train-count 16 `
  --validation-count 2
```

The importer writes `training-index.json`, factual source scores, frozen piano
targets, exact supervision packages, hashes, licence attribution, and an import
summary. Pass that index to `dataset_builder.py` exactly like reviewed human
data. Keep at least one composition completely outside both fitting and model
selection.

1. Run a song through Admin → Piano Model Lab, or upload an existing raw
   MuScriptor MIDI/JSON.
2. Upload the desired piano MIDI/JSON to the Supervised Learning workbench.
3. Inspect the global line, local nonlinear map, 5-second confidence blocks,
   tempo/pause segments, and pitch/timing statistics.
4. Add manual coordinate anchors around real matching moments when the automatic
   map is wrong. Accept or reject every uncertain block.
5. Download the full `polymath-supervision-package-v1` JSON.
6. Before building a dataset, create a new reviewed copy with the deterministic
   local quality gate. The command never overwrites the alignment source:

```powershell
npm run review:alignment -- `
  --input "C:\private-training\aligned-training-labels.json" `
  --output "C:\private-training\reviewed-labels.json"
```

The gate checks local note support, exact pitch, median and tail timing error,
tempo warp, and structural similarity. Manual accept/reject decisions remain
authoritative and are recorded in the output package. A globally plausible
alignment is not enough: only locally accepted windows can reach gradients.
7. Create a private training index from `training-index.example.json`. Record
   actual rights/provenance; `allowedForTraining` must be true.
8. Build manifests:

```powershell
python ml/training/dataset_builder.py `
  --index "C:\private-training\training-index.json" `
  --out "C:\private-training\dataset-v001"
```

9. Locate bundled FFmpeg and render the exact 5-second, mono 16 kHz audio. When
   the files will be uploaded to RunPod, write Linux volume paths into the
   manifests at preparation time:

```powershell
$trainingFfmpeg = node -e "process.stdout.write(require('./server/node_modules/ffmpeg-static'))"
python ml/training/prepare_audio_clips.py `
  --manifest "C:\private-training\dataset-v001\train.jsonl" `
  --out "C:\private-training\dataset-v001" `
  --ffmpeg $trainingFfmpeg `
  --manifest-audio-root "/runpod-volume/training/phase-2-v001"
```

Repeat the clip-preparation command for validation and test manifests.

10. Freeze the test split before model tuning. A song belongs to only one split;
   clips from the same song never leak across train/validation/test.
11. Audit the prepared manifests without touching weights:

```powershell
python -m ml.training.train_muscriptor_piano `
  --train-manifest "C:\private-training\dataset-v001\prepared-train.jsonl" `
  --validation-manifest "C:\private-training\dataset-v001\prepared-validation.jsonl" `
  --base "C:\models\original\model.safetensors" `
  --out "C:\models\muscriptor-tester\v002"
```

12. Only on RunPod, after rights and review gates pass, add `--execute` and
    `--rights-acknowledgement I_HAVE_TRAINING_RIGHTS`. The trainer refuses fewer
    than 20 training songs by default, updates only the final transformer block
    plus output head, validates after every epoch, and never overwrites a nonempty
    output directory. Do not overwrite `original/`.
13. Evaluate every candidate against the frozen baseline:

```powershell
python -m ml.training.evaluate_predictions `
  --reference "C:\labels\reviewed-labels.json" `
  --predicted "C:\predictions\candidate-v002.json" `
  --onset-tolerance 0.05
```

The detailed evaluator now reports each instrument independently and separates:

- ignored notes and spurious notes;
- wrong-instrument, octave, near-pitch, and near-timing substitutions;

### Match training and inference conditioning

The normal individual-instrument experiments use MuScriptor's instrument-group
condition and must be evaluated with the same hard constraint.  A route-specific
full-mix-to-piano experiment is different: production listens without an
instrument constraint before arranging the result, so both training and
checkpoint evaluation must use `--conditioning-mode unconditioned`.  Mixing
those two modes can lower teacher-forcing loss while catastrophically changing
the instruments emitted on a real song; the mixed-song canary remains mandatory.
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

## Route-specific arranger research

Foundation-model loss and Piano-route quality are separate experiments. A
candidate foundation checkpoint must first decode a real mixed-song canary; a
lower teacher-forcing loss is not proof that it transcribes or arranges better.
The Piano route can instead learn a small auditable selector/duration profile
while leaving the instrument-aware listener unchanged.

When an authored MIDI has no matching audio file, synthesize a neutral
alignment-only reference. This WAV is a clock/chroma aid, not a production piano
render and not new training content:

```powershell
python -m ml.training.synthesize_note_reference `
  --notes "C:\private-training\desired.json" `
  --output "C:\private-training\desired-alignment-reference.wav"
```

Train the arranger only from trusted alignment windows. Then evaluate it with
leave-one-song-out folds. If a residual profile is used, the base profile must
also exclude the graded song; otherwise the fold is contaminated even when the
new residual excluded it:

```powershell
python ml/training/cross_validate_piano_arranger_adapter.py `
  --manifest "C:\private-training\arranger-training.json" `
  --cleaned-source-dir "C:\private-training\sources" `
  --baseline-dir "C:\private-training\fold-baselines" `
  --output-dir "C:\private-training\loso-results" `
  --residual-base-profile-dir "C:\private-training\fold-base-profiles" `
  --residual-base-share 0.95 `
  --adaptive-policy-profile "C:\private-training\candidate-policy.json" `
  --expand-sparse-harmony
```

The cross-validator pools weighted cutoff events over weighted matched notes.
This prevents a song with only a handful of baseline matches from dominating a
rate, while still retaining per-song metrics for inspection. Register shift,
confidence/backfill, source-duration blending, minimum hold, and factual
source-density controls are explicit research options; never tune them on the
sealed test split.

Pitch and hold quality are deliberately scored with different match rules.
Exact-register F1 still requires the exact MIDI pitch. Duration, visual-hold,
and physical-hold errors use a pitch-class plus onset match at 250 ms so that a
wrong-octave prediction cannot disappear from the cutoff statistics. Evaluation
reports record this as `durationMatchingPolicy: pitch-class-onset-250ms`.

The optional learned register table is research-only. Do not pool authored MIDI
and a performed Pianella transcription without recording their target-register
convention: the same musically correct pitch class may be written an octave
apart. A four-song experiment learned this annotation difference instead of a
general musical rule and catastrophically failed its next full-mix and synthetic
checks. Until substantially more style-consistent songs exist, keep the frozen
whole-score Pianella register policy and treat register learning as an ablation,
not a deployable feature.

When a trusted target intentionally retains a singer's written octave while the
frozen piano route applies its established whole-score octave lift, declare that
register convention in the evaluation manifest before candidate inference:

```json
{
  "id": "untouched-song",
  "target": "C:/private-training/untouched-song-target.json",
  "alignmentReport": "C:/private-training/untouched-song-alignment.json",
  "referenceTransposeSemitones": 12
}
```

Only octave multiples from -48 through 48 are accepted. This is metadata about
the reference convention, not a per-song pitch search, and the setting is
recorded in the evaluation report.

Residual selectors are blended in raw logit space because independently
standardized weights cannot be averaged directly:

```powershell
python -m ml.training.blend_piano_arranger_profiles `
  --base "C:\models\pianella-supervised-v006.json" `
  --other "C:\models\pianella-supervised-v010.json" `
  --base-share 0.95 `
  --profile-id pianella-residual-candidate `
  --output "C:\models\pianella-residual-candidate.json"
```

An optional adaptive residual policy may use factual source density to retain
more of the frozen selector on a sparse arrangement and admit more of the
learned correction on a dense vocal mix. Gate the aggressive endpoint on a
minimum detected-voice ratio so dense instrumental material does not drift:

```powershell
python -m ml.training.blend_piano_arranger_profiles `
  --base "C:\models\pianella-supervised-v006.json" `
  --other "C:\models\pianella-supervised-v010.json" `
  --base-share 0.925 `
  --profile-id pianella-adaptive-candidate `
  --output "C:\models\pianella-adaptive-candidate.json" `
  --adaptive-selection-low-source-nps 20 `
  --adaptive-selection-high-source-nps 30 `
  --adaptive-selection-low-base-share 0.95 `
  --adaptive-selection-high-base-share 0.80 `
  --adaptive-selection-minimum-voice-ratio 0.05
```

The policy interpolates selector logits, not final notes. Its diagnostics record
the factual density, voice ratio, effective frozen share, and whether the voice
gate kept the conservative endpoint. Thresholds must be frozen before a new
holdout is opened; this mechanism is not permission to recognize song names or
choose settings after seeing the reference.

The same factual gate can protect piano-rich or instrumental material while a
singer-led, piano-light mix uses cleaner octave expansion and steadier holds.
Supply all six source-density controls, then pair the voice/piano boundary with
the duration and sparse-expansion controls. For the frozen v031 research policy,
the important additions are:

```powershell
  --adaptive-minimum-voice-ratio-density 0.05 `
  --adaptive-non-vocal-density 1.45 `
  --adaptive-maximum-piano-ratio-vocal 0.05 `
  --adaptive-minimum-voice-ratio-sparse-disable 0.05 `
  --adaptive-minimum-voice-ratio-duration 0.05 `
  --adaptive-non-vocal-duration-weight 0 `
  --adaptive-short-source-duration-weight 0.92 `
  --adaptive-long-source-duration-weight 0.92 `
  --adaptive-non-vocal-source-duration-weight 1
```

The maximum-piano threshold applies to density, selection, sparse expansion,
and duration routing. Thus a vocal+piano performance keeps the conservative
piano policy instead of being mistaken for a singer over non-piano backing.

Manifests may declare `auxiliaryPairs` that are always included in training but
never graded. Pseudo-label pairs must explicitly set `pseudoLabel: true` and may
supervise selection only. `alignmentMode: source-index` uses hard labels from a
shared source index; `alignmentMode: teacher-profile` distills the frozen
teacher's probabilities as soft labels. Neither is independent accuracy
evidence, and an auxiliary track cannot later be described as untouched test
data. Hard and soft teacher regularization both remain ablations unless they
improve real-song folds and all stability gates.

After automated gates pass, export baseline/candidate JSON through the same
browser duration and velocity contract for a blind same-player listening test:

```powershell
npm run export:piano-listening -- `
  --input "C:\private-training\candidate.json" `
  --out "C:\private-training\listening" `
  --label B
```

Automated promotion is not deployment. A candidate stays opt-in until blind
listening passes, a genuinely untouched real-song holdout survives, licensing
permits its intended use, and the production profile path is changed explicitly.

### Conservative raw-note recovery

The arranger may omit a real low accompaniment tone even when the source
transcription and the leave-one-song-out selector strongly support it. The
research recovery pass can complete an existing gesture without rebuilding the
approved arrangement:

```powershell
python -m ml.training.apply_raw_support_recovery `
  --candidate "D:\research\phase80-candidate.json" `
  --source "D:\research\raw-transcription.json" `
  --profile "D:\research\held-out-selector\profile.json" `
  --output "D:\research\phase82-candidate.json" `
  --threshold 0.90 `
  --coverage-radius-seconds 0.15 `
  --maximum-additions-per-onset 1 `
  --gesture-anchor-radius-seconds 0.18 `
  --maximum-piano-source-midi 52 `
  --register-shift-semitones 0
```

Keep the native register. The old `+12` recovery experiment found correct pitch
classes but placed them in the wrong octave. The gesture anchor and one-note cap
prevent new rhythmic attacks and dense stem restoration. Piano-labelled support
above E3 is excluded because it caused repeated false chord tones on an
independent piano-rich safety song. The frozen Phase-82 policy remains
research-only until its sealed listening comparison is completed.

### Adaptive melody/harmony register separation

A correct pitch class can still sound wrong when the melody and accompaniment
occupy nearly the same octave. The Phase-84 research operator measures the
whole-song median MIDI for notes already labelled `melody` and `harmony`. It
raises melody by one octave only when both roles are stable and their median
gap is at most seven semitones:

```powershell
python -m ml.training.apply_adaptive_pianist_register_zones `
  --input "D:\research\phase82-safe.json" `
  --profile "ml\training\configs\phase84-adaptive-melody-separation-v002.json" `
  --output "D:\research\phase84-candidate.json" `
  --report "D:\research\phase84-register-report.json"
```

This decision is inference-safe: it uses no title, song identity, reference,
or alignment. It preserves every onset, duration, velocity, and pitch class.
On five complete development songs it improved weighted exact-note F1@250 ms
by `0.038086` with no song regression. On the separate A Thousand Years
transfer it improved exact-note F1@250 ms by `0.074786`. Songs whose melody was
already separated by 10-14.5 semitones correctly received no edit. The frozen
candidate then won its committed, sealed same-player Kiss Me comparison: the
reviewer chose `B` before reveal, and `B` resolved to candidate `v002`. This is
still a research result, not production authorization; it needs a new untouched
hard-song holdout and commercial-rights clearance before deployment.

### Conditional onset-slot selection

A richer context selector may improve notes that the broad transcriber labels
as guitar without being safe enough to replace the established selector for
the whole score. Install it behind a family/register gate and preserve the
base model's onset slots:

```powershell
python -m ml.training.apply_conditional_selection_profile `
  --base "C:\models\pianella-current.json" `
  --alternative-profile "C:\research\held-out-context-profile.json" `
  --output "C:\research\conditional-candidate.json" `
  --profile-id pianella-conditional-candidate `
  --alternative-share 0.80 `
  --source-families guitar `
  --minimum-source-midi 60 `
  --maximum-source-midi 76 `
  --preserve-base-onset-counts `
  --onset-slot-window-seconds 0.035 `
  --validation-report "C:\research\whole-song-loso-summary.json"
```

The runtime removes each selector's own probability threshold before blending
their logits. It freezes all non-target notes and the number of eligible notes
at every established onset, so the alternate model may choose a better pitch
without adding rhythm, displacing voice/bass, or making the passage denser.
The alternate profile for each validation fold must have been trained without
that fold's song. Record the number of replaced source slots and reject a
candidate that wins only by increasing note count downstream.

After the reviewer records the still-blinded A/B verdict as JSON, verify every
frozen hash and gate before issuing a research promotion receipt:

```powershell
python -m ml.training.verify_piano_arranger_promotion `
  --freeze "C:\private-training\frozen-candidate\FREEZE.json" `
  --result "C:\private-training\FINAL_HOLDOUT_RESULT.json" `
  --mapping "C:\private-training\listening\SEALED-MAPPING.json" `
  --verdict "C:\private-training\listening\BLIND-VERDICT.json" `
  --output "C:\private-training\PROMOTION-RECEIPT.json"
```

The verifier fails closed if a profile/runtime hash changed, an automated gate
failed, the decisive blind row did not prefer the candidate, a supporting row
preferred the baseline, or the reviewer heard a blocker. Its receipt is
research-only and never grants commercial deployment rights.

Do not repeatedly tune against the same safety song. Once its result changes a
model-selection decision, that song is development evidence and a new untouched
song must replace it for the final claim.

### Focused singer pass and melody-path repair

A broad full-mix transcription can identify the correct instruments yet retain
too few singer events. The research-only two-pass route therefore runs the same
audio twice: once without constraints for factual instrumentation and once with
the instrument constraint `voice`. The focused result is never trusted merely
because it was requested. It must pass all of these guards:

1. The broad pass must contain a minimum amount and ratio of voice evidence.
2. Focused events must agree in pitch and onset with the broad pitched evidence.
3. Only an abnormally fragmented focused pass enters the monophonic path decoder.
4. The decoder uses continuity, singer range, and broad-pass anchors to select
   one plausible vocal pitch per onset frame and join same-pitch fragments.
5. A recovered broad singer anchor replaces a conflicting focused guess near the
   same onset.
6. Before physical hold shaping, decoded-vocal melody collisions inside 55 ms
   collapse to one note. This prevents a false second melody onset from choking
   the singer note after only a few milliseconds.

The collision repair is provenance-gated. If no note came from the focused
melody decoder, ordinary polyphonic melody events pass through unchanged. The
arranger records `focusedMelodyOnsetCollisionsRemoved` so every intervention is
auditable.

Singer clarity is a mix decision as well as a pitch decision. Do not infer a
60/40 balance from MIDI velocity: simultaneous chord notes add together, and
sample loudness varies across the keyboard. Measure it through the fixed render
chain instead:

```powershell
python -m ml.training.analyze_piano_mix_balance `
  --input "C:\private-training\candidate.json" `
  --samples "public\samples\iowa-mf" `
  --output-dir "C:\private-training\mix-balance" `
  --target-lead-share 0.60
```

The tool creates melody-only, accompaniment-only, and combined WAVs on one
timeline. Its primary number is RMS amplitude share during melody-active spans:
`melody / (melody + accompaniment)`. Treat the target as a tolerance band, not
fake decimal precision. A dynamics candidate must pass multiple development
songs and a blind listening preference; the measurement cannot promote it by
itself.

An untouched singer holdout must be chosen *after* the candidate hashes are
frozen. Never inspect a sealed mapping through a directory-wide JSON glob. If a
mapping is exposed before the listener records A, B, or TIE, invalidate that
pack and generate a fresh randomized pack; do not reuse it as blind evidence.

### Human-approved excerpts and hard cutoffs

If only part of a reference performance is trustworthy, the boundary is part
of the label contract, not a playback preference. Trim the target first, align
that immutable excerpt, then audit every downstream artifact:

```powershell
node scripts/training/trimPianoJsonTarget.mjs `
  --input "C:\private-training\pianella-full.json" `
  --output "C:\private-training\pianella-trusted.json" `
  --start 0 `
  --end 150 `
  --reason "Human review rejected the performance at and after 02:30."

npm run align:notes -- `
  --reference "C:\private-training\pianella-trusted.json" `
  --observed "C:\private-training\full-mix-notes.json" `
  --manual-anchors "C:\private-training\trusted-anchors.json" `
  --out "C:\private-training\alignment"
```

The interval is half-open: `[0, 150)`. Notes beginning at `150.0` are excluded,
and a note beginning earlier is physically clipped at the boundary. Derive a
boundary anchor using only earlier approved anchors; never consult a later
anchor and then pretend the fit is independent of the rejected tail.

Run `audit_trusted_excerpt.py` before treating retrained artifacts as clean. It
fails closed if a target onset, note end, alignment window, mapped label, or
song-specific chord prototype crosses the declared boundary. It can also scan
profiles for superseded target paths. Preview renders require a hard stop too:
release tails must not make rejected reference audio audible after the cutoff.

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
- A `decoder.performanceOnly` profile must preserve every accepted note's MIDI
  pitch, onset, and duration. It may calibrate one coherent hammer velocity per
  gesture and role-specific playback gain only; verify score identity before
  building a blind listening pack.

## Files

- `dataset_builder.py`: validates supervision/rights, splits by song, and creates
  5-second JSONL clip manifests.
- `import_slakh.py`: turns exact aligned Slakh stems into auditable piano-route
  supervision and records immutable song-level splits and CC BY attribution.
- `prepare_audio_clips.py`: uses FFmpeg to render mono 16 kHz WAV clips.
- `evaluate_predictions.py`: instrument-aware pattern/error diagnostics and note metrics.
- `synthesize_note_reference.py`: renders deterministic note JSON into a neutral
  WAV used only for audio-clock alignment.
- `analyze_piano_mix_balance.py`: renders isolated role stems and measures the
  real melody/accompaniment balance during singer-active spans.
- `compare_pianist_candidates.py`: compares two candidates on identical trusted
  windows and separates exact-key selection from common-note velocity errors.
- `audit_trusted_excerpt.py`: proves that a human-rejected reference tail did
  not enter labels, alignment windows, chord prototypes, or profile provenance.
- `calibrate_piano_gesture_dynamics.py`: fits eligible gesture velocity
  quantiles and chord-size strength without allowing long songs or rejected
  alignment windows to dominate.
- `fit_paired_gesture_velocity.py`: learns a conservative pianist-touch
  correction from aligned gestures with whole-song holdouts. Use it with a
  `decoder.performanceOnly` base profile when note selection and timing must
  remain frozen.
- `train_piano_arranger_adapter.py`: fits the small Piano-route event selector,
  duration model, role ranges, and density/style statistics.
- `blend_piano_arranger_profiles.py`: composes conservative residual selectors
  in raw logit space and records both source profiles in provenance.
- `apply_conditional_selection_profile.py`: gates a context selector to one
  source family/register and can freeze the base model's onset-slot budget.
- `cross_validate_piano_arranger_adapter.py`: performs leakage-aware
  leave-one-song-out route evaluation and pooled event-rate gates.
- `verify_piano_arranger_promotion.py`: verifies frozen hashes, held-out gates,
  and the sealed blind-listening verdict before issuing a research receipt.
- `scripts/training/exportPianoJsonMidi.mjs`: creates reproducible listening MIDI
  files using the browser playback-duration contract.
- `muscriptor_tokens.py`: MT3-like individual-instrument targets, overlap normalization, and ties.
- `train_muscriptor_piano.py`: audit-first partial checkpoint fine-tuning adapter.
- `configs/piano_v001.json`: base revision, data contract and promotion gates.
- `training-index.example.json`: private dataset index template.
- `tests/`: deterministic validation, leakage and metric tests.
