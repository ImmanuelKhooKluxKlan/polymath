# Polymath transcription: the 90% plan

## The promise we are measuring

“90% accuracy” must name a route, a metric, and unseen data. Polymath has two different problems:

1. **Direct piano transcription**: a clean piano recording becomes piano notes.
2. **Full-song piano reduction**: a commercial mix becomes a playable Pianella-style solo-piano arrangement that preserves melody and accompaniment.

The headline technical metric is micro-averaged exact pitch plus onset F1 at a 100 ms tolerance on continuous song timelines from a sealed, song-disjoint holdout. Artificial five-second training boundaries are merged before the headline score; a held note is not counted as a required re-strike. Precision and recall are both required, so deleting difficult notes cannot fake success. A clip-local score remains as a harsher debugging metric. We also report exact F1 at 50 and 250 ms, pitch-class F1 and recall, onset-plus-offset F1, frame F1, severe cutoff rate, rapid retriggers, note-count ratio, and blind-listening preference.

A route reaches the 90% milestone only when all of the following are true:

- exact pitch/onset F1 at 100 ms is at least 0.90 on a pre-registered sealed set;
- no important song subgroup is hidden by a strong aggregate;
- timing, duration, cutoff, and retrigger safety gates pass;
- a listener cannot identify a new “stutter”, “machine gun”, or masking regression;
- the exact model, profiles, code, and hashes used for the result are frozen.

Opened development songs can guide engineering, but they cannot certify the claim.

## Baseline frozen on 18 September 2026

| Route | Exact F1 @ 100 ms | Exact F1 @ 250 ms | Pitch-class F1 @ 250 ms | Meaning |
|---|---:|---:|---:|---|
| Direct piano | 0.8652 | 0.9139 | 0.9281 | Close enough that event/timing recovery can plausibly cross 0.90. |
| Full mix to Pianella reduction | 0.2671 | 0.3682 | 0.5947 | This is an arrangement/source-separation problem, not a post-processing-only problem. |

The direct route is 3.48 F1 points from the headline target. Its source already contains 91.72% pitch-class recall at 250 ms, so the evidence ceiling is high enough.

The full-mix source contains only 80.29% pitch-class recall at 250 ms. No arranger can honestly produce 90% factual coverage from an 80% evidence ceiling without another audio view or a stronger upstream model.

## Error budget

Every miss is assigned to one layer before code changes:

1. **Upstream evidence miss**: the correct pitch is absent from every model pass.
2. **Timing miss**: pitch exists but falls outside the scoring window.
3. **Register miss**: pitch class is correct but octave is wrong.
4. **Selection miss**: the correct source note existed and the arranger removed it.
5. **Insertion error**: a noisy stem or instrument leak was accepted.
6. **Performance error**: note identity is right but duration, retrigger, velocity, or pedal behavior is wrong.
7. **Alignment/label uncertainty**: the answer key and recording are not actually synchronized.

We work in that order. Velocity polishing cannot repair a missing note, and a pretty render cannot repair target leakage.

## Workstream A: measurement and leakage control

- Freeze manifests with full-song train, development, and sealed roles.
- Split by complete song and recording, never random five-second windows from the same song.
- Hash source audio, target, alignment, model checkpoint, profiles, and code.
- Use micro F1 for the headline so long songs and sparse songs cannot be cherry-picked.
- Keep per-song tables and worst-song gates beside the aggregate.
- Reject experiments that improve only data used to select their parameters.
- Keep commercial-rights status separate from technical accuracy.

Promotion sequence:

```text
train songs -> whole-song validation -> frozen candidate
                                      -> one untouched song
                                      -> sealed blind listening
                                      -> canary traffic
                                      -> production or rollback
```

## Workstream B: direct piano, 86.52% to 90%

The four opened pieces are deliberately uneven: Schubert is 96.93%, Chopin 83.36%, Rachmaninoff 80.12%, and Debussy 87.33% at 100 ms. The correct strategy is not to tune Schubert further.

### B1. Build a checkpoint-complement audit

Run the original, phase46-v007, and phase46-v017 checkpoints on the same four 90-second recordings with identical preprocessing. The trained checkpoint lost head-to-head as a replacement, but it may still contain complementary true events.

**Result:** completed. The three-checkpoint target-informed oracle reached 0.9015 recall, but no checkpoint won consistently: v017 helped Chopin and Rachmaninoff while hurting Schubert and Debussy. This proved complementary signal exists, but not that it can be selected safely at inference.

For every event, record:

- checkpoints agreeing on pitch and onset;
- events found only by the original checkpoint;
- events found only by a trained checkpoint;
- confidence, local polyphony, pitch class, register, onset density, and duration;
- whether the event matches the fixed answer key, for development analysis only.

### B2. Train an inference-safe ensemble selector

The selector may use only features available at inference. It must be fitted with one complete piece held out. Candidate operations are conservative:

- keep consensus events;
- recover a unique event only when another checkpoint and the waveform support it;
- abstain in dense or contradictory clusters;
- never globally shift a song or use its title as a routing feature.

**Result:** rejected for now. A conservative consensus rule gained only 0.0029 aggregate F1 and regressed two songs. A 32-feature event classifier fell from 0.8652 to 0.8538 in complete-song leave-one-out evaluation. Four opened pieces are too few and too heterogeneous to learn a trustworthy checkpoint router.

### B3. Timing correction only where confidence is high

Naive spectral-flux onset snapping was tested and rejected: exact F1 at 100 ms fell from 0.8652 to 0.8617. The next timing model must distinguish hammer attacks from pedal resonance and accompaniment energy. Apply it only to events whose pitch already agrees across passes, then validate at 50, 100, and 250 ms simultaneously.

### B4. Direct-route success gate

Development target: at least 0.90 micro exact F1 at 100 ms with every held-out piece no more than 0.5 point worse. Then freeze and run a new MAESTRO piece exactly once. MAESTRO evidence is research-only because of its licence.

### B5. Broader direct-piano checkpoint

Phase 93 is retained as a diagnostic, not a certification run. Its first split treated a composite credit such as `Robert Schumann / Franz Liszt` as one string, so individual people crossed train, validation, and test. All three epochs lowered teacher-forced loss, but every epoch failed decoded-note safety: recall fell and the worst 100 ms song regression reached 3.66 F1 points. No checkpoint was written and the Phase 93 test stayed unopened.

Phase 94 fixes the unit of separation. Composite credits are decomposed into individual people, connected composer/arranger components are assigned as indivisible groups, and 42 previously opened recordings are excluded. It contains 30 training recordings (2,861 clips), six validation recordings (247 clips), and eight unmaterialized sealed-test recordings. There is zero individual-person overlap across every split pair and zero duplicate audio hashes among the 3,108 materialized clips.

The immutable original checkpoint baseline on all 247 Phase 94 validation clips is:

| Tolerance | Precision | Recall | Micro F1 | Complete-song bootstrap 95% interval |
|---|---:|---:|---:|---:|
| 50 ms | 0.8457 | 0.8080 | 0.8264 | 0.8001-0.8459 |
| 100 ms | 0.9169 | 0.8761 | 0.8960 | 0.8820-0.9096 |
| 250 ms | 0.9311 | 0.8896 | 0.9099 | 0.8953-0.9260 |

That table is the deliberately strict clip-local audit. It counts 452 reviewed notes that continue through an artificial five-second cut as fresh labels in the next clip, even though a correct continuous performance must not re-strike them. The additional frozen song-timeline audit merges those explicit continuations before scoring:

| Continuous-song tolerance | Precision | Recall | Micro F1 | Complete-song bootstrap 95% interval |
|---|---:|---:|---:|---:|
| 50 ms | 0.8466 | 0.8310 | 0.8387 | 0.8183-0.8582 |
| 100 ms | 0.9183 | 0.9013 | 0.9097 | 0.8980-0.9237 |
| 250 ms | 0.9327 | 0.9155 | 0.9240 | 0.9081-0.9422 |

The original checkpoint therefore crosses 0.90 at the user-facing 100 ms onset point estimate, but its lower song-cluster confidence bound does not. It also does **not** have 90% duration quality: at 100 ms onset tolerance, onset-plus-offset F1 is 0.3432 and 20 ms frame F1 is 0.5586. This distinction explains why a numerically strong note-onset transcription can still sound chopped or overlong.

Duration labels have an additional ambiguity. Every Phase 94 validation performance contains substantial sustain-pedal CC64 traffic, while the current MAESTRO importer records physical key release and ignores pedal. The waveform cannot reliably reveal a key release while the damper remains lifted. A simple pedal-to-release extension raised frame F1 only from 0.5586 to 0.5921 and created many apparent cutoffs, so it is not adopted blindly. Future duration work must either model pedal explicitly or mask acoustically unobservable release targets.

On the clip-local metric, only 61 extra correct matches at fixed note counts would move the 100 ms point estimate to 0.90. Precision is already strong; the residual onset deficit is recall. On stitched songs, notes below velocity 0.20 are missed 68.0% of the time, notes from 0.20 to 0.34 are missed 25.9%, and the low A0-B2 range is weaker than middle/high ranges. These are development diagnostics, not permission to tune on the sealed reserve.

Candidate `phase94-v001` starts from the byte-hashed immutable original and changes only the final transformer layer plus output head. It retains the Phase 93 learning recipe for one epoch but evaluates at optimizer steps 90, 180, 270, and 358. This tests whether a safe early specialization point exists before the full-epoch recall regression. Every saved state must improve teacher-forced loss and pass decoded 100 ms/250 ms aggregate, recall, and per-song regression gates. A lower loss alone cannot win.

The first v001 save attempt exposed two infrastructure findings. The network-volume namespace rejected the final 5.47 GB serialization at quota; no original or prior model was deleted, and the exact log was archived before rerunning to Pod-local ephemeral storage. More importantly, nominally identical attempts produced opposite-signed 100 ms deltas at step 180 because the old seed controlled file order but not PyTorch/CUDA dropout. The next control now seeds Python, NumPy, PyTorch and CUDA, disables TF32, and requires deterministic algorithms. A paired `phase94-v002-note-on-135` plan was frozen before seeing the full v001 result; its only scientific change versus that deterministic control is note-on loss weight 1.0 to 1.35.

The recovered nondeterministic v001 checkpoint passed the complete 247-clip development gate. Clip-local 100 ms F1 rose from 0.8960 to 0.8974. On continuous songs it rose from 0.9097 to 0.9111; precision and recall both improved, five songs improved or tied, and the sixth changed by only -0.0002. Onset-plus-offset F1 improved by 0.0043 and frame F1 by 0.0036, but overlong notes increased from 977 to 985 and rapid retriggers from 55 to 57. It is therefore retained as a research comparator, not a production winner. Its complete-song lower 95% bound remains 0.8985 and the run is not causally reproducible until the deterministic control/candidate pair finishes.

The Phase 94 corpus is MAESTRO CC BY-NC-SA 4.0. It is valid research evidence but the resulting checkpoint is blocked from commercial production. MuScriptor's paper describes 1.45 million synthetic MIDI files and an internal 170,000-recording corpus without publishing an item inventory, so overlap between MAESTRO and the foundation checkpoint is unknown. Phase 94 is a valid Polymath fine-tuning holdout, but it is not proven unseen to the foundation model. A commercially licensed or first-party post-foundation corpus must reproduce the result before public promotion.

## Workstream C: full mix, raise the evidence ceiling first

### C1. Multiple audio views

Transcribe the same timebase through:

- original full mix;
- isolated or emphasized vocals;
- no-vocals/accompaniment;
- optional piano/guitar stem when source separation confidence is high.

The passes are evidence, not independent songs. All must share one audio clock and one preprocessing receipt.

### C2. Focused vocal decoder

Raw vocal-stem insertion increased pitch-class coverage but damaged exact-note precision. The frozen Phase 92 candidate therefore:

- requires exact-pitch corroboration from the broad pass;
- decodes the vocal cloud into one continuous held melody path;
- preserves broad-pass vocal anchors;
- rejects unanchored polyphonic leakage;
- rejects the entire focused pass below 1.4 corroborated notes/second.

Whole-song leave-one-out development result:

| Metric | Delta versus production-v003 |
|---|---:|
| Exact F1 @ 100 ms | +0.0069 |
| Exact recall @ 100 ms | +0.0093 |
| Exact F1 @ 250 ms | +0.0184 |
| Pitch-class F1 @ 250 ms | +0.0168 |
| Pitch-class recall @ 250 ms | +0.0266 |
| Severe cutoff rate | -0.0172 |
| Rapid retriggers | no change |

This is a frozen research candidate, not a deployment. Earlier focused-vocal candidates sometimes won automated scores and still lost blind listening, so a fresh listening gate is mandatory.

### C3. Accompaniment selector

The no-vocals pass often improves precision but loses recall. Use it to propose low-hand tones, then accept only notes anchored by a nearby broad-pass gesture. Evaluate dense clusters separately because clusters of eight or more source notes currently have the weakest precision.

### C4. Style after facts

Only after pitch/onset selection passes do we tune Pianella-style properties: hand range, melody/accompaniment balance, held notes, pedal, chord voicing, and dynamics. Style scores and factual note scores remain separate.

## Workstream D: data needed to reach 90%

The current few-song corpus is enough to discover failure modes, not enough to establish a broad 90% product claim.

Minimum next corpus:

- 30 clean-piano songs for direct transcription;
- 30 full-mix songs with synchronized, human-approved piano reductions;
- balanced slow/fast tempo, sparse/dense texture, male/female/no vocals, acoustic/electronic production, and multiple keys;
- at least 20% never opened until the candidate is frozen;
- no artist or source recording appearing across train and sealed partitions.

Labels need a confidence mask. Uncertain alignment regions are ignored rather than taught as truth. Five-second clips are useful training units only after their parent songs have been split.

## Seven-day execution board

### Day 1: lock measurement

- Completed: route-aware scorecard and source-evidence ceilings.
- Completed: provenance indices preserved without changing musical output.
- Completed: naive onset snapping rejected by whole-song transfer.
- Completed: focused-vocal density router reached sealed-holdout-candidate status.

### Day 2: checkpoint complement

- Completed: generated original, v007, and v017 outputs for four direct-piano pieces.
- Completed: confirmed complementary recall but inconsistent checkpoint winners.
- Completed: rejected unsafe consensus and learned-selector branches.

### Day 3: direct selector

- Completed: song-held-out selector evaluation failed the transfer gate.
- Completed: rejected Phase 93 as certification evidence after finding individual-person leakage in composite credits; no candidate was saved.
- Completed: built and integrity-checked the corrected Phase 94 individual-person-disjoint corpus and measured the immutable original on every validation clip.
- Completed: full validation and continuous-song attribution for the recovered `phase94-v001` research comparator; it improved but did not certify 90% across songs.
- Completed: paired deterministic control/note-on-weight comparison and the `phase94-v003` decoded-quality checkpoint-selection replay.
- Completed: frozen shifted-window boundary experiment and its release-preserving repair; the repaired policy is the first development result whose complete-song 100 ms lower 95% bound exceeds 0.90 while every onset/duration safety check passes.
- Completed and rejected: the fixed v002 plus 0.60-second release-preserving overlap combination. It produced the strongest onset result, but missed the frozen overlong-rate limit by 0.108282 per 1,000 matches, so it was not advanced.
- Completed: decoded the already-passing original-weights plus release-preserving-overlap winner on an opened Kiss Me full-song transfer pack and built a committed blind A/B listening test. The mapping remains sealed pending the owner's verdict.

### Day 4: full-mix selector

- Add at least one new, properly licensed full-mix/target pair.
- Run the frozen Phase 92 vocal route exactly once.
- If it fails, consume the song as development evidence and diagnose; never reuse it as sealed proof.

### Day 5: duration and performance

- Tune holds, repeated-note release, and pedal only on candidates whose pitch/onset gates already pass.
- Render blind A/B audio with identical piano samples and loudness normalization.

### Day 6: freeze and blind test

- Freeze code, profiles, candidate output, hashes, mapping commitment, and pass/fail thresholds.
- Conduct blind listening before revealing A/B identities.

### Day 7: canary or reject

- If every gate passes, enable for admin traffic only and log route decisions.
- Compare failure rate, latency, note density, and user preference against v003.
- Roll back automatically on error-rate or latency regression.

## Stop rules

- Stop a branch after two song-hidden regressions; document it and move on.
- Do not add parameters unless an error analysis names the failure they address.
- Do not lower a gate after seeing the test result.
- Do not fine-tune weights until the inference-only ceiling audit proves weights are the limiting layer.
- Do not call a post-processing profile “new weights.”
- Do not deploy non-commercial upstream weights commercially without written permission or replacement.

## Current next action

The nondeterministic `phase94-v001` comparator has completed full validation. It passes the subsequently automated research-quality onset/duration gate but not certification: continuous-song 100 ms F1 improved by 0.001411, the worst song changed by only -0.000196, and duration/retrigger limits passed, while the complete-song bootstrap lower bound remains 0.898536. Change attribution found 65 recovered versus 48 regressed reference notes and 26 fewer false positives. Most recovered notes were short events in medium/dense texture, not the very quiet notes.

The paired deterministic control versus `phase94-v002-note-on-135` full comparison is complete. Uniform extra note-on loss pressure improved continuous-song 100 ms F1 from 0.911122 to 0.911717 and recall from 0.902431 to 0.903292. Its complete-song lower 95% bound rose from 0.898536 to 0.899542. The candidate passed every pre-frozen v002 onset, release, frame, retrigger, and per-song safety limit, but 50 ms F1, 250 ms F1, onset-plus-offset F1, and frame F1 each dipped slightly. It is retained as a promising research candidate, not certified or production-ready.

The deterministic `phase94-v003-decoded-selection-control` replay selected optimizer step 90 rather than the lowest-loss final step 358. That proves the checkpoint-selection repair works: later loss continued falling while decoded note quality declined. On all 247 clips, however, step 90 reached only 0.910008 continuous-song 100 ms F1 with a 0.898224 lower 95% song bound. It passed the frozen research safety gate but was weaker than v002's 0.911717 F1 and 0.899542 lower bound, so v002 remains the leading checkpoint.

The first frozen shifted-window policy confirmed the boundary hypothesis. Its 0.60-second winner raised continuous-song 100 ms F1 from 0.909711 to 0.913561 and the lower 95% bound to 0.902543, but hard replacement increased overlong notes from 71.801 to 80.996 per 1,000 matches. The pre-frozen gate rejected it; no post-result radius substitution was allowed.

The separately frozen `phase94-overlap-release-preserving-v003` repair uses the shifted onset near a five-second cut but keeps the primary absolute release when both passes contain the same instrument/pitch within 100 ms. It filters candidates through the complete safety gate before ranking. Four of six radii were eligible; the 0.60-second winner reached 0.914147 F1 with a 0.903087 lower 95% bound. Every onset, per-song, onset-plus-offset, frame, cutoff, overlong, and rapid-retrigger check passed. Overlong growth fell to 1.289 per 1,000 matches, inside the frozen allowance, and the worst song changed by only -0.000232.

The fixed `phase94-v002-overlap-release-preserving-v004` combination has now completed without retuning. It reached 0.915869 continuous-song 100 ms F1, 0.904456 lower 95% bound, 0.348394 onset-plus-offset F1, and 0.566510 frame F1. All six songs improved and cutoffs fell from 669 to 664. The complete frozen gate nevertheless rejected it: overlong rate rose from 71.801279 to 73.909561 per 1,000 matched notes, an increase of 2.108282 against a maximum allowed increase of 2.0. This is approximately a two-note near miss at this panel size, but the threshold must not be relaxed after observing it.

The original-weights `phase94-overlap-release-preserving-v003` result therefore remains the development winner because it is the only candidate that combines a lower 95% bound above 0.90 with every safety check passing.

Its fixed-policy opened-song transfer is now complete on Kiss Me. The immutable input was split into 40 primary and 40 overlapping clips, all 80 clips decoded without error, and the research Pod was stopped after the artifacts were downloaded. The primary timeline contains 836 notes and the overlap candidate contains 867 notes over the same 197.76 seconds.

The separately frozen unlabeled structural audit passed all 13 checks without opening the blind mapping. All 187 changes are confined to the predeclared 0.60-second boundary region: 70 same-pitch pairs moved, 43 primary notes were removed, and 74 candidate notes were added. Maximum onset movement is 100 ms, maximum release movement is 50 ms, the 75 ms retrigger count remains zero, the 100 ms retrigger count remains one, duration P95 changes from 2.400 to 2.397 seconds, neither render clips, and the loudness difference is only 0.169054 dB RMS.

The blind listening pack, source artifacts, freezes, and commitment all pass pre-reveal integrity verification. `finalize_overlap_transfer_verdict.py` now records the listener's A/B/TIE choice atomically before it parses the sealed mapping, rejects mutated artifacts, and fails closed on a tie, baseline preference, structural failure, or audible blocker.

The single-use eight-song protocol is also frozen before the blind reveal. Its preflight proves that all eight selected audio/MIDI pairs are absent, the prepared test manifest is empty, and no test content has been parsed. The custody chain authorizes audio first, prepares label-free primary and shifted passes, records checkpoint SHA-256 in both GPU evaluations, locks every prediction hash, and only then permits MIDI labels. The post-lock label builder verifies every downloaded MIDI hash and proves that adding references changed none of the frozen clip identities, timestamps, or audio sources. A changed artifact fails closed. The final score must bind to those locked hashes, use the same authorized reference manifest on both sides, beat the immutable original, pass every duration/stability gate, reach at least 0.900 point F1, and keep the song-bootstrap lower 95% bound at or above 0.900. A failure consumes the test permanently. The full repository suite passes with 665 tests and 2 subtests.

The immediate next input is the owner's blind A/B/TIE verdict on the two Kiss Me renders. Do not inspect `SEALED-MAPPING.json` before recording that choice. If the candidate loses or has an audible blocker, reject this branch and use the opened song only for future development. If it wins, freeze the entire v003 pipeline for the single-use sealed eight-song test; do not retune it first. Any further duration repair must be designed and selected on new duration-labelled development material or an inner split; the six opened Phase 94 validation songs cannot be used for another post-hoc two-note adjustment.

Only after one candidate, all thresholds, and the model-to-label mapping are frozen may the eight-song reserve be materialized and opened once. No production flag changes are allowed until sealed automated evaluation, blind listening, rights-cleared reproduction, and rollback checks all pass.
