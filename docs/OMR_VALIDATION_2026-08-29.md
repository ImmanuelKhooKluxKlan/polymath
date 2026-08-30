# Local PDF-to-piano validation — 2026-08-29

## Purpose

This record measures the deterministic SMuFL/vector-PDF path against the MIDI
files supplied with three Taylor Swift piano scores. It is a regression record,
not a claim of universal OMR accuracy. A MIDI performance can differ from the
printed score, so every residual mismatch still requires score-level review.

## Matching rule

- Flatten non-percussion MIDI tracks into `(pitch, onset, duration)` notes.
- Match each reference note once to the closest generated note of the same MIDI
  pitch within 40 milliseconds.
- Report recall against the reference count and precision against the generated
  count.
- Count a duration match when the matched duration differs by at most 80 ms.
- Use the generated score duration, before browser articulation/pedal shaping.

## Results

| Score | Reference | Generated | Onset + pitch matches | Recall | Precision | Duration matches |
|---|---:|---:|---:|---:|---:|---:|
| 22 | 1,565 | 1,567 | 1,553 | 99.23% | 99.11% | 1,540 |
| We Are Never Ever Getting Back Together | 1,260 | 1,277 | 1,248 | 99.05% | 97.73% | 1,182 |
| Blank Space (holdout) | 1,431 | 1,431 | 1,425 | 99.58% | 99.58% | 1,414 |

The original geometry-only timing baseline matched 804/1,565 notes (51.37%)
for **22** and 633/1,260 notes (50.24%) for **We Are Never Ever Getting Back
Together**. The holdout score was not used to design the parallel-voice fix.

## Changes represented by this benchmark

1. Preserve straight vector paths instead of retaining only Bezier curves.
2. Read exact two-segment vector stems and four-segment beam polygons.
3. Assign beams to the correct piano staff and correct side of a notehead.
4. Treat SMuFL flags, explicit note values, augmentation dots, and vector beams
   as authoritative rhythmic evidence.
5. Balance every measure on a 48-tick-per-quarter legal rhythm grid.
6. Detect two simultaneous piano voices when mixed held-note columns provide
   repeated anchors, then solve the moving voice within each held-note span.
7. Join printed tie halves across system and page boundaries before playback.
8. Keep written duration, physical key hold, repeated-key release, and damper
   duration separate.

## Remaining accuracy boundary

These scores are vector PDFs with a usable SMuFL text layer. Scans, handwriting,
tuplets, ornaments, unusual navigation, cross-staff notation, and visually
encoded expression remain separate test classes. `embedded-musicxml` is the
only path that should be described as near-exact without human review.

