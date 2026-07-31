# Polymath Musician 3.1 — Unified MIDI/JSON Audio Upgrade

## What changed

- MIDI files are normalized into the same ready-to-play performance JSON model used by native JSON songs.
- MIDI tempo changes, note-on/note-off timing, velocity, channel volume/expression, sustain CC64, soft pedal CC67, program changes, time signature, key signature, and track names are retained where available.
- General-MIDI percussion on channel 10 is separated from melodic piano notes, preventing drum events from sounding as incorrect piano pitches.
- Structured JSON songs now preserve their supplied score, visual, and audio durations by default.
- Piano autoplay no longer obeys dangerously short release tails literally. Musical minimum tails, register-aware releases, smoother two-stage damping, pedal release resonance, deterministic room tone, and exact scheduled starts reduce dry or cut-off playback.
- Guitar playback uses a fuller body response, longer natural tails, deterministic strum humanization, and a safer release floor.
- Ensemble instruments now share a studio master chain with EQ, deterministic room ambience, compression, limiting, and smoother releases.
- Added free ready-to-play songs:
  - Piano: Kiss Me, Enchanted, Style, We Are Never Getting Back Together, Stay Stay Stay
  - Guitar: Wildest Dreams

## Compatibility

The existing upload, PDF translation, marketplace, account, PayPal, YouTube comparison, visual-teacher, mobile, piano, guitar, and ensemble features remain in place.
