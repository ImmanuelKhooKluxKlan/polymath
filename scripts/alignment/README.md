# Polymath note-coordinate alignment prototype

This research tool compares a desired MIDI performance with note events produced
by MuScriptor. It estimates a global start/average-speed line, rejects unrelated
notes, constructs a local nonlinear timeline, and writes an SVG coordinate plot.

It deliberately does not trust the first note. A candidate timeline must be
supported by an ordered sequence of notes distributed across the recording.
Notes in the same pitch class but a different octave can support a hypothesis at
reduced weight, while exact pitches receive more weight.

## Run

```powershell
npm run align:notes -- --muscriptor "C:\path\transcription.json" --midi "C:\path\desired.mid" --out "C:\path\alignment-output"
```

Outputs:

- `alignment-plot.svg`: x = desired MIDI time, y = source/MuScriptor time.
- `alignment-report.json`: matches, rejected-note counts, timing errors, and confidence.
- `aligned-training-labels.json`: desired notes moved onto the source-audio timeline.

Green plot points are exact pitches. Amber points match the pitch class but have
an octave disagreement, such as C5 versus C6. The dashed line is the robust
global estimate; the purple path is the locally changing time map.

The output verdict is a quality gate, not a promise of ground truth. Low-confidence
or structurally different songs must be manually anchored or rejected before
they are used for supervised learning.
