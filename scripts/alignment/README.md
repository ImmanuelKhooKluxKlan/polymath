# Polymath note-coordinate alignment prototype

This research tool compares an ideal performance with notes produced by
Polymath/MuScriptor. Either side can be a MIDI file or a note JSON file. It
estimates a global start/average-speed line, rejects unrelated
notes, constructs a local nonlinear timeline, and writes an SVG coordinate plot.

It deliberately does not trust the first note. A candidate timeline must be
supported by an ordered sequence of notes distributed across the recording.
Notes in the same pitch class but a different octave can support a hypothesis at
reduced weight, while exact pitches receive more weight.

## Run

```powershell
npm run align:notes -- --muscriptor "C:\path\transcription.json" --midi "C:\path\desired.mid" --out "C:\path\alignment-output"
```

For new comparisons, use `--reference` for the ideal file and `--observed` for
the model file. Both inputs accept `.mid`, `.midi`, or `.json`. The older
`--midi` and `--muscriptor` names remain available for existing commands.

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
