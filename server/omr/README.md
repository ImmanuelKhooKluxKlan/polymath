# Polymath Local OMR

This folder converts a PDF music score into `polymath-musician-json-v1`
without sending the score to a paid inference API.

PDF rendering and text extraction use PDFium through `pypdfium2`, whose Python
binding is Apache-2.0/BSD-3-Clause and whose bundled PDFium is BSD-style. The
strong-copyleft PyMuPDF/MuPDF dependency is deliberately not used.

## Recognition ladder

The engine chooses the strongest evidence available, in this order:

1. **Embedded MusicXML** — exact symbolic pitch, voices, chords, duration,
   tempo changes, key/time signatures, ties, rests, and staff assignment.
2. **SMuFL vector PDF** — reads standardized music-font noteheads, clefs,
   accidentals, flags, augmentation dots, and repeat dots directly from the PDF
   text layer. Raster vision is used only to cross-check stems, beams, staffs,
   and barlines.
3. **Scanned-page vision** — renders at high resolution, deskews, applies an
   adaptive threshold, detects five-line staffs and systems, removes staff
   lines, detects conservative notehead candidates, maps pitch from staff
   coordinates, and reconstructs quantized measure timing.

Low-confidence pages fail the job and trigger the existing allowance/Mcoin
refund. The engine never silently returns an empty or guessed result.

## Files

- `run_omr.py`: stable command-line boundary used by Node.
- `polymath_omr/pipeline.py`: provider selection and final result metadata.
- `polymath_omr/musicxml.py`: exact embedded-MusicXML conversion.
- `polymath_omr/vision.py`: PDF rendering, SMuFL extraction, staff/symbol
  recognition, pitch/rhythm reconstruction, repeats, diagnostics, confidence.
- `polymath_omr/music.py`: clef, pitch, key-signature, and tie helpers.
- `tests/test_omr.py`: symbolic, embedded-source, and raster score tests.

## Run locally

```powershell
python -m pip install -r server/omr/requirements.txt
python server/omr/run_omr.py `
  --input "C:\path\score.pdf" `
  --output "$env:TEMP\score.json" `
  --instrument piano
```

Run the automated suite:

```powershell
npm run test:omr
npm test --prefix server
```

## Server configuration

```dotenv
OMR_ENABLED=true
OMR_PYTHON_BIN=python3
OMR_RENDER_DPI=300
OMR_MAX_PAGES=20
OMR_TIMEOUT_MS=1500000
```

`Dockerfile.api` installs the pinned Python requirements. The Node queue invokes
the worker through `server/localOmr.js`; no shell interpolation is used.

## Accuracy contract

The JSON contains an overall `confidence`, warnings, and `omrDiagnostics` with
page/staff/system/measure counts, accepted and rejected marks, engine choice,
deskew angles, repeat expansion, and component confidence.

Embedded MusicXML is the only path treated as near-exact. SMuFL PDFs have exact
notehead positions and strong pitch evidence, but complex tuplets, ornaments,
voltas, D.S./D.C./Coda jumps, cross-staff voices, ties, lyrics, dynamics, and
pedal semantics still need a review layer. Scans have lower confidence and may
omit unclear marks. These limitations must not be hidden from users.

## Refinement path

The deterministic engine is also a data generator. Save review corrections as
symbol bounding boxes plus the correct glyph/pitch/duration. Those examples can
train local staff, glyph, and relationship models later while the same
ready-to-play schema and job infrastructure remain unchanged.
