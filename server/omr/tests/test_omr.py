import json
import sys
import tempfile
import unittest
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw


OMR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OMR_ROOT))

from polymath_omr.music import staff_step_to_midi  # noqa: E402
from polymath_omr.musicxml import parse_musicxml  # noqa: E402
from polymath_omr.pipeline import transcribe_pdf  # noqa: E402


MUSICXML = b'''<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <work><work-title>Exact Embedded Score</work-title></work>
  <identification><creator type="composer">Polymath Test</creator></identification>
  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>4</divisions><key><fifths>1</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time><staves>2</staves>
      </attributes>
      <direction><sound tempo="90"/></direction>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><staff>1</staff></note>
      <note><chord/><pitch><step>E</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><staff>1</staff></note>
      <note><pitch><step>G</step><octave>3</octave></pitch><duration>8</duration><voice>2</voice><staff>2</staff></note>
    </measure>
  </part>
</score-partwise>'''


class MusicHelpersTest(unittest.TestCase):
    def test_staff_pitch_mapping_uses_standard_clefs(self):
        self.assertEqual(staff_step_to_midi("treble", 0), 64)  # E4
        self.assertEqual(staff_step_to_midi("treble", 2), 67)  # G4
        self.assertEqual(staff_step_to_midi("bass", 0), 43)  # G2

    def test_embedded_musicxml_preserves_chords_hands_and_tempo(self):
        result = parse_musicxml(MUSICXML, "piano", "score.musicxml")
        self.assertEqual(result["bpm"], 90)
        self.assertEqual(result["keySignature"], "G major")
        self.assertEqual(len(result["notes"]), 3)
        self.assertEqual(result["notes"][0]["time"], result["notes"][1]["time"])
        self.assertEqual({note["hand"] for note in result["notes"]}, {"left", "right"})
        self.assertEqual(result["confidence"], 0.995)


class PdfPipelineTest(unittest.TestCase):
    def test_embedded_musicxml_is_preferred_over_visual_guessing(self):
        with tempfile.TemporaryDirectory() as temp:
            pdf_path = Path(temp) / "embedded.pdf"
            document = pdfium.PdfDocument.new()
            document.new_page(595, 842)
            attachment = document.new_attachment("score.musicxml")
            attachment.set_data(MUSICXML)
            document.save(pdf_path)
            document.close()

            result = transcribe_pdf(pdf_path, "piano")
            self.assertEqual(result["translationProvider"], "Polymath Local OMR")
            self.assertEqual(result["omrDiagnostics"]["engine"], "embedded-musicxml")
            self.assertEqual(len(result["notes"]), 3)

    def test_visual_reader_finds_staff_and_playable_notes(self):
        with tempfile.TemporaryDirectory() as temp:
            pdf_path = Path(temp) / "vision-score.pdf"
            image = Image.new("L", (1240, 1754), 255)
            drawing = ImageDraw.Draw(image)
            for y in (400, 420, 440, 460, 480):
                drawing.line((140, y, 1090, y), fill=0, width=2)
            # Three filled quarter notes after the clef/key-signature guard.
            for x, y, stem_up in ((350, 480, True), (520, 460, True), (800, 440, False)):
                drawing.ellipse((x - 12, y - 8, x + 12, y + 8), fill=0)
                stem_x = x + 10 if stem_up else x - 10
                stem_end = y - 64 if stem_up else y + 64
                drawing.line((stem_x, y, stem_x, stem_end), fill=0, width=3)
            drawing.line((660, 396, 660, 484), fill=0, width=3)
            drawing.line((1088, 396, 1088, 484), fill=0, width=3)
            image.save(pdf_path, "PDF", resolution=150.0)

            result = transcribe_pdf(pdf_path, "piano", dpi=300)
            self.assertGreaterEqual(result["omrDiagnostics"]["staffs"], 1)
            self.assertGreaterEqual(len(result["notes"]), 2)
            self.assertTrue(all(note["duration"] > 0 for note in result["notes"]))
            json.dumps(result)


if __name__ == "__main__":
    unittest.main()
