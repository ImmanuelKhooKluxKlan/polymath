from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import mido


TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from select_csd_final_holdout import select_from_archive  # noqa: E402


def midi_bytes(note_count: int) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as handle:
        path = Path(handle.name)
    try:
        midi = mido.MidiFile()
        track = mido.MidiTrack()
        midi.tracks.append(track)
        for index in range(note_count):
            # Deliberately vary pitch: selection must still expose only the count.
            track.append(mido.Message("note_on", note=48 + index % 24, velocity=80, time=0))
            track.append(mido.Message("note_off", note=48 + index % 24, velocity=0, time=1))
        midi.save(path)
        return path.read_bytes()
    finally:
        path.unlink(missing_ok=True)


class SelectCsdFinalHoldoutTests(unittest.TestCase):
    def test_selects_first_non_excluded_midi_meeting_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "CSD.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("CSD/english/mid/en001a.mid", midi_bytes(120))
                archive.writestr("CSD/english/mid/en002a.mid", midi_bytes(80))
                archive.writestr("CSD/english/mid/en003a.mid", midi_bytes(100))
                archive.writestr("CSD/english/mid/en004a.mid", midi_bytes(130))

            selected, scan = select_from_archive(
                archive_path,
                excluded={"en001a"},
                minimum_notes=100,
            )

            self.assertEqual(selected, "en003a")
            self.assertEqual(
                scan,
                [
                    {"basename": "en002a", "targetMidiNoteEvents": 80, "eligible": False},
                    {"basename": "en003a", "targetMidiNoteEvents": 100, "eligible": True},
                ],
            )


if __name__ == "__main__":
    unittest.main()
