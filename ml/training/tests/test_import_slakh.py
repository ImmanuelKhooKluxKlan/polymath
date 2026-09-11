import tempfile
import unittest
import wave
from pathlib import Path

import mido

from ml.training.import_slakh import (
    StemInfo,
    canonical_group_for_program,
    discover_stems,
    exact_quality_windows,
    fixed_splits,
    read_midi_notes,
    supervision_package,
)


class SlakhImportTests(unittest.TestCase):
    def test_general_midi_programs_map_to_public_model_groups(self):
        self.assertEqual(canonical_group_for_program(0), "acoustic_piano")
        self.assertEqual(canonical_group_for_program(24), "acoustic_guitar")
        self.assertEqual(canonical_group_for_program(33), "electric_bass")
        self.assertEqual(canonical_group_for_program(40), "violin")
        self.assertEqual(canonical_group_for_program(52), "voice")
        self.assertEqual(canonical_group_for_program(80), "synth_lead")
        self.assertEqual(canonical_group_for_program(0, is_drum=True), "drums")

    def test_split_is_song_level_and_leaves_test(self):
        splits = fixed_splits([f"Track{i:05d}" for i in range(1, 21)], 16, 2)
        self.assertEqual(sum(value == "train" for value in splits.values()), 16)
        self.assertEqual(sum(value == "validation" for value in splits.values()), 2)
        self.assertEqual(sum(value == "test" for value in splits.values()), 2)
        self.assertEqual(splits["Track00001"], "train")
        self.assertEqual(splits["Track00020"], "test")

    def test_midi_tempo_and_retrigger_are_decoded_in_seconds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "S01.mid"
            midi = mido.MidiFile(ticks_per_beat=480)
            track = mido.MidiTrack()
            midi.tracks.append(track)
            track.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
            track.append(mido.Message("note_on", note=60, velocity=100, time=0))
            track.append(mido.Message("note_on", note=60, velocity=90, time=240))
            track.append(mido.Message("note_off", note=60, velocity=0, time=240))
            midi.save(path)
            notes = read_midi_notes(
                StemInfo("S01", path, 0, "acoustic_piano", False),
                maximum_duration=2.0,
            )
            self.assertEqual(len(notes), 2)
            self.assertAlmostEqual(notes[0]["time"], 0.0)
            self.assertAlmostEqual(notes[0]["duration"], 0.25)
            self.assertAlmostEqual(notes[1]["time"], 0.25)
            self.assertAlmostEqual(notes[1]["duration"], 0.25)

    def test_exact_windows_cover_partial_tail(self):
        windows = exact_quality_windows(11.2)
        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[-1]["sourceStart"], 10.0)
        self.assertEqual(windows[-1]["sourceEnd"], 11.2)
        self.assertTrue(all(window["status"] == "trusted" for window in windows))

    def test_existing_artifacts_override_stale_babyslakh_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            track = Path(directory)
            (track / "MIDI").mkdir()
            (track / "stems").mkdir()
            (track / "MIDI" / "S00.mid").write_bytes(b"midi")
            (track / "stems" / "S00.wav").write_bytes(b"wave")
            stems = discover_stems(
                track,
                {
                    "midi_dir": "MIDI",
                    "audio_dir": "stems",
                    "stems": {
                        "S00": {
                            "midi_saved": False,
                            "audio_rendered": False,
                            "program_num": 0,
                            "is_drum": False,
                        }
                    },
                },
            )
            self.assertEqual(len(stems), 1)
            self.assertEqual(stems[0].instrument, "acoustic_piano")

    def test_supervision_has_provenance_and_clamps_note_to_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "mix.wav"
            with wave.open(str(audio), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(b"\x00\x00" * 16000)
            target = {
                "notes": [
                    {"midi": 60, "time": 0.8, "duration": 1.0, "velocity": 0.7},
                ]
            }
            package = supervision_package(
                "Track00001", audio, target, 1.0, "arranger-teacher"
            )
            self.assertEqual(package["schema"], "polymath-supervision-package-v1")
            self.assertEqual(package["provenance"]["license"], "CC-BY-4.0")
            self.assertAlmostEqual(package["notes"][0]["duration"], 0.2)
            self.assertTrue(package["notes"][0]["trainingEligible"])


if __name__ == "__main__":
    unittest.main()
