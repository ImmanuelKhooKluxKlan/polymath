import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from ml.training.render_piano_json_audio import (
    SAMPLE_RATE,
    midi_to_sample_name,
    normalize_render_notes,
    render_payload,
)


class RenderPianoJsonAudioTests(unittest.TestCase):
    def test_midi_to_flat_sample_name_covers_piano_edges(self):
        self.assertEqual(midi_to_sample_name(21), "A0.wav")
        self.assertEqual(midi_to_sample_name(61), "Db4.wav")
        self.assertEqual(midi_to_sample_name(108), "C8.wav")

    def test_audio_duration_has_priority_and_invalid_notes_are_removed(self):
        notes = normalize_render_notes(
            {
                "notes": [
                    {"midi": 60, "time": 1, "duration": 0.2, "audioDuration": 0.8, "velocity": 2},
                    {"midi": 4, "time": 0, "duration": 1},
                ]
            }
        )
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["duration"], 0.8)
        self.assertEqual(notes[0]["velocity"], 1.0)
        self.assertEqual(notes[0]["performanceGain"], 1.0)

    def test_performance_gain_is_bounded_separately_from_velocity(self):
        notes = normalize_render_notes(
            {
                "notes": [
                    {
                        "midi": 60,
                        "time": 0,
                        "duration": 0.2,
                        "velocity": 0.7,
                        "performanceGain": 4,
                    }
                ]
            }
        )
        self.assertEqual(notes[0]["velocity"], 0.7)
        self.assertEqual(notes[0]["performanceGain"], 1.5)

    def test_training_eligible_filter_excludes_unreviewed_notes(self):
        notes = normalize_render_notes(
            {
                "notes": [
                    {
                        "midi": 60,
                        "time": 0,
                        "duration": 0.2,
                        "trainingEligible": True,
                    },
                    {
                        "midi": 62,
                        "time": 0.2,
                        "duration": 0.2,
                        "trainingEligible": False,
                    },
                ]
            },
            training_eligible_only=True,
        )

        self.assertEqual([note["midi"] for note in notes], [60])

    def test_renderer_writes_fixed_length_non_silent_stereo_wav(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = root / "samples"
            samples.mkdir()
            frames = np.zeros((SAMPLE_RATE, 2), dtype="<i2")
            frames[:1000, :] = 8000
            with wave.open(str(samples / "C4.wav"), "wb") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(2)
                handle.setframerate(SAMPLE_RATE)
                handle.writeframes(frames.tobytes())
            destination = root / "render.wav"
            summary = render_payload(
                {"notes": [{"midi": 60, "time": 0.1, "duration": 0.2, "velocity": 0.8}]},
                samples,
                destination,
                duration_seconds=2.0,
            )
            self.assertTrue(destination.is_file())
            self.assertEqual(summary["seconds"], 2.0)
            self.assertGreater(summary["peak"], 0.0)
            self.assertFalse(summary["perFileNormalization"])
            with wave.open(str(destination), "rb") as handle:
                self.assertEqual(handle.getnchannels(), 2)
                self.assertEqual(handle.getnframes(), SAMPLE_RATE * 2)

    def test_hard_stop_clips_notes_and_release_tails_beyond_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = root / "samples"
            samples.mkdir()
            frames = np.zeros((SAMPLE_RATE, 2), dtype="<i2")
            frames[:, :] = 8000
            with wave.open(str(samples / "C4.wav"), "wb") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(2)
                handle.setframerate(SAMPLE_RATE)
                handle.writeframes(frames.tobytes())
            destination = root / "hard-stop.wav"

            summary = render_payload(
                {
                    "notes": [
                        {"midi": 60, "time": 0.1, "duration": 4.0, "velocity": 0.8},
                        {"midi": 60, "time": 3.0, "duration": 1.0, "velocity": 0.8},
                    ]
                },
                samples,
                destination,
                hard_stop_seconds=1.25,
            )

            self.assertEqual(summary["seconds"], 1.25)
            self.assertEqual(summary["hardStopSeconds"], 1.25)
            with wave.open(str(destination), "rb") as handle:
                self.assertEqual(handle.getnframes(), SAMPLE_RATE * 5 // 4)


if __name__ == "__main__":
    unittest.main()
