import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from ml.training.synthesize_note_reference import synthesize_notes, write_pcm16_wav


class SynthesizeNoteReferenceTest(unittest.TestCase):
    def test_preserves_silence_onset_and_timeline(self):
        waveform = synthesize_notes(
            [{"midi": 69, "time": 0.10, "duration": 0.20, "velocity": 0.8}],
            sample_rate=8_000,
            tail_seconds=0.10,
        )
        self.assertEqual(len(waveform), 3_200)
        self.assertTrue(np.allclose(waveform[:790], 0.0))
        self.assertGreater(float(np.max(np.abs(waveform[810:2_300]))), 0.1)
        self.assertLessEqual(float(np.max(np.abs(waveform))), 0.861)

    def test_writes_mono_pcm16_wav_and_creates_parent(self):
        waveform = synthesize_notes(
            [{"midi": 60, "time": 0.0, "duration": 0.12}],
            sample_rate=8_000,
            tail_seconds=0.05,
        )
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "nested" / "reference.wav"
            write_pcm16_wav(destination, waveform, 8_000)
            with wave.open(str(destination), "rb") as source:
                self.assertEqual(source.getnchannels(), 1)
                self.assertEqual(source.getsampwidth(), 2)
                self.assertEqual(source.getframerate(), 8_000)
                self.assertEqual(source.getnframes(), len(waveform))

    def test_rejects_empty_payload(self):
        with self.assertRaisesRegex(ValueError, "no playable piano notes"):
            synthesize_notes([], sample_rate=8_000)


if __name__ == "__main__":
    unittest.main()
