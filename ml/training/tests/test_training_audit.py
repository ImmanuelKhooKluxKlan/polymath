import tempfile
import unittest
import wave
from pathlib import Path

from ml.training.train_muscriptor_piano import (
    TrainingError,
    audit_records,
    conditioning_group,
)


class TrainingAuditTests(unittest.TestCase):
    def test_conditioning_mode_matches_route_contract(self):
        self.assertEqual(conditioning_group("acoustic_piano", "instrument"), "0")
        self.assertIsNone(conditioning_group("acoustic_piano", "unconditioned"))
        with self.assertRaises(TrainingError):
            conditioning_group("acoustic_piano", "mismatched")

    def test_composed_manifest_can_preflight_through_local_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "clip.wav"
            with wave.open(str(audio), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(b"\x00\x00" * 1600)
            audit = audit_records(
                [
                    {
                        "clipId": "local-audit-001",
                        "songId": "song-001",
                        "audioClip": "/runpod-volume/training/missing.wav",
                        "localAudioSource": str(audio),
                        "durationSeconds": 0.1,
                        "instrumentFocus": "acoustic_piano",
                        "notes": [
                            {
                                "midi": 60,
                                "time": 0.0,
                                "duration": 0.1,
                                "instrument": "acoustic_piano",
                            }
                        ],
                    }
                ]
            )
            self.assertEqual(audit["clips"], 1)
            self.assertEqual(audit["songs"], 1)


if __name__ == "__main__":
    unittest.main()
