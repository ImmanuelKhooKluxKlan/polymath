import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from ml.training.analyze_piano_mix_balance import (
    SAMPLE_RATE,
    active_mask,
    amplitude_share,
    analyze_mix_balance,
    payload_for_roles,
)


class AnalyzePianoMixBalanceTests(unittest.TestCase):
    def test_role_filter_preserves_payload_metadata(self):
        payload = {
            "title": "example",
            "notes": [
                {"midi": 72, "time": 0, "duration": 0.2, "arrangementRole": "melody"},
                {"midi": 48, "time": 0, "duration": 0.2, "arrangementRole": "bass"},
            ],
        }
        lead = payload_for_roles(payload, {"melody"})
        background = payload_for_roles(payload, {"melody"}, invert=True)
        self.assertEqual(lead["title"], "example")
        self.assertEqual([note["midi"] for note in lead["notes"]], [72])
        self.assertEqual([note["midi"] for note in background["notes"]], [48])

    def test_active_mask_uses_audio_duration_and_release(self):
        payload = {
            "notes": [
                {
                    "midi": 60,
                    "time": 0.1,
                    "duration": 0.1,
                    "audioDuration": 0.5,
                    "velocity": 0.8,
                }
            ]
        }
        mask = active_mask(payload, SAMPLE_RATE * 2, release_seconds=0.2)
        self.assertFalse(mask[round(0.05 * SAMPLE_RATE)])
        self.assertTrue(mask[round(0.6 * SAMPLE_RATE)])
        self.assertFalse(mask[round(0.9 * SAMPLE_RATE)])

    def test_amplitude_share_is_an_intuitive_sixty_forty_ratio(self):
        self.assertAlmostEqual(amplitude_share(0.06, 0.04), 0.6)

    def test_analysis_renders_stems_and_reports_dominant_melody(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = root / "samples"
            samples.mkdir()
            waveform = np.zeros((SAMPLE_RATE, 2), dtype="<i2")
            waveform[: SAMPLE_RATE // 2, :] = 8000
            for filename in ("C3.wav", "C5.wav"):
                with wave.open(str(samples / filename), "wb") as handle:
                    handle.setnchannels(2)
                    handle.setsampwidth(2)
                    handle.setframerate(SAMPLE_RATE)
                    handle.writeframes(waveform.tobytes())
            payload = {
                "notes": [
                    {
                        "midi": 72,
                        "time": 0.1,
                        "audioDuration": 0.4,
                        "velocity": 0.95,
                        "arrangementRole": "melody",
                    },
                    {
                        "midi": 48,
                        "time": 0.1,
                        "audioDuration": 0.4,
                        "velocity": 0.3,
                        "arrangementRole": "bass",
                    },
                ]
            }
            result = analyze_mix_balance(
                payload,
                samples,
                root / "analysis",
                duration_seconds=2.0,
            )
            self.assertGreater(result["melodyActive"]["melodyShare"], 0.6)
            self.assertTrue((root / "analysis" / "01-MELODY-STEM.wav").is_file())
            self.assertTrue((root / "analysis" / "02-ACCOMPANIMENT-STEM.wav").is_file())
            self.assertTrue((root / "analysis" / "03-COMBINED.wav").is_file())
            self.assertTrue((root / "analysis" / "MIX-BALANCE.json").is_file())
            self.assertEqual(
                result["renders"]["melody"]["seconds"],
                result["renders"]["accompaniment"]["seconds"],
            )


if __name__ == "__main__":
    unittest.main()
