from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from ml.training.prepare_stem_views import prepare_views


class PrepareStemViewsTests(unittest.TestCase):
    def test_prepares_synchronized_stereo_views(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample_rate = 44_100
            timeline = np.arange(sample_rate, dtype=np.float32) / sample_rate
            tone = np.sin(2 * np.pi * 440 * timeline).astype(np.float32)
            stereo = np.stack((tone, tone * 0.5), axis=1)
            first = root / "vocals.wav"
            second = root / "no_vocals.wav"
            sf.write(first, stereo, sample_rate, subtype="PCM_16")
            sf.write(second, stereo * 0.25, sample_rate, subtype="PCM_16")

            report = prepare_views([first, second], root / "prepared")

            self.assertEqual(len(report["views"]), 2)
            self.assertLessEqual(report["clockSkewSeconds"], 0.002)
            frames = {view["frames"] for view in report["views"]}
            self.assertEqual(len(frames), 1)
            for view in report["views"]:
                info = sf.info(view["output"])
                self.assertEqual(info.samplerate, 16_000)
                self.assertEqual(info.channels, 1)

    def test_rejects_clock_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.wav"
            second = root / "second.wav"
            sf.write(first, np.zeros(16_000, dtype=np.float32), 16_000)
            sf.write(second, np.zeros(16_100, dtype=np.float32), 16_000)

            with self.assertRaisesRegex(ValueError, "do not share one clock"):
                prepare_views([first, second], root / "prepared")


if __name__ == "__main__":
    unittest.main()
