import unittest

import numpy as np

from ml.training.render_reference_audio import normalized_notes, render_notes


class ReferenceAudioRendererTests(unittest.TestCase):
    def test_renders_finite_non_silent_reference_clock(self):
        notes = normalized_notes(
            {
                "notes": [
                    {"midi": 60, "time": 0.0, "duration": 0.5, "velocity": 0.8},
                    {"midi": 64, "time": 0.5, "duration": 0.5, "velocity": 0.7},
                    {"midi": 67, "time": 1.0, "duration": 0.5, "velocity": 0.9},
                ]
            }
        )
        rendered = render_notes(notes, sample_rate=8000)
        self.assertGreater(len(rendered), 1.5 * 8000)
        self.assertTrue(np.isfinite(rendered).all())
        self.assertGreater(float(np.max(np.abs(rendered))), 0.1)
        self.assertLessEqual(float(np.max(np.abs(rendered))), 0.921)


if __name__ == "__main__":
    unittest.main()
