import unittest

from ml.training.analyze_pianist_gesture_patterns import (
    analyze,
    gesture_category,
    match_gestures,
)


class PianistGesturePatternTests(unittest.TestCase):
    def test_classifies_exact_octave_subset_and_wrong_gestures(self):
        def group(*midis):
            return [{"midi": midi, "time": 0.0, "duration": 0.2, "velocity": 0.7} for midi in midis]

        self.assertEqual(gesture_category(group(60, 64), group(60, 64)), "exact-key-set")
        self.assertEqual(
            gesture_category(group(60, 64), group(72, 76)),
            "same-pitch-classes-different-octave",
        )
        self.assertEqual(gesture_category(group(60, 64), group(60)), "candidate-subset")
        self.assertEqual(gesture_category(group(60), group(61)), "wrong-chord")

    def test_gesture_match_prefers_pitch_class_over_merely_nearby_wrong_chord(self):
        reference = [[{"midi": 60, "time": 1.0}]]
        candidate = [
            [{"midi": 61, "time": 1.0}],
            [{"midi": 72, "time": 1.04}],
        ]
        matches, missing, extra = match_gestures(reference, candidate, 0.1)
        self.assertEqual(matches, [(0, 1)])
        self.assertEqual(missing, [])
        self.assertEqual(extra, [0])

    def test_hard_cutoffs_exclude_untrusted_tail_and_report_velocity_pattern(self):
        reference = {
            "notes": [
                {"midi": 60, "time": 1.0, "duration": 0.4, "velocity": 0.5},
                {"midi": 64, "time": 1.0, "duration": 0.4, "velocity": 0.5},
                {"midi": 67, "time": 151.0, "duration": 0.4, "velocity": 0.9},
            ]
        }
        source = {
            "notes": [
                {"midi": 60, "time": 2.0, "duration": 0.4, "velocity": 0.8, "instrument": "voice"},
                {"midi": 64, "time": 2.0, "duration": 0.4, "velocity": 0.8, "instrument": "guitar"},
                {"midi": 67, "time": 152.0, "duration": 0.4, "velocity": 0.8, "instrument": "voice"},
            ]
        }
        alignment = {
            "anchors": [
                {"referenceTime": 0.0, "observedTime": 1.0},
                {"referenceTime": 160.0, "observedTime": 161.0},
            ],
            "matches": [
                {
                    "reference": {"midi": 60, "velocity": 0.5},
                    "observed": {"sourceIndex": 0},
                    "exactPitch": True,
                    "coarseResidual": 0.0,
                },
                {
                    "reference": {"midi": 64, "velocity": 0.5},
                    "observed": {"sourceIndex": 1},
                    "exactPitch": True,
                    "coarseResidual": 0.0,
                },
            ],
        }
        candidate = {
            "notes": [
                {
                    "midi": 60,
                    "time": 2.0,
                    "duration": 0.3,
                    "velocity": 0.7,
                    "sourceIndex": 0,
                    "selectionProbability": 0.8,
                },
                {
                    "midi": 67,
                    "time": 152.0,
                    "duration": 0.3,
                    "velocity": 0.9,
                    "sourceIndex": 2,
                },
            ]
        }
        report = analyze(
            reference,
            source,
            alignment,
            candidate,
            reference_end=150.0,
            candidate_end=150.0,
        )
        self.assertEqual(report["notes"]["reference"], 2)
        self.assertEqual(report["notes"]["candidate"], 1)
        self.assertEqual(report["gestures"]["reference"], 1)
        self.assertEqual(report["sourceDecisions"]["supportedButIgnored"], 1)
        self.assertAlmostEqual(report["gestures"]["pairedVelocityError"]["mean"], 0.2)
        self.assertAlmostEqual(report["gestures"]["pairedDurationError"]["mean"], -0.1)
        self.assertAlmostEqual(
            report["gestures"]["pairedAudioDurationError"]["mean"], -0.1
        )
        self.assertAlmostEqual(report["gestures"]["pairedOnsetError"]["mae"], 0.0)
        self.assertEqual(
            report["gestures"]["byReferenceDurationBand"][0]["name"],
            "medium-0.18-0.54s",
        )
        self.assertEqual(report["gestures"]["byReferenceWindow5s"][0]["name"], "000-005s")
        self.assertAlmostEqual(
            report["gestures"]["byReferenceWindow5s"][0]["referenceMeanVelocity"],
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
