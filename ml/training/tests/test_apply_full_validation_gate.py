import unittest

from ml.training.apply_full_validation_gate import evaluate_gate


def timeline(f1=0.91, precision=0.92, recall=0.90, frame=0.56, duration=0.34):
    return {
        "schema": "polymath-stitched-song-timeline-score-v1",
        "songIds": ["song"],
        "metrics": {
            "100ms": {"microF1": f1, "precision": precision, "recall": recall},
        },
        "perSong": {"song": {"100ms": {"microF1": f1}}},
        "songClusterBootstrap95": {"100ms": {"lower95": f1 - 0.01}},
        "diagnostics100ms": {
            "matchedNotes": 900,
            "predictedNotes": 1000,
            "cutOffNotes": 40,
            "overlongNotes": 60,
            "rapidRetriggers": 3,
            "onsetAndOffset": {"f1": duration},
            "frame": {"f1": frame},
        },
    }


def gate():
    return {
        "schema": "polymath-phase94-full-validation-gate-v1",
        "candidateVersion": "test",
        "requiredChecks": {
            "microF1_100msMinimum": 0.90,
            "precision100msMaximumRegression": 0.001,
            "recall100msMaximumRegression": 0.001,
            "maximumPerSongF1Regression": 0.005,
            "onsetAndOffsetF1MaximumRegression": 0.002,
            "frameF1MaximumRegression": 0.002,
            "cutOffPer1000MatchedMaximumIncrease": 2.0,
            "overlongPer1000MatchedMaximumIncrease": 2.0,
            "rapidRetriggersPer1000PredictedMaximumIncrease": 0.25,
        },
        "certificationCheck": {"songBootstrapLower95_100msMinimum": 0.90},
    }


class FullValidationGateTests(unittest.TestCase):
    def test_small_clean_improvement_passes(self):
        baseline = timeline()
        candidate = timeline(f1=0.912, precision=0.921, recall=0.902, frame=0.561, duration=0.341)
        result = evaluate_gate(baseline, candidate, gate())
        self.assertTrue(result["researchGatePassed"])
        self.assertTrue(result["certification"]["passed"])

    def test_onset_gain_cannot_hide_duration_regression(self):
        baseline = timeline()
        candidate = timeline(f1=0.912, precision=0.921, recall=0.902, frame=0.55, duration=0.33)
        result = evaluate_gate(baseline, candidate, gate())
        self.assertFalse(result["researchGatePassed"])
        self.assertIn("frame_f1", result["failedChecks"])
        self.assertIn("onset_and_offset_f1", result["failedChecks"])


if __name__ == "__main__":
    unittest.main()
