import unittest

from ml.training.analyze_aligned_source_decisions import analyze


class AlignedSourceDecisionAuditTests(unittest.TestCase):
    def test_separates_retained_ignored_and_unsupported_source_events(self):
        source = {
            "notes": [
                {"midi": 48, "time": 1.0, "duration": 0.3, "velocity": 0.4, "instrument": "bass"},
                {"midi": 60, "time": 1.0, "duration": 0.3, "velocity": 0.8, "instrument": "guitar"},
                {"midi": 64, "time": 2.0, "duration": 0.2, "velocity": 0.7, "instrument": "guitar"},
            ]
        }
        alignment = {
            "qualityWindows": [{"status": "trusted", "sourceStart": 0, "sourceEnd": 3}],
            "matches": [
                {
                    "observed": {"sourceIndex": 0},
                    "reference": {"midi": 48, "velocity": 0.5},
                    "exactPitch": True,
                    "coarseResidual": 0.01,
                },
                {
                    "observed": {"sourceIndex": 1},
                    "reference": {"midi": 60, "velocity": 0.9},
                    "exactPitch": True,
                    "coarseResidual": 0.02,
                },
            ],
        }
        candidate = {
            "notes": [
                {"sourceIndex": 0, "midi": 60, "time": 1.0, "velocity": 0.6, "sourceVelocityBeforeArrangement": 0.4},
                {"sourceIndex": 2, "midi": 76, "time": 2.0, "velocity": 0.8, "sourceVelocityBeforeArrangement": 0.7},
            ]
        }

        result = analyze(source, alignment, candidate, reference_transpose=12)

        self.assertEqual(result["counts"]["alignmentSupportedSourceNotes"], 2)
        self.assertEqual(result["counts"]["supportedAndSelected"], 1)
        self.assertEqual(result["counts"]["supportedButIgnored"], 1)
        self.assertEqual(result["counts"]["selectedWithoutAlignmentSupport"], 1)
        self.assertEqual(result["counts"]["selectedWithExactTargetPitch"], 1)
        self.assertEqual(result["velocity"]["candidateRaisedAtLeast010"], 2)


if __name__ == "__main__":
    unittest.main()
