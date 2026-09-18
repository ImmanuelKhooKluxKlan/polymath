import unittest

from ml.training.analyze_chord_retrieval_decisions import summarize


class ChordRetrievalDecisionAuditTests(unittest.TestCase):
    def test_summary_separates_helpful_and_harmful_replacements(self):
        result = summarize(
            [
                {"pitchClassF1Delta": 0.5},
                {"pitchClassF1Delta": 0.0},
                {"pitchClassF1Delta": -0.25},
            ]
        )
        self.assertEqual(result["improved"], 1)
        self.assertEqual(result["unchanged"], 1)
        self.assertEqual(result["worsened"], 1)
        self.assertAlmostEqual(result["netImprovement"], 0.25)


if __name__ == "__main__":
    unittest.main()
