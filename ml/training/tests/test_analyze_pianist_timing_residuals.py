import unittest

from ml.training.analyze_pianist_timing_residuals import quantile, summarize


class PianistTimingResidualTests(unittest.TestCase):
    def test_quantile_interpolates(self):
        self.assertAlmostEqual(quantile([0.0, 0.1, 0.2], 0.25), 0.05)

    def test_summary_reports_milliseconds(self):
        result = summarize([-0.1, 0.0, 0.1])
        self.assertEqual(result["matches"], 3)
        self.assertEqual(result["medianMs"], 0.0)
        self.assertEqual(result["p10Ms"], -80.0)
        self.assertEqual(result["p90Ms"], 80.0)


if __name__ == "__main__":
    unittest.main()
