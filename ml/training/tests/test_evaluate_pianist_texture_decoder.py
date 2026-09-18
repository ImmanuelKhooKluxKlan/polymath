import unittest

import numpy as np

from ml.training.evaluate_pianist_texture_decoder import top_share_metrics


class PianistTextureDecoderEvaluationTests(unittest.TestCase):
    def test_top_share_selects_the_requested_number_of_highest_scores(self):
        labels = np.asarray([0.0, 1.0, 1.0, 0.0])
        probabilities = np.asarray([0.1, 0.9, 0.8, 0.2])

        result = top_share_metrics(labels, probabilities, 0.5)

        self.assertEqual(result["predictedPositive"], 2)
        self.assertEqual(result["truePositive"], 2)
        self.assertEqual(result["precision"], 1.0)


if __name__ == "__main__":
    unittest.main()
