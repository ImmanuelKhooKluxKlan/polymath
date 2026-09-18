import unittest

import numpy as np

from ml.training.fit_pianist_onset_phase_router import (
    CLASSES,
    FEATURE_NAMES,
    route,
)


class PianistOnsetPhaseRouterTests(unittest.TestCase):
    def test_feature_contract_has_unique_names(self):
        self.assertEqual(len(FEATURE_NAMES), len(set(FEATURE_NAMES)))
        self.assertGreater(len(FEATURE_NAMES), 40)

    def test_route_abstains_below_threshold(self):
        values = np.asarray([[0.60, 0.20, 0.10]])
        self.assertEqual(route(values, threshold=0.70, margin=0.10).tolist(), [0])

    def test_route_keeps_explicit_zero_as_no_change(self):
        zero_index = CLASSES.index(0)
        values = np.zeros((1, len(CLASSES)))
        values[0, zero_index] = 0.99
        self.assertEqual(route(values, threshold=0.50, margin=0.10).tolist(), [0])


if __name__ == "__main__":
    unittest.main()
