import unittest

import numpy as np

from ml.training.fit_pianist_pitch_class_decoder import (
    FEATURE_NAMES,
    decoded_set_metrics,
    examples_from_song,
)


def cell(candidate, source, reference):
    return {
        "candidatePitchClasses": candidate,
        "sourcePitchClasses": source,
        "referencePitchClasses": [reference] if reference else [],
        "candidateChordSize": len(candidate),
        "sourcePitchClassCount": len(source),
        "previousGapSeconds": 0.4,
        "nextGapSeconds": 0.4,
        "candidateDuration": 0.5,
        "candidateVelocity": 0.6,
        "sourceNoteCount": len(source),
        "sourceAttackNoteCount": len(source),
        "candidateRoleSignature": "harmony",
    }


class PitchClassDecoderTest(unittest.TestCase):
    def test_feature_vector_is_transposition_invariant(self):
        original = {
            "id": "a",
            "cells": [cell([0, 4], [0, 4, 7], [0, 4, 7])],
        }
        shifted = {
            "id": "b",
            "cells": [cell([5, 9], [0, 5, 9], [0, 5, 9])],
        }
        left = examples_from_song(original)
        right = examples_from_song(shifted)
        self.assertEqual(len(left), 12)
        self.assertEqual(len(left[0]["features"]), len(FEATURE_NAMES))
        for pitch_class in range(12):
            shifted_pc = (pitch_class + 5) % 12
            np.testing.assert_allclose(
                left[pitch_class]["features"],
                right[shifted_pc]["features"],
            )
            self.assertEqual(left[pitch_class]["label"], right[shifted_pc]["label"])

    def test_hybrid_decoder_only_adds_source_supported_pitch_classes(self):
        song = {
            "id": "test",
            "cells": [cell([0], [0, 4], [0, 4])],
        }
        examples = examples_from_song(song)
        probabilities = np.zeros(12, dtype=float)
        probabilities[0] = 0.9
        probabilities[4] = 0.9
        probabilities[7] = 0.99
        metrics = decoded_set_metrics(
            examples,
            probabilities,
            add_threshold=0.5,
            remove_threshold=0.1,
        )
        self.assertEqual(metrics["additions"], 1)
        self.assertEqual(metrics["removals"], 0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["exactSetRate"], 1.0)

    def test_hybrid_decoder_can_remove_a_confident_false_positive(self):
        song = {
            "id": "test",
            "cells": [cell([0, 7], [0, 7], [0])],
        }
        examples = examples_from_song(song)
        probabilities = np.full(12, 0.5, dtype=float)
        probabilities[0] = 0.9
        probabilities[7] = 0.01
        metrics = decoded_set_metrics(
            examples,
            probabilities,
            add_threshold=0.9,
            remove_threshold=0.1,
        )
        self.assertEqual(metrics["additions"], 0)
        self.assertEqual(metrics["removals"], 1)
        self.assertEqual(metrics["exactSetRate"], 1.0)


if __name__ == "__main__":
    unittest.main()
