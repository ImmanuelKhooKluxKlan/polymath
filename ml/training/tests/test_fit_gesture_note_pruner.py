import unittest

from ml.training.fit_gesture_note_pruner import metrics


class GestureNotePrunerTrainingTests(unittest.TestCase):
    def test_metrics_reward_removing_an_extra_pitch_class(self):
        group = {
            "songId": "song",
            "candidateGroup": [
                {"midi": 60, "arrangementRole": "harmony"},
                {"midi": 61, "arrangementRole": "harmony"},
            ],
            "referencePitchClasses": [0],
            "features": [[1.0], [1.0]],
            "labels": [True, False],
        }
        model = {
            "featureNames": ["bias"],
            "weights": [0.0],
            "means": [0.0],
            "scales": [1.0],
        }
        # Override probabilities deterministically via two distinct bias rows.
        group["features"] = [[5.0], [-5.0]]
        model["weights"] = [1.0]
        policy = {
            "threshold": 0.5,
            "minimumProbabilityGap": 0.1,
            "maximumRemovalsPerGesture": 1,
            "minimumGroupSize": 2,
            "minimumRemainingNotes": 1,
            "preserveMelody": True,
        }

        baseline = metrics([group])
        candidate = metrics([group], model, policy)

        self.assertGreater(candidate["f1"], baseline["f1"])
        self.assertEqual(candidate["wrongRemovals"], 0)
        self.assertEqual(candidate["exactPitchClassSets"], 1)


if __name__ == "__main__":
    unittest.main()
