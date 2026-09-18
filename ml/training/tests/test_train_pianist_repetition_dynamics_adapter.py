import unittest

from ml.training.train_pianist_repetition_dynamics_adapter import (
    apply_profile,
    dynamics_gate,
)


def group(time, pitches, velocity):
    return [
        {
            "time": time,
            "midi": midi,
            "duration": 0.2,
            "scoreDuration": 0.2,
            "velocity": velocity,
            "hand": "left" if midi < 60 else "right",
        }
        for midi in pitches
    ]


class PianistRepetitionDynamicsAdapterTest(unittest.TestCase):
    def test_only_destination_gesture_velocity_changes(self):
        candidate = {
            "notes": [
                note
                for item in (
                    group(1.0, [48, 64], 0.6),
                    group(1.5, [55], 0.6),
                    group(5.0, [48, 64], 0.6),
                    group(5.5, [55], 0.6),
                )
                for note in item
            ]
        }
        profile = {
            "id": "touch-test",
            "profileSha256": "touch-sha",
            "template": {
                "startSeconds": 1.0,
                "endSeconds": 2.0,
                "gestures": [
                    {
                        "relativeTime": 0.0,
                        "midis": [48, 64],
                        "velocity": 0.4,
                    },
                    {
                        "relativeTime": 0.5,
                        "midis": [55],
                        "velocity": 0.8,
                    },
                ],
            },
            "detection": {
                "minimumOffsetSeconds": 3.5,
                "maximumOffsetSeconds": 4.5,
                "coarseStepSeconds": 0.1,
                "fineStepSeconds": 0.01,
                "maximumTimeDistanceSeconds": 0.35,
                "gapCost": 0.85,
                "minimumMeanPitchClassF1": 0.99,
                "maximumNormalizedScore": 0.05,
            },
            "application": {
                "velocityBlend": 1.0,
                "energyBlend": 0.0,
                "minimumEnergyScale": 0.75,
                "maximumEnergyScale": 1.25,
            },
        }
        output, diagnostics = apply_profile(candidate, profile)
        self.assertTrue(diagnostics["applied"])
        self.assertEqual(
            [note["velocity"] for note in output["notes"]],
            [0.6, 0.6, 0.6, 0.4, 0.4, 0.8],
        )
        self.assertEqual(
            [(note["time"], note["midi"], note["duration"]) for note in output["notes"]],
            [(note["time"], note["midi"], note["duration"]) for note in candidate["notes"]],
        )

    def test_gate_requires_velocity_gain_with_identical_structure(self):
        structural = {
            "referenceGestureRecall": 1.0,
            "candidateGesturePrecision": 1.0,
            "coverageAdjustedPitchClassF1": 1.0,
            "coverageAdjustedExactPitchClassRate": 1.0,
            "coverageAdjustedOccupancyAccuracy": 1.0,
            "onsetMaeSeconds": 0.03,
            "exactKeyDurationMaeSeconds": 0.05,
            "sequenceScore": 1.0,
            "gestureVelocityMae": 0.10,
        }
        candidate_local = {**structural, "gestureVelocityMae": 0.04}
        candidate_full = {**structural, "gestureVelocityMae": 0.09}
        passed, reason = dynamics_gate(
            structural, candidate_local, structural, candidate_full
        )
        self.assertTrue(passed)
        self.assertEqual(reason, "safe-dynamics-refinement")


if __name__ == "__main__":
    unittest.main()
