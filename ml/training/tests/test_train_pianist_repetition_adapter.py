import unittest

from ml.training.train_pianist_repetition_adapter import (
    apply_profile,
    detect_repeat,
    evaluation_gate,
    infer_transposition,
    map_time,
    monotonic_anchors,
)


def group(time, pitches, velocity=0.7):
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


class PianistRepetitionAdapterTest(unittest.TestCase):
    def test_gate_accepts_small_exact_set_fix_when_every_safety_metric_holds(self):
        baseline = {
            "referenceGestureRecall": 1.0,
            "candidateGesturePrecision": 0.875,
            "coverageAdjustedPitchClassF1": 0.971,
            "coverageAdjustedExactPitchClassRate": 0.857,
            "coverageAdjustedOccupancyAccuracy": 1.0,
            "onsetMaeSeconds": 0.053,
            "gestureVelocityMae": 0.030,
            "exactKeyDurationMaeSeconds": 0.100,
            "sequenceScore": 1.38,
        }
        candidate = {
            **baseline,
            "candidateGesturePrecision": 0.875,
            "coverageAdjustedPitchClassF1": 1.0,
            "coverageAdjustedExactPitchClassRate": 1.0,
            "onsetMaeSeconds": 0.017,
            "exactKeyDurationMaeSeconds": 0.089,
            "sequenceScore": 0.94,
        }
        passed, reason = evaluation_gate(baseline, candidate)
        self.assertTrue(passed)
        self.assertEqual(reason, "safe-structural-refinement")

    def test_gate_rejects_small_fix_that_loses_precision(self):
        baseline = {
            "referenceGestureRecall": 1.0,
            "candidateGesturePrecision": 1.0,
            "coverageAdjustedPitchClassF1": 0.90,
            "coverageAdjustedExactPitchClassRate": 0.80,
            "coverageAdjustedOccupancyAccuracy": 1.0,
            "onsetMaeSeconds": 0.05,
            "gestureVelocityMae": 0.03,
            "exactKeyDurationMaeSeconds": 0.10,
            "sequenceScore": 1.0,
        }
        candidate = {
            **baseline,
            "candidateGesturePrecision": 0.90,
            "coverageAdjustedPitchClassF1": 0.92,
            "coverageAdjustedExactPitchClassRate": 0.90,
            "sequenceScore": 0.80,
        }
        passed, reason = evaluation_gate(baseline, candidate)
        self.assertFalse(passed)
        self.assertEqual(reason, "gate-not-met")

    def test_detects_repeat_from_input_sequence(self):
        first = [
            group(1.0, [48, 64]),
            group(1.4, [55]),
            group(1.8, [60, 67]),
            group(2.2, [55]),
        ]
        repeated = [
            group(6.0, [48, 64]),
            group(6.4, [55]),
            group(6.8, [60, 67]),
            group(7.2, [55]),
        ]
        best, ranked = detect_repeat(
            first + repeated,
            template_start=1.0,
            template_end=2.5,
            minimum_offset=4.5,
            maximum_offset=5.5,
        )
        self.assertAlmostEqual(best["offsetSeconds"], 5.0, places=2)
        offsets = [round(float(row["offsetSeconds"]), 6) for row in ranked]
        self.assertEqual(len(offsets), len(set(offsets)))

    def test_infers_pitch_class_transposition(self):
        first = [group(0.0, [48, 64]), group(0.5, [55, 67])]
        repeated = [group(5.0, [50, 66]), group(5.5, [57, 69])]
        shift, _ = infer_transposition(first, repeated, [(0, 0), (1, 1)])
        self.assertEqual(shift, 2)

    def test_piecewise_time_map_uses_input_warp(self):
        first = [group(1.0, [48]), group(1.5, [55]), group(2.0, [60])]
        repeated = [group(6.0, [48]), group(6.6, [55]), group(7.2, [60])]
        anchors = monotonic_anchors(
            first,
            repeated,
            [(0, 0), (1, 1), (2, 2)],
            template_start=1.0,
            template_end=2.5,
            offset=5.0,
        )
        self.assertAlmostEqual(map_time(1.75, anchors), 6.9, places=6)

    def test_low_similarity_repeat_is_not_applied(self):
        candidate = {
            "notes": [
                note
                for item in (
                    group(1.0, [48, 64]),
                    group(1.4, [55]),
                    group(1.8, [60, 67]),
                    group(2.2, [55]),
                    group(6.0, [49, 61]),
                    group(6.4, [54, 66]),
                    group(6.8, [58, 70]),
                    group(7.2, [51, 63]),
                )
                for note in item
            ]
        }
        profile = {
            "id": "guard-test",
            "profileSha256": "guard-sha",
            "template": {
                "startSeconds": 1.0,
                "endSeconds": 2.5,
                "notes": [
                    {
                        "relativeTime": 0.0,
                        "midi": 48,
                        "duration": 0.2,
                        "velocity": 0.7,
                        "hand": "left",
                    }
                ],
            },
            "detection": {
                "minimumOffsetSeconds": 4.5,
                "maximumOffsetSeconds": 5.5,
                "coarseStepSeconds": 0.1,
                "fineStepSeconds": 0.01,
                "maximumTimeDistanceSeconds": 0.35,
                "gapCost": 0.85,
                "minimumMeanPitchClassF1": 0.95,
                "maximumNormalizedScore": 0.05,
            },
            "application": {"timeWarpBlend": 1.0, "velocityScaleBlend": 1.0},
        }
        output, diagnostics = apply_profile(candidate, profile)
        self.assertFalse(diagnostics["applied"])
        self.assertEqual(output["notes"], candidate["notes"])

    def test_disabled_timing_gate_still_reports_failed_timing_evidence(self):
        candidate = {
            "notes": [
                note
                for item in (
                    group(1.0, [48]),
                    group(1.4, [55]),
                    group(1.8, [60]),
                    group(2.2, [55]),
                    group(6.0, [49]),
                    group(6.4, [54]),
                    group(6.8, [58]),
                    group(7.2, [51]),
                )
                for note in item
            ]
        }
        profile = {
            "id": "diagnostic-test",
            "profileSha256": "diagnostic-sha",
            "template": {
                "startSeconds": 1.0,
                "endSeconds": 2.5,
                "notes": [
                    {
                        "relativeTime": 0.0,
                        "midi": 48,
                        "duration": 0.2,
                        "velocity": 0.7,
                        "hand": "left",
                    }
                ],
            },
            "detection": {
                "minimumOffsetSeconds": 4.5,
                "maximumOffsetSeconds": 5.5,
                "coarseStepSeconds": 0.1,
                "fineStepSeconds": 0.01,
                "maximumTimeDistanceSeconds": 0.35,
                "gapCost": 0.85,
                "minimumMeanPitchClassF1": 0.0,
                "maximumNormalizedScore": 10.0,
                "minimumTimingMeanPitchClassF1": 1.1,
                "maximumTimingNormalizedScore": -1.0,
                "requireTimingConfidence": False,
            },
            "application": {"timeWarpBlend": 1.0, "velocityScaleBlend": 0.0},
        }
        _, diagnostics = apply_profile(candidate, profile)
        self.assertTrue(diagnostics["applied"])
        self.assertFalse(diagnostics["timingConfidenceRequired"])
        self.assertFalse(diagnostics["timingEvidencePassed"])
        self.assertTrue(diagnostics["timingConfidencePassed"])

    def test_snapped_first_gesture_removes_stale_destination_chord(self):
        candidate = {
            "notes": [
                note
                for item in (
                    group(1.0, [48]),
                    group(1.4, [55]),
                    group(1.8, [60]),
                    group(2.2, [55]),
                    # Timing input is 10 ms early at the destination.  Before
                    # the gesture-boundary guard, MIDI 49 survived and was
                    # layered into the generated chord at the same onset.
                    group(5.99, [49]),
                    group(6.4, [55]),
                    group(6.8, [60]),
                    group(7.2, [55]),
                )
                for note in item
            ]
        }
        detection = {
            "notes": [
                note
                for item in (
                    group(1.0, [48]),
                    group(1.4, [55]),
                    group(1.8, [60]),
                    group(2.2, [55]),
                    group(6.0, [48]),
                    group(6.4, [55]),
                    group(6.8, [60]),
                    group(7.2, [55]),
                )
                for note in item
            ]
        }
        profile = {
            "id": "gesture-boundary-test",
            "profileSha256": "gesture-boundary-sha",
            "template": {
                "startSeconds": 1.0,
                "endSeconds": 2.5,
                "notes": [
                    {
                        "relativeTime": 0.0,
                        "midi": 64,
                        "duration": 0.2,
                        "scoreDuration": 0.2,
                        "velocity": 0.8,
                        "hand": "right",
                    }
                ],
            },
            "detection": {
                "minimumOffsetSeconds": 4.5,
                "maximumOffsetSeconds": 5.5,
                "coarseStepSeconds": 0.1,
                "fineStepSeconds": 0.01,
                "maximumTimeDistanceSeconds": 0.35,
                "gapCost": 0.85,
                "minimumMeanPitchClassF1": 0.0,
                "maximumNormalizedScore": 10.0,
                "minimumTimingMeanPitchClassF1": 0.0,
                "maximumTimingNormalizedScore": 10.0,
                "requireTimingConfidence": False,
            },
            "application": {
                "timeWarpBlend": 0.0,
                "velocityScaleBlend": 0.0,
                "onsetSnapBlend": 1.0,
            },
        }
        output, diagnostics = apply_profile(
            candidate,
            profile,
            detection_payload=detection,
            timing_payload=candidate,
        )
        destination = [
            note for note in output["notes"] if 5.95 <= float(note["time"]) < 6.05
        ]
        self.assertEqual([note["midi"] for note in destination], [64])
        self.assertAlmostEqual(float(destination[0]["duration"]), 0.2, places=6)
        self.assertLess(diagnostics["replacementStartSeconds"], 5.99)
        self.assertEqual(diagnostics["replacementOnsetGuardSeconds"], 0.035)


if __name__ == "__main__":
    unittest.main()
