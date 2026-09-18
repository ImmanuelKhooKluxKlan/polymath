import unittest

from ml.training.train_pianist_intro_motif import (
    apply_intro_motif,
    evaluation,
    infer_grid,
    infer_intro_register_application,
    infer_pre_voice_cadence,
    nearest_octave_shift,
    repeating_prefix,
    source_cycle_transition_anchors,
    source_moving_pitch,
)


def group(time, state):
    notes = []
    if state in {"left-only", "both"}:
        notes.append({"time": time, "midi": 48, "hand": "left", "duration": 0.1})
    if state in {"right-only", "both"}:
        notes.append({"time": time, "midi": 72, "hand": "right", "duration": 0.1})
    return notes


class PianistIntroMotifTest(unittest.TestCase):
    def test_learns_partial_final_cycle_as_relative_pre_voice_cadence(self):
        pattern = ["both", "left-only"]
        reference = [
            group(index * 0.2, state)
            for index, state in enumerate(pattern * 3)
        ]
        reference.extend(
            [
                [
                    {"time": 1.2, "midi": 48, "hand": "left", "duration": 0.4, "velocity": 0.7},
                    {"time": 1.2, "midi": 55, "hand": "right", "duration": 0.2, "velocity": 0.7},
                ]
            ]
        )
        source_notes = []
        for time in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2):
            for midi in (48, 55, 60):
                source_notes.append(
                    {"time": time, "midi": midi, "instrument": "guitar", "duration": 0.2}
                )
        source_notes.append(
            {"time": 1.6, "midi": 72, "instrument": "voice", "duration": 0.2}
        )

        cadence = infer_pre_voice_cadence(
            reference,
            {"notes": source_notes},
            {"periodGestures": 2, "cycles": 3},
            {"slots": [0, 2], "pulseSeconds": 0.2},
        )

        self.assertTrue(cadence["enabled"])
        self.assertEqual(cadence["replaceFromGridSlot"], 0)
        self.assertEqual(
            [item["intervalFromRoot"] for item in cadence["notes"]], [0, 7]
        )
        self.assertEqual(cadence["rootSourceOctaveShift"], 0)

    def test_evaluation_counts_missing_gestures_in_coverage_scores(self):
        reference = [group(0.0, "right-only"), group(1.0, "right-only")]
        candidate = [group(0.0, "right-only")]
        result = evaluation(reference, candidate)
        self.assertEqual(result["meanPitchClassF1"], 1.0)
        self.assertEqual(result["referenceGestureRecall"], 0.5)
        self.assertEqual(result["coverageAdjustedPitchClassF1"], 0.5)
        self.assertEqual(result["coverageAdjustedExactPitchClassRate"], 0.5)

    def test_finds_shortest_high_coverage_repeated_prefix(self):
        pattern = ["both", "left-only", "both", "right-only", "left-only"]
        groups = [group(index * 0.1, state) for index, state in enumerate(pattern * 4)]
        result = repeating_prefix(groups, minimum_period=4, maximum_period=10)
        self.assertEqual(result["periodGestures"], 5)
        self.assertEqual(result["cycles"], 4)
        self.assertEqual(result["agreement"], 1.0)

    def test_infers_sixteen_slot_clock(self):
        slots = [0, 1, 2, 4, 6, 7, 9, 10, 12, 14, 15]
        groups = []
        for cycle in range(4):
            for slot in slots:
                groups.append(group(cycle * 2.4 + slot * 0.15, "both" if slot % 2 else "left-only"))
        result = infer_grid(groups, len(slots), 4)
        self.assertEqual(result["subdivisions"], 16)
        self.assertEqual(result["slots"], slots)

    def test_moving_pitch_rejects_stable_chord_members(self):
        observed = [
            {"midi": 51, "instrument": "clean_electric_guitar"},
            {"midi": 55, "instrument": "clean_electric_guitar"},
            {"midi": 61, "instrument": "clean_electric_guitar"},
            {"midi": 63, "instrument": "clean_electric_guitar"},
            {"midi": 67, "instrument": "clean_electric_guitar"},
        ]
        self.assertEqual(
            source_moving_pitch(
                observed,
                bass_midi=51,
                top_midi=67,
                stable_pitch_classes={3, 7, 10},
            ),
            61,
        )

    def test_register_calibration_undoes_stale_upward_octave_shift(self):
        self.assertEqual(nearest_octave_shift(51, 39), -12)
        self.assertEqual(nearest_octave_shift(63, 62), 0)
        source = {
            "notes": [
                {
                    "time": 4.0,
                    "midi": midi,
                    "instrument": "clean_electric_guitar",
                }
                for midi in (51, 58, 63, 67)
            ]
        }
        application = infer_intro_register_application(
            source,
            [
                {
                    "stableNotes": [
                        {"intervalFromBass": 0},
                        {"intervalFromBass": 28},
                    ]
                }
            ],
            {
                "bassMidiByCycle": [39, 39, 39],
                "movingIntervalByCycle": [24, 23, 22],
            },
        )
        self.assertEqual(application["bassSourceOctaveShift"], -12)
        self.assertEqual(application["movingSourceOctaveShift"], 0)

    def test_cycle_anchors_follow_nearest_source_voice_transition(self):
        groups = []
        for time, moving in (
            (0.0, 64),
            (0.4, 64),
            (1.8, 63),
            (2.1, 63),
            (4.2, 62),
            (6.0, 64),
        ):
            groups.append(
                [
                    {"time": time, "midi": midi, "instrument": "guitar"}
                    for midi in (48, 55, moving, 67)
                ]
            )
        anchors = source_cycle_transition_anchors(
            groups,
            start=0.0,
            cutoff=5.9,
            cycle_seconds=2.0,
            cycles=3,
            source_bass_midi=48,
            top_source_midi=67,
            stable_pitch_classes={0, 7},
        )
        self.assertEqual(anchors, [0.0, 1.8, 4.2, 6.0])

    def test_application_preserves_learned_phrase_velocity(self):
        source_notes = []
        for time in (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05):
            for midi in (48, 55, 60, 64):
                source_notes.append(
                    {
                        "time": time,
                        "midi": midi,
                        "duration": 0.15,
                        "velocity": 0.2,
                        "instrument": "clean_electric_guitar",
                    }
                )
        source_notes.append(
            {
                "time": 1.20,
                "midi": 72,
                "duration": 0.2,
                "velocity": 0.9,
                "instrument": "voice",
            }
        )
        profile = {
            "id": "test-profile",
            "profileSha256": "test-sha",
            "clock": {"subdivisions": 4, "attackedSlots": [0]},
            "templates": [
                {
                    "velocityMedian": 0.73,
                    "stableNotes": [
                        {
                            "intervalFromBass": 0,
                            "hand": "left",
                            "durationPulses": 1.0,
                        }
                    ],
                }
            ],
            "application": {
                "bassSourceOctaveShift": -12,
                "movingSourceOctaveShift": 12,
            },
        }
        output, _ = apply_intro_motif(
            {"notes": source_notes},
            {"notes": [], "pianoArrangement": {}},
            profile,
        )
        self.assertTrue(output["notes"])
        self.assertEqual(min(note["midi"] for note in output["notes"]), 36)
        self.assertTrue(
            all(note["velocity"] == 0.73 for note in output["notes"])
        )

    def test_application_keeps_warped_cycle_boundaries_from_colliding(self):
        source_notes = []
        for time, moving in (
            (0.0, 60),
            (0.15, 60),
            (0.30, 60),
            (0.45, 60),
            (0.60, 60),
            (0.75, 60),
            (0.90, 60),
            (1.05, 60),
            (1.55, 59),
            (1.70, 59),
            (1.85, 59),
            (2.00, 59),
        ):
            for midi in (48, 55, moving, 64):
                source_notes.append(
                    {
                        "time": time,
                        "midi": midi,
                        "duration": 0.15,
                        "velocity": 0.5,
                        "instrument": "clean_electric_guitar",
                    }
                )
        source_notes.append(
            {
                "time": 3.2,
                "midi": 72,
                "duration": 0.2,
                "velocity": 0.9,
                "instrument": "voice",
            }
        )
        profile = {
            "id": "boundary-test",
            "profileSha256": "boundary-test-sha",
            "clock": {"subdivisions": 4, "attackedSlots": [0, 3]},
            "templates": [
                {
                    "velocityMedian": 0.7,
                    "stableNotes": [
                        {"intervalFromBass": 0, "hand": "left", "durationPulses": 1.0}
                    ],
                },
                {
                    "velocityMedian": 0.7,
                    "stableNotes": [
                        {"intervalFromBass": 7, "hand": "right", "durationPulses": 1.0}
                    ],
                },
            ],
            "application": {
                "bassSourceOctaveShift": -12,
                "movingSourceOctaveShift": 0,
                "sourceCycleAnchorBlend": 1.0,
                "sourceCycleTempoBlend": 0.0,
                "minimumCycleSeparationRatio": 0.95,
            },
        }

        output, diagnostics = apply_intro_motif(
            {"notes": source_notes},
            {"notes": [], "pianoArrangement": {}},
            profile,
        )
        starts = sorted(
            {
                note["time"]
                for note in output["notes"]
                if note.get("motifGridSlot") == 0
            }
        )

        self.assertGreaterEqual(starts[1] - starts[0], 0.95 * diagnostics["cycleSeconds"])
        self.assertGreaterEqual(diagnostics["adjustedCycleBoundaries"], 1)


if __name__ == "__main__":
    unittest.main()
