import unittest

from ml.training.train_pianist_contextual_gesture_adapter import (
    add_learned_keys_to_destination,
    apply_profile,
    coalesced_note_duration,
    complete_destination_pitch_classes,
    correct_destination_registers,
    contextual_gate,
    destination_duration,
    destination_exact_or_learned_duration,
    fit_profile,
    fit_long_melody_hold,
    local_time_scale,
    nearby_isolated_pitch_evidence,
    preserve_allowed_destination_pitch_classes,
    preserve_allowed_destination_exact_keys,
    projected_destination_time,
    reanchor_delayed_isolated_gesture,
    regularized_transposition,
    select_projected_destination,
    trained_long_melody_duration,
)


def group(time, pitches, velocity=0.7, duration=0.2):
    return [
        {
            "time": time,
            "midi": midi,
            "duration": duration,
            "scoreDuration": duration,
            "velocity": velocity,
            "hand": "left" if midi < 60 else "right",
        }
        for midi in pitches
    ]


class PianistContextualGestureAdapterTest(unittest.TestCase):
    def test_profile_can_learn_only_the_requested_register_layer(self):
        reference = [group(1.0, [48, 60, 82, 91])]
        profile = fit_profile(
            reference,
            profile_id="upper-layer",
            song_id="example",
            template_start=0.8,
            template_end=1.2,
            target_time=1.0,
            repeat_search_minimum=1.0,
            repeat_search_maximum=2.0,
            learned_minimum_midi=80,
        )
        self.assertEqual(
            [note["midi"] for note in profile["learnedGesture"]["notes"]],
            [82, 91],
        )
        self.assertEqual(profile["training"]["learnedMidiRange"], [80, 108])

    def test_profile_can_select_two_voice_roles_from_a_larger_chord(self):
        reference = [group(1.0, [67, 70, 75, 89])]
        profile = fit_profile(
            reference,
            profile_id="voice-pair",
            song_id="example",
            template_start=0.8,
            template_end=1.2,
            target_time=1.0,
            repeat_search_minimum=1.0,
            repeat_search_maximum=2.0,
            learned_midis={67, 89},
        )
        self.assertEqual(
            [note["midi"] for note in profile["learnedGesture"]["notes"]],
            [67, 89],
        )
        self.assertEqual(profile["training"]["learnedMidiAllowlist"], [67, 89])

    def test_local_time_scale_uses_surrounding_input_intervals(self):
        source = [group(1.0, [48]), group(1.4, [60]), group(1.8, [55])]
        destination = [group(5.0, [48]), group(5.2, [60]), group(5.4, [55])]
        self.assertAlmostEqual(
            local_time_scale(source, destination, [(0, 0), (1, 1), (2, 2)], 1),
            0.5,
        )

    def test_neighbor_timing_projects_the_center_without_using_its_match(self):
        source = [group(1.0, [48]), group(1.4, [60]), group(1.8, [55])]
        destination = [
            group(5.0, [48]),
            group(5.35, [67]),
            group(5.4, [60]),
            group(5.8, [55]),
        ]
        projected = projected_destination_time(
            source,
            destination,
            [(0, 0), (1, 1), (2, 3)],
            1,
            4.0,
        )
        self.assertAlmostEqual(projected, 5.4)
        selected, reason = select_projected_destination(
            destination,
            mapped_destination=destination[1],
            projected_time=projected,
            maximum_distance=0.40,
        )
        self.assertIs(selected, destination[2])
        self.assertEqual(reason, "neighbor-time-projection")

    def test_projection_resists_one_bad_neighbor_correspondence(self):
        source = [
            group(0.000, [48]),
            group(0.158, [51]),
            group(0.445, [65]),
            group(0.733, [55]),
        ]
        destination = [
            group(10.000, [48]),
            group(10.239, [50]),
            group(10.378, [51]),
            group(10.527, [65]),
            group(10.666, [55]),
        ]
        projected = projected_destination_time(
            source,
            destination,
            [(0, 0), (1, 2), (2, 3), (3, 4)],
            2,
            10.0,
        )
        self.assertAlmostEqual(projected, 10.4045, places=3)
        selected, reason = select_projected_destination(
            destination,
            mapped_destination=destination[3],
            projected_time=projected,
            maximum_distance=0.40,
        )
        self.assertIs(selected, destination[2])
        self.assertEqual(reason, "neighbor-time-projection")

    def test_projection_cannot_jump_to_a_worse_harmony(self):
        destination = [
            group(5.0, [58]),
            group(5.30, [65]),
        ]
        selected, reason = select_projected_destination(
            destination,
            mapped_destination=destination[0],
            projected_time=5.28,
            maximum_distance=0.40,
            expected_pitch_classes={10},
        )
        self.assertIs(selected, destination[0])
        self.assertEqual(reason, "sequence-alignment")

    def test_projection_cannot_leave_a_better_direct_offset_match(self):
        destination = [
            group(5.0, [53, 65]),
            group(5.30, [84, 92]),
        ]
        selected, reason = select_projected_destination(
            destination,
            mapped_destination=destination[1],
            projected_time=5.05,
            direct_offset_time=5.27,
            maximum_distance=0.40,
            expected_pitch_classes={0, 5},
        )
        self.assertIs(selected, destination[1])
        self.assertEqual(reason, "sequence-alignment")

    def test_direct_clock_recovers_target_when_extra_attack_distorts_projection(self):
        destination = [
            group(63.940, [70]),
            group(64.098, [58, 75]),
        ]
        selected, reason = select_projected_destination(
            destination,
            mapped_destination=destination[1],
            projected_time=64.069,
            direct_offset_time=63.882,
            maximum_distance=0.40,
            expected_pitch_classes={10},
        )
        self.assertIs(selected, destination[0])
        self.assertEqual(reason, "direct-recurrence-clock")

    def test_harmonic_mask_preserves_valid_keys_and_performance(self):
        destination = group(5.0, [67, 70, 72, 75, 87], velocity=0.81, duration=0.42)
        retained = preserve_allowed_destination_pitch_classes(
            destination,
            {0, 3, 7},
        )
        self.assertEqual([note["midi"] for note in retained], [67, 72, 75, 87])
        self.assertTrue(all(note["velocity"] == 0.81 for note in retained))
        self.assertTrue(all(note["duration"] == 0.42 for note in retained))

    def test_pitch_completion_places_color_tone_above_existing_right_hand(self):
        completed = complete_destination_pitch_classes(
            group(5.0, [58, 75], velocity=0.81, duration=0.42),
            group(1.0, [67, 70, 75, 91]),
        )
        self.assertEqual([note["midi"] for note in completed], [58, 75, 79])
        added = next(note for note in completed if note["midi"] == 79)
        self.assertEqual(added["velocity"], 0.81)
        self.assertEqual(added["duration"], 0.42)

    def test_pitch_completion_restores_missing_bass_for_two_hand_gesture(self):
        learned = group(1.0, [70, 73, 91])
        learned[0]["hand"] = "left"
        learned[1]["hand"] = "right"
        learned[2]["hand"] = "right"
        completed = complete_destination_pitch_classes(
            group(5.0, [73, 79]),
            learned,
        )
        self.assertEqual([note["midi"] for note in completed], [58, 73, 79])
        self.assertEqual(completed[0]["hand"], "left")

    def test_new_bass_hold_stops_at_the_next_accompaniment_onset(self):
        learned = group(1.0, [67, 89], duration=0.44)
        learned[0]["hand"] = "left"
        learned[1]["hand"] = "right"
        destination = group(5.0, [89], duration=0.29)
        completed = complete_destination_pitch_classes(
            destination,
            learned,
            groups=[destination, group(5.15, [48, 60])],
        )
        self.assertEqual([note["midi"] for note in completed], [55, 89])
        self.assertAlmostEqual(completed[0]["duration"], 0.15)

    def test_layering_adds_missing_bass_key_without_replacing_melody(self):
        layered = add_learned_keys_to_destination(
            group(5.0, [53, 84], velocity=0.78),
            group(1.0, [53, 65], duration=0.27),
        )
        self.assertEqual([note["midi"] for note in layered], [53, 65, 84])
        self.assertEqual(next(note for note in layered if note["midi"] == 65)["duration"], 0.27)
        self.assertEqual(next(note for note in layered if note["midi"] == 84)["velocity"], 0.78)

    def test_new_chord_tone_uses_supported_destination_articulation(self):
        destination = group(4.0, [60, 64], duration=0.14)
        learned = {"midi": 67, "duration": 0.30}
        self.assertEqual(
            destination_duration(67, destination, learned, {0, 4, 7}),
            0.14,
        )
        self.assertEqual(
            destination_duration(67, destination, learned, {7}),
            0.30,
        )

    def test_wrong_octave_does_not_control_replacement_articulation(self):
        destination = group(4.0, [87], duration=0.50)
        learned = {"midi": 51, "duration": 0.28}
        self.assertEqual(
            destination_duration(
                51,
                destination,
                learned,
                {3},
                {51, 63},
            ),
            0.28,
        )

    def test_harmonic_median_can_reuse_wrong_octave_timing(self):
        destination = (
            group(4.0, [46], duration=0.94)
            + group(4.0, [53], duration=0.57)
            + group(4.0, [82], duration=0.54)
        )
        self.assertEqual(
            destination_duration(
                58,
                destination,
                {"midi": 58, "duration": 0.40},
                {5, 10},
                None,
            ),
            0.57,
        )

    def test_exact_or_learned_articulation_preserves_independent_releases(self):
        destination = group(4.0, [89], duration=0.29)
        existing = {"midi": 89, "duration": 0.53}
        missing = {"midi": 82, "duration": 0.53}
        self.assertEqual(
            destination_exact_or_learned_duration(89, destination, existing),
            0.29,
        )
        self.assertEqual(
            destination_exact_or_learned_duration(82, destination, missing),
            0.53,
        )

    def test_long_melody_hold_is_learned_only_from_earlier_evidence(self):
        reference = [
            group(1.0, [89], duration=1.14),
            group(2.5, [84, 92]),
            group(4.0, [89], duration=1.14),
            group(5.5, [84, 92]),
            group(7.0, [89], duration=1.14),
            group(8.5, [84, 92]),
            group(12.0, [91], duration=1.49),
            group(13.5, [84, 92]),
        ]
        settings = fit_long_melody_hold(reference, evidence_end=10.0)
        self.assertEqual(settings["trainingExamples"], 3)
        self.assertAlmostEqual(settings["holdToNextHighCueRatio"], 0.76)

    def test_trained_long_hold_uses_next_high_register_cue(self):
        candidate = [
            group(10.0, [48, 60]),
            group(10.6, [67, 70, 73, 75]),
            group(11.5, [84, 92]),
        ]
        duration, cue = trained_long_melody_duration(
            candidate,
            destination_time=10.0,
            midi=91,
            learned_note={"midi": 91, "duration": 0.27},
            settings={
                "holdToNextHighCueRatio": 0.76,
                "minimumCueGapSeconds": 1.2,
                "maximumCueGapSeconds": 2.1,
                "highCueFloorSemitonesBelowTarget": 7,
                "releaseGuardSeconds": 0.03,
            },
        )
        self.assertEqual(cue, 11.5)
        self.assertAlmostEqual(duration, 1.14)

    def test_transposition_requires_a_meaningful_gain_over_unison(self):
        trials = [
            {"semitones": 0, "meanPitchClassF1": 0.508},
            {"semitones": 5, "meanPitchClassF1": 0.524},
        ]
        shift, gain = regularized_transposition(5, trials, 0.08)
        self.assertEqual(shift, 0)
        self.assertAlmostEqual(gain, 0.016)
        shift, _ = regularized_transposition(5, trials, 0.01)
        self.assertEqual(shift, 5)

    def test_exact_key_mask_removes_only_wrong_register_doublings(self):
        destination = group(12.0, [67, 70, 75, 79, 82])
        output = preserve_allowed_destination_exact_keys(
            destination,
            {67, 70, 75, 82},
        )
        self.assertEqual([int(note["midi"]) for note in output], [67, 70, 75, 82])
        self.assertTrue(
            all(
                note["source"]
                == "polymath-pianist-contextual-exact-voicing-mask"
                for note in output
            )
        )

    def test_register_correction_preserves_unrelated_melody_and_detected_hold(self):
        destination = group(20.0, [70, 85, 91], duration=0.27)
        destination[0]["duration"] = 0.57
        destination[0]["scoreDuration"] = 0.57
        learned = group(4.0, [70, 73], duration=0.57)
        output = correct_destination_registers(destination, learned)
        self.assertEqual([int(note["midi"]) for note in output], [70, 73, 91])
        corrected = next(note for note in output if int(note["midi"]) == 73)
        self.assertAlmostEqual(float(corrected["duration"]), 0.57)
        self.assertEqual(
            corrected["source"],
            "polymath-pianist-contextual-register-correction",
        )

    def test_register_correction_removes_an_unwanted_octave_doubling(self):
        destination = group(8.0, [46, 58])
        learned = group(3.0, [46])
        output = correct_destination_registers(destination, learned)
        self.assertEqual([int(note["midi"]) for note in output], [46])

    def test_nearby_evidence_requires_an_isolated_matching_pitch(self):
        groups = [
            [{**group(4.0, [60])[0], "_payloadIndex": 0}],
            [{**group(3.90, [67], duration=0.80)[0], "_payloadIndex": 4}],
            [{**group(4.16, [67], duration=0.44)[0], "_payloadIndex": 1}],
            [
                {**group(4.10, [64])[0], "_payloadIndex": 2},
                {**group(4.10, [67])[0], "_payloadIndex": 3},
            ],
        ]
        evidence = nearby_isolated_pitch_evidence(
            groups,
            destination_time=4.0,
            midi=67,
            maximum_distance=0.22,
            excluded_indices={0},
        )
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence["_payloadIndex"], 1)
        self.assertEqual(evidence["duration"], 0.44)

    def test_late_solo_tone_reanchors_to_a_preceding_partial_chord(self):
        preceding = group(5.0, [60, 62])
        destination = group(5.2, [64])
        selected, diagnostics = reanchor_delayed_isolated_gesture(
            [preceding, destination],
            destination=destination,
            learned_midis=[60, 64],
            maximum_distance=0.22,
        )
        self.assertIs(selected, preceding)
        self.assertEqual(diagnostics["fromTimeSeconds"], 5.2)
        self.assertEqual(diagnostics["toTimeSeconds"], 5.0)

    def test_coalesced_tone_uses_chord_hold_when_training_holds_match(self):
        learned = [
            {"midi": 60, "duration": 0.2},
            {"midi": 67, "duration": 0.2},
        ]
        duration, strategy = coalesced_note_duration(
            midi=67,
            learned_note=learned[1],
            learned_notes=learned,
            destination=group(5.0, [60], duration=0.36),
            learned_pitch_classes={0, 7},
            learned_midis={60, 67},
            evidence={"time": 5.16, "midi": 67, "duration": 0.7},
            destination_time=5.0,
        )
        self.assertEqual(duration, 0.36)
        self.assertEqual(strategy, "destination-chord-articulation")

    def test_shortest_destination_hold_can_drive_a_shared_chord(self):
        candidate = {
            "notes": [
                note
                for item in (
                    group(1.0, [48]),
                    group(1.4, [60, 64], duration=0.5),
                    group(1.8, [55]),
                    group(5.0, [48]),
                    [
                        {**group(5.4, [60], duration=0.52)[0]},
                        {**group(5.4, [67], duration=0.28)[0]},
                    ],
                    group(5.8, [55]),
                )
                for note in item
            ]
        }
        profile = {
            "id": "shortest-hold-test",
            "profileSha256": "shortest-hold-sha",
            "context": {
                "templateStartSeconds": 1.0,
                "templateEndSeconds": 2.0,
                "targetTimeSeconds": 1.4,
                "targetPitchClasses": [0, 4],
                "targetOrdinal": 1,
                "gestures": [
                    {"relativeTime": 0.0, "midis": [48]},
                    {"relativeTime": 0.4, "midis": [60, 64]},
                    {"relativeTime": 0.8, "midis": [55]},
                ],
            },
            "learnedGesture": {
                "notes": [
                    {"midi": 60, "hand": "right", "duration": 0.5},
                    {"midi": 64, "hand": "right", "duration": 0.5},
                ],
            },
            "detection": {
                "minimumOffsetSeconds": 3.5,
                "maximumOffsetSeconds": 4.5,
                "coarseStepSeconds": 0.1,
                "fineStepSeconds": 0.01,
                "maximumTimeDistanceSeconds": 0.35,
                "gapCost": 0.85,
                "minimumMeanPitchClassF1": 0.70,
                "maximumNormalizedScore": 0.40,
                "maximumTargetTimeDistanceSeconds": 0.40,
                "minimumTemplateTargetPitchClassF1": 0.50,
            },
            "application": {"articulationSource": "destination-shared-shortest"},
        }
        output, diagnostics = apply_profile(candidate, profile)
        self.assertTrue(diagnostics["applied"])
        destination = [note for note in output["notes"] if note["time"] == 5.4]
        self.assertEqual([note["duration"] for note in destination], [0.28, 0.28])

    def test_replaces_only_repeated_center_keys(self):
        candidate = {
            "notes": [
                note
                for item in (
                    group(1.0, [48]),
                    group(1.4, [60, 64]),
                    group(1.8, [55]),
                    group(5.0, [48]),
                    group(5.4, [60, 67], velocity=0.82, duration=0.35),
                    group(5.8, [55]),
                )
                for note in item
            ]
        }
        profile = {
            "id": "context-test",
            "profileSha256": "context-sha",
            "context": {
                "templateStartSeconds": 1.0,
                "templateEndSeconds": 2.0,
                "targetTimeSeconds": 1.4,
                "targetPitchClasses": [0],
                "targetOrdinal": 1,
                "gestures": [
                    {"relativeTime": 0.0, "midis": [48]},
                    {"relativeTime": 0.4, "midis": [60]},
                    {"relativeTime": 0.8, "midis": [55]},
                ],
            },
            "learnedGesture": {
                "notes": [{"midi": 60, "hand": "right", "duration": 0.2}],
            },
            "detection": {
                "minimumOffsetSeconds": 3.5,
                "maximumOffsetSeconds": 4.5,
                "coarseStepSeconds": 0.1,
                "fineStepSeconds": 0.01,
                "maximumTimeDistanceSeconds": 0.35,
                "gapCost": 0.85,
                "minimumMeanPitchClassF1": 0.75,
                "maximumNormalizedScore": 0.25,
                "maximumTargetTimeDistanceSeconds": 0.40,
                "minimumTemplateTargetPitchClassF1": 0.50,
            },
        }
        output, diagnostics = apply_profile(candidate, profile)
        self.assertTrue(diagnostics["applied"])
        destination = [note for note in output["notes"] if note["time"] == 5.4]
        self.assertEqual([note["midi"] for note in destination], [60])
        self.assertEqual(destination[0]["velocity"], 0.82)
        self.assertEqual(destination[0]["duration"], 0.35)
        source = [note for note in output["notes"] if note["time"] == 1.4]
        self.assertEqual([note["midi"] for note in source], [60, 64])

    def test_coalesces_a_nearby_isolated_melody_tone_and_keeps_its_hold(self):
        candidate = {
            "notes": [
                note
                for item in (
                    group(1.0, [48]),
                    group(1.4, [60, 64, 67]),
                    group(1.8, [55]),
                    group(5.0, [48]),
                    group(5.4, [60, 64], velocity=0.82, duration=0.29),
                    group(5.56, [67], velocity=0.74, duration=0.44),
                    group(5.8, [55]),
                )
                for note in item
            ]
        }
        profile = {
            "id": "coalescing-test",
            "profileSha256": "coalescing-sha",
            "context": {
                "templateStartSeconds": 1.0,
                "templateEndSeconds": 2.0,
                "targetTimeSeconds": 1.4,
                "targetPitchClasses": [0, 4, 7],
                "targetOrdinal": 1,
                "gestures": [
                    {"relativeTime": 0.0, "midis": [48]},
                    {"relativeTime": 0.4, "midis": [60, 64, 67]},
                    {"relativeTime": 0.8, "midis": [55]},
                ],
            },
            "learnedGesture": {
                "notes": [
                    {"midi": 60, "hand": "right", "duration": 0.4},
                    {"midi": 64, "hand": "right", "duration": 0.4},
                    {"midi": 67, "hand": "right", "duration": 0.1},
                ],
            },
            "detection": {
                "minimumOffsetSeconds": 3.5,
                "maximumOffsetSeconds": 4.5,
                "coarseStepSeconds": 0.1,
                "fineStepSeconds": 0.01,
                "maximumTimeDistanceSeconds": 0.35,
                "gapCost": 0.85,
                "minimumMeanPitchClassF1": 0.70,
                "maximumNormalizedScore": 0.40,
                "maximumTargetTimeDistanceSeconds": 0.40,
                "minimumTemplateTargetPitchClassF1": 0.50,
            },
            "application": {
                "articulationSource": "nearby-isolated-pitch-coalesced",
                "maximumCoalescingDistanceSeconds": 0.22,
            },
        }
        output, diagnostics = apply_profile(candidate, profile)
        self.assertTrue(diagnostics["applied"])
        destination = [note for note in output["notes"] if note["time"] == 5.4]
        self.assertEqual([note["midi"] for note in destination], [60, 64, 67])
        self.assertEqual(
            next(note for note in destination if note["midi"] == 67)["duration"],
            0.60,
        )
        self.assertFalse(any(note["time"] == 5.56 for note in output["notes"]))
        self.assertEqual(diagnostics["coalescedNearbyNotes"], 1)

    def test_gate_accepts_structure_gain_without_performance_regression(self):
        baseline = {
            "referenceGestureRecall": 1.0,
            "candidateGesturePrecision": 1.0,
            "coverageAdjustedPitchClassF1": 0.8,
            "coverageAdjustedExactPitchClassRate": 0.5,
            "coverageAdjustedOccupancyAccuracy": 0.8,
            "onsetMaeSeconds": 0.03,
            "gestureVelocityMae": 0.04,
            "exactKeyDurationMaeSeconds": 0.05,
            "sequenceScore": 2.0,
        }
        candidate = {
            **baseline,
            "coverageAdjustedPitchClassF1": 1.0,
            "coverageAdjustedExactPitchClassRate": 1.0,
            "coverageAdjustedOccupancyAccuracy": 1.0,
            "sequenceScore": 1.0,
        }
        passed, reason = contextual_gate(baseline, candidate, baseline, candidate)
        self.assertTrue(passed)
        self.assertEqual(reason, "safe-contextual-gesture-refinement")


if __name__ == "__main__":
    unittest.main()
