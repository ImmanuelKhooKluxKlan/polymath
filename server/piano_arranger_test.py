import copy
import json
import unittest
from pathlib import Path

from piano_arranger import (
    COMPACT_PIANO_MAX_MIDI,
    COMPACT_PIANO_MIN_MIDI,
    MAX_ARRANGED_NOTES_PER_SECOND,
    PIANO_MAX_MIDI,
    PIANO_MIN_MIDI,
    adapt_profile_to_source_density,
    adapt_physical_performance_to_source,
    adapt_melody_register_separation,
    apply_authored_duration_style,
    arrange_payload,
    collapse_exact_left_hand_duplicates,
    collapse_focused_melody_collisions,
    compact_authored_two_hand_register,
    compact_harmony,
    compact_pianella_register,
    interpolate_raw_selection_models,
    learned_physical_performance_limits,
    refine_left_hand_accompaniment,
    route_conditional_learned_profile,
    shape_gesture_coherent_expression,
    shape_melody_forward_expression,
    source_supported_cyclic_harmony,
    suppress_monophonic_vocal_floor_runs,
)
from piano_arranger_adapter import (
    CONTEXT_SELECTION_FEATURE_NAMES,
    DURATION_FEATURE_NAMES,
    FEATURE_NAMES,
    HARMONIC_LEFT_HAND_SELECTION_FEATURE_NAMES,
    LEFT_HAND_SELECTION_FEATURE_NAMES,
    ROBUST_CONTEXT_SELECTION_FEATURE_NAMES,
    _pick_window_notes,
    _rerank_conditional_selection_slots,
    _rerank_conditional_onset_slots,
    _render_note,
    apply_learned_register_model,
    contextual_selection_scores,
    duration_predictions,
    normalize_source_notes,
    raw_feature_rows,
    selection_feature_rows,
    selection_scores,
)


LEARNED_PROFILE = json.loads(
    (Path(__file__).parent / "models" / "piano-arranger" / "pianella-supervised-v006.json")
    .read_text(encoding="utf-8")
)


def note(midi, time, instrument, duration=0.3, velocity=0.75):
    return {
        "midi": midi,
        "time": time,
        "duration": duration,
        "velocity": velocity,
        "instrument": instrument,
    }


def _group_test_notes(notes):
    groups = {}
    for item in notes:
        groups.setdefault(round(float(item["time"]), 6), []).append(item)
    return [
        sorted(groups[time], key=lambda item: item["midi"])
        for time in sorted(groups)
    ]


class ConditionalLearnedRouteTests(unittest.TestCase):
    def profile(self):
        return {
            "schema": "polymath-piano-arranger-profile-v1",
            "id": "conditional-v1",
            "selectionModel": {"weights": [1.0]},
            "decoder": {
                "conditionalLearnedRoute": {
                    "enabled": True,
                    "maximumVoiceRatio": 0.02,
                    "minimumBassRatio": 0.5,
                    "maximumPianoRatio": 0.08,
                    "minimumSourceNotes": 64,
                    "fallbackProfileId": "safe-default-v1",
                    "fallbackDecoder": {
                        "defaultPipeline": True,
                        "gestureDynamics": {"enabled": True},
                    },
                    "monophonicVocalRoute": {
                        "enabled": True,
                        "minimumVoiceRatio": 0.95,
                        "maximumBassRatio": 0.02,
                        "maximumPianoRatio": 0.02,
                        "minimumSourceNotes": 32,
                        "fallbackProfileId": "monophonic-vocal-v1",
                        "fallbackDecoder": {
                            "defaultPipeline": True,
                            "monophonicVocalCleanup": {
                                "enabled": True,
                                "maximumFloorVelocity": 0.46,
                                "minimumRunNotes": 6,
                                "maximumRunGapSeconds": 0.35,
                                "onsetDelaySeconds": 0.02,
                            },
                        },
                    },
                }
            },
        }

    def test_uses_learned_selector_for_bass_collapsed_source(self):
        profile = self.profile()
        selected, diagnostics = route_conditional_learned_profile(
            profile,
            {
                "instrumentCounts": {"electric_bass": 90, "drums": 10},
                "voiceRatio": 0.0,
                "bassRatio": 0.9,
                "pianoRatio": 0.0,
            },
        )
        self.assertIs(selected, profile)
        self.assertEqual(diagnostics["selectedRoute"], "learned-selector")
        self.assertTrue(all(diagnostics["checks"].values()))

    def test_falls_back_when_voice_evidence_is_healthy(self):
        profile = self.profile()
        selected, diagnostics = route_conditional_learned_profile(
            profile,
            {
                "instrumentCounts": {"electric_bass": 20, "voice": 20, "drums": 60},
                "voiceRatio": 0.2,
                "bassRatio": 0.2,
                "pianoRatio": 0.0,
            },
        )
        self.assertIsNot(selected, profile)
        self.assertEqual(selected["id"], "safe-default-v1")
        self.assertTrue(selected["decoder"]["defaultPipeline"])
        self.assertEqual(diagnostics["selectedRoute"], "default-pipeline-fallback")

    def test_rejects_enabled_gate_without_safe_fallback(self):
        profile = self.profile()
        del profile["decoder"]["conditionalLearnedRoute"]["fallbackDecoder"]
        with self.assertRaisesRegex(ValueError, "fallbackDecoder"):
            route_conditional_learned_profile(
                profile,
                {
                    "instrumentCounts": {
                        "voice": 20,
                        "electric_bass": 20,
                        "drums": 60,
                    },
                    "voiceRatio": 0.2,
                    "bassRatio": 0.2,
                    "pianoRatio": 0.0,
                },
            )

    def test_routes_pure_voice_to_monophonic_default(self):
        selected, diagnostics = route_conditional_learned_profile(
            self.profile(),
            {
                "instrumentCounts": {"voice": 80},
                "voiceRatio": 1.0,
                "bassRatio": 0.0,
                "pianoRatio": 0.0,
            },
        )
        self.assertEqual(selected["id"], "monophonic-vocal-v1")
        self.assertTrue(selected["decoder"]["defaultPipeline"])
        self.assertNotIn("gestureDynamics", selected["decoder"])
        self.assertEqual(diagnostics["selectedRoute"], "monophonic-vocal-default")

    def test_arrange_payload_records_the_selected_route(self):
        profile = copy.deepcopy(LEARNED_PROFILE)
        profile["decoder"]["conditionalLearnedRoute"] = {
            "enabled": True,
            "maximumVoiceRatio": 0.02,
            "minimumBassRatio": 0.5,
            "maximumPianoRatio": 0.08,
            "minimumSourceNotes": 8,
            "fallbackProfileId": "safe-default-v1",
            "fallbackDecoder": {"defaultPipeline": True},
        }
        bass_payload = {
            "notes": [
                note(40 + index % 8, index * 0.2, "electric_bass")
                for index in range(16)
            ]
        }
        learned = arrange_payload(bass_payload, "full", profile)
        self.assertEqual(
            learned["pianoArrangement"]["conditionalLearnedRoute"]["selectedRoute"],
            "learned-selector",
        )

        voice_payload = {
            "notes": [
                note(60 + index % 8, index * 0.2, "voice")
                for index in range(16)
            ]
        }
        fallback = arrange_payload(voice_payload, "full", profile)
        self.assertEqual(
            fallback["pianoArrangement"]["conditionalLearnedRoute"]["selectedRoute"],
            "default-pipeline-fallback",
        )
        self.assertTrue(fallback["pianoArrangement"]["defaultPipelineProfileApplied"])


class MonophonicVocalCleanupTests(unittest.TestCase):
    def test_removes_only_a_long_consecutive_floor_run_and_delays_retained_notes(self):
        notes = [note(64, 0.0, "voice", velocity=0.9)]
        notes.extend(
            note(60 + index % 4, 1.0 + index * 0.25, "voice", velocity=0.448)
            for index in range(6)
        )
        notes.extend(
            [
                note(67, 3.0, "voice", velocity=0.95),
                note(69, 3.5, "voice", velocity=0.448),
            ]
        )

        cleaned, diagnostics = suppress_monophonic_vocal_floor_runs(
            notes,
            {
                "enabled": True,
                "maximumFloorVelocity": 0.46,
                "minimumRunNotes": 6,
                "maximumRunGapSeconds": 0.35,
                "onsetDelaySeconds": 0.02,
            },
        )

        self.assertEqual(len(cleaned), 3)
        self.assertEqual([item["midi"] for item in cleaned], [64, 67, 69])
        self.assertEqual([item["time"] for item in cleaned], [0.02, 3.02, 3.52])
        self.assertEqual(diagnostics["detectedRuns"], 1)
        self.assertEqual(diagnostics["removedNotes"], 6)

    def test_keeps_short_quiet_phrases(self):
        quiet = [
            note(60 + index, index * 0.25, "voice", velocity=0.448)
            for index in range(5)
        ]
        cleaned, diagnostics = suppress_monophonic_vocal_floor_runs(
            quiet,
            {
                "enabled": True,
                "maximumFloorVelocity": 0.46,
                "minimumRunNotes": 6,
                "maximumRunGapSeconds": 0.35,
                "onsetDelaySeconds": 0.0,
            },
        )
        self.assertEqual(len(cleaned), 5)
        self.assertEqual(diagnostics["removedNotes"], 0)


class AdaptiveMelodyRegisterSeparationTests(unittest.TestCase):
    @staticmethod
    def arranged(midi, time, role):
        return {
            **note(
                midi,
                time,
                "voice" if role == "melody" else "clean_electric_guitar",
            ),
            "note": "source-note",
            "arrangementRole": role,
            "hand": "right" if midi >= 60 else "left",
        }

    def policy(self, **overrides):
        return {
            "enabled": True,
            "minimumMelodyNotes": 4,
            "minimumComparedNotes": 4,
            "nearbyHarmonyRadiusSeconds": 0.12,
            "collisionClearanceSemitones": 2,
            "minimumCollisionShare": 0.5,
            "octaveShiftSemitones": 12,
            **overrides,
        }

    def test_lifts_the_complete_melody_when_register_collisions_repeat(self):
        notes = []
        for index in range(6):
            onset = index * 0.25
            notes.append(self.arranged(64 + index % 2, onset, "melody"))
            notes.append(self.arranged(63, onset, "harmony"))

        result, diagnostics = adapt_melody_register_separation(
            notes, self.policy()
        )

        melody = [item for item in result if item["arrangementRole"] == "melody"]
        harmony = [item for item in result if item["arrangementRole"] == "harmony"]
        self.assertTrue(diagnostics["applied"])
        self.assertEqual(diagnostics["shiftedNotes"], 6)
        self.assertEqual([item["midi"] for item in melody], [76, 77, 76, 77, 76, 77])
        self.assertEqual({item["adaptiveMelodyRegisterShiftSemitones"] for item in melody}, {12})
        self.assertEqual([item["midi"] for item in harmony], [63] * 6)

    def test_keeps_an_already_separated_melody_unchanged(self):
        notes = []
        for index in range(6):
            onset = index * 0.25
            notes.append(self.arranged(72, onset, "melody"))
            notes.append(self.arranged(55, onset, "harmony"))

        result, diagnostics = adapt_melody_register_separation(
            notes, self.policy()
        )

        self.assertFalse(diagnostics["applied"])
        self.assertEqual(diagnostics["reason"], "melody-register-already-separated")
        self.assertEqual([item["midi"] for item in result], [item["midi"] for item in notes])

    def test_applies_one_consistent_optional_melody_delay(self):
        notes = []
        for index in range(6):
            onset = index * 0.25
            notes.append(self.arranged(64, onset, "melody"))
            notes.append(self.arranged(63, onset, "harmony"))

        result, diagnostics = adapt_melody_register_separation(
            notes, self.policy(onsetDelaySeconds=0.02)
        )

        melody = [item for item in result if item["arrangementRole"] == "melody"]
        harmony = [item for item in result if item["arrangementRole"] == "harmony"]
        self.assertEqual(
            [item["time"] for item in melody],
            [0.02, 0.27, 0.52, 0.77, 1.02, 1.27],
        )
        self.assertEqual([item["time"] for item in harmony], [0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
        self.assertEqual(diagnostics["appliedOnsetDelaySeconds"], 0.02)

    def test_refuses_a_partial_shift_that_would_create_octave_jumps(self):
        notes = []
        for index, midi in enumerate((64, 65, 100, 64, 65, 64)):
            onset = index * 0.25
            notes.append(self.arranged(midi, onset, "melody"))
            notes.append(self.arranged(midi - 1, onset, "harmony"))

        result, diagnostics = adapt_melody_register_separation(
            notes, self.policy()
        )

        self.assertFalse(diagnostics["applied"])
        self.assertEqual(diagnostics["reason"], "whole-melody-shift-exceeds-piano-range")
        self.assertEqual([item["midi"] for item in result], [item["midi"] for item in notes])

    def test_is_inert_until_a_profile_explicitly_enables_it(self):
        notes = [
            self.arranged(64, 0.0, "melody"),
            self.arranged(63, 0.0, "harmony"),
        ]

        result, diagnostics = adapt_melody_register_separation(notes)

        self.assertFalse(diagnostics["applied"])
        self.assertEqual(diagnostics["reason"], "disabled")
        self.assertEqual(result, notes)


class GestureCoherentExpressionTests(unittest.TestCase):
    PROFILE = {
        "decoder": {
            "gestureDynamics": {
                "enabled": True,
                "sourceBlend": 0.82,
                "melodyPerformanceGain": 1.08,
                "accompanimentGainDuringMelody": 0.94,
            }
        }
    }

    @staticmethod
    def arranged(midi, time, velocity, source_velocity, role="harmony"):
        return {
            **note(midi, time, "acoustic_piano", velocity=velocity),
            "sourceInstrument": "voice" if role == "melody" else "clean_electric_guitar",
            "arrangementRole": role,
            "sourceVelocityBeforeArrangement": source_velocity,
            "hand": "right" if midi >= 60 else "left",
        }

    def test_one_onset_uses_one_hammer_velocity_and_separate_melody_gain(self):
        notes = [
            self.arranged(48, 0.0, 0.35, 0.38),
            self.arranged(72, 0.0, 0.92, 0.38, "melody"),
            self.arranged(50, 0.5, 0.42, 0.64),
            self.arranged(74, 0.52, 0.88, 0.64, "melody"),
            self.arranged(52, 1.0, 0.46, 0.94),
            self.arranged(76, 1.0, 0.90, 0.94, "melody"),
        ]

        result, diagnostics = shape_gesture_coherent_expression(notes, self.PROFILE)

        self.assertTrue(diagnostics["applied"])
        self.assertTrue(diagnostics["sourceVelocityRangeWasUsable"])
        groups = {}
        for item in result:
            groups.setdefault(item["gestureDynamicGroup"], []).append(item)
        self.assertEqual(len(groups), 3)
        self.assertTrue(
            all(len({item["velocity"] for item in group}) == 1 for group in groups.values())
        )
        first = groups[0]
        self.assertEqual(
            {item["performanceGain"] for item in first if item["arrangementRole"] == "melody"},
            {1.08},
        )
        self.assertEqual(
            {item["performanceGain"] for item in first if item["arrangementRole"] != "melody"},
            {0.94},
        )

    def test_flat_source_uses_structural_gesture_dynamics(self):
        notes = [self.arranged(60, 0.0, 0.55, 0.78)]
        notes.extend(
            self.arranged(midi, 1.0, 0.45 + index * 0.1, 0.78)
            for index, midi in enumerate((48, 55, 64, 72))
        )

        result, diagnostics = shape_gesture_coherent_expression(notes, self.PROFILE)

        self.assertFalse(diagnostics["sourceVelocityRangeWasUsable"])
        first = [item for item in result if item["time"] == 0.0][0]
        chord = [item for item in result if item["time"] == 1.0]
        self.assertGreater(chord[0]["velocity"], first["velocity"])
        self.assertEqual(len({item["velocity"] for item in chord}), 1)

    def test_optional_overlap_ducking_catches_accompaniment_that_starts_before_melody(self):
        profile = copy.deepcopy(self.PROFILE)
        profile["decoder"]["gestureDynamics"].update(
            {
                "duckOverlappingAccompaniment": True,
                "performanceGainLookBehindSeconds": 0.04,
                "performanceGainLookAheadSeconds": 0.08,
                "accompanimentGainDuringMelody": 0.4,
            }
        )
        harmony = self.arranged(48, 0.0, 0.55, 0.60)
        harmony["duration"] = 0.7
        melody = self.arranged(72, 0.5, 0.82, 0.80, "melody")
        melody["duration"] = 0.3
        outside = self.arranged(50, 1.2, 0.55, 0.60)
        outside["duration"] = 0.2

        result, diagnostics = shape_gesture_coherent_expression(
            [harmony, melody, outside], profile
        )
        by_time = {item["time"]: item for item in result}

        self.assertEqual(by_time[0.0]["performanceGain"], 0.4)
        self.assertEqual(by_time[0.5]["performanceGain"], 1.08)
        self.assertEqual(by_time[1.2]["performanceGain"], 1.0)
        self.assertTrue(diagnostics["duckOverlappingAccompaniment"])
        self.assertEqual(diagnostics["overlappingAccompanimentNotes"], 1)

    def test_overlap_ducking_is_inert_until_explicitly_enabled(self):
        harmony = self.arranged(48, 0.0, 0.55, 0.60)
        harmony["duration"] = 0.7
        melody = self.arranged(72, 0.5, 0.82, 0.80, "melody")

        result, diagnostics = shape_gesture_coherent_expression(
            [harmony, melody], self.PROFILE
        )
        by_time = {item["time"]: item for item in result}

        self.assertEqual(by_time[0.0]["performanceGain"], 1.0)
        self.assertFalse(diagnostics["duckOverlappingAccompaniment"])

    def test_performance_only_mode_preserves_the_accepted_gesture_median(self):
        profile = copy.deepcopy(self.PROFILE)
        profile["decoder"]["gestureDynamics"].update(
            {
                "preserveInputGestureVelocity": True,
                "minimumVelocity": 0.05,
                "maximumVelocity": 1.0,
            }
        )
        notes = [
            self.arranged(48, 0.0, 0.30, 0.90),
            self.arranged(72, 0.0, 0.90, 0.20, "melody"),
        ]

        result, diagnostics = shape_gesture_coherent_expression(notes, profile)

        self.assertEqual({item["velocity"] for item in result}, {0.60})
        self.assertTrue(diagnostics["preservedInputGestureVelocity"])

    def test_disabled_profile_preserves_legacy_notes(self):
        notes = [self.arranged(60, 0.0, 0.55, 0.78)]
        result, diagnostics = shape_gesture_coherent_expression(notes, {})
        self.assertIs(result, notes)
        self.assertFalse(diagnostics["applied"])

    def test_optional_quantile_map_restores_a_wide_pianist_dynamic_range(self):
        profile = copy.deepcopy(self.PROFILE)
        profile["decoder"]["gestureDynamics"].update(
            {
                "sourceBlend": 0.0,
                "durationAccentStrength": 0.0,
                "velocityQuantileMap": {"0": 0.38, "1": 0.94},
                "quantileCalibrationBlend": 1.0,
            }
        )
        notes = [
            self.arranged(60, 0.0, 0.55, 0.78),
            self.arranged(48, 1.0, 0.45, 0.78),
            self.arranged(55, 1.0, 0.55, 0.78),
            self.arranged(64, 1.0, 0.65, 0.78),
            self.arranged(72, 1.0, 0.75, 0.78, "melody"),
        ]

        result, diagnostics = shape_gesture_coherent_expression(notes, profile)
        velocities = {
            time: {item["velocity"] for item in result if item["time"] == time}
            for time in (0.0, 1.0)
        }

        self.assertEqual(velocities[0.0], {0.38})
        self.assertEqual(velocities[1.0], {0.94})
        self.assertTrue(diagnostics["velocityQuantileCalibrationApplied"])
        self.assertGreater(diagnostics["meanAbsoluteQuantileCalibrationChange"], 0)

    def test_optional_paired_calibration_corrects_whole_gesture_after_quantiles(self):
        profile = copy.deepcopy(self.PROFILE)
        profile["decoder"]["gestureDynamics"]["pairedCalibration"] = {
            "enabled": True,
            "id": "test-soft-touch",
            "featureNames": ["bias"],
            "weights": [-0.08],
            "means": [0.0],
            "scales": [1.0],
            "blend": 1.0,
            "maximumCorrection": 0.1,
        }
        notes = [
            self.arranged(48, 0.0, 0.55, 0.38),
            self.arranged(72, 0.0, 0.85, 0.38, "melody"),
        ]

        result, diagnostics = shape_gesture_coherent_expression(notes, profile)

        self.assertEqual(len({item["velocity"] for item in result}), 1)
        self.assertTrue(diagnostics["pairedCalibrationApplied"])
        self.assertEqual(diagnostics["pairedCalibrationProfile"], "test-soft-touch")
        self.assertAlmostEqual(
            diagnostics["meanAbsolutePairedCalibrationChange"], 0.08, places=3
        )

    def test_optional_duration_calibration_scales_a_gesture_without_moving_notes(self):
        profile = copy.deepcopy(self.PROFILE)
        profile["decoder"]["gestureDynamics"]["pairedDurationCalibration"] = {
            "enabled": True,
            "id": "test-articulation",
            "featureNames": ["bias"],
            "weights": [0.20],
            "means": [0.0],
            "scales": [1.0],
            "blend": 1.0,
            "maximumLogCorrection": 0.5,
            "minimumDurationSeconds": 0.05,
            "maximumDurationSeconds": 4.0,
        }
        notes = [
            self.arranged(48, 0.0, 0.55, 0.38),
            self.arranged(72, 0.0, 0.85, 0.38, "melody"),
        ]
        notes[0]["duration"] = notes[0]["scoreDuration"] = 0.2
        notes[1]["duration"] = notes[1]["scoreDuration"] = 0.4
        original = [(item["time"], item["midi"], item["duration"]) for item in notes]

        result, diagnostics = shape_gesture_coherent_expression(notes, profile)

        self.assertEqual(
            [(item["time"], item["midi"]) for item in result],
            [(time, midi) for time, midi, _duration in original],
        )
        self.assertTrue(diagnostics["pairedDurationCalibrationApplied"])
        self.assertEqual(
            diagnostics["pairedDurationCalibrationProfile"], "test-articulation"
        )
        self.assertGreater(diagnostics["meanAbsolutePairedDurationChangeSeconds"], 0)
        self.assertAlmostEqual(result[0]["duration"] / 0.2, result[1]["duration"] / 0.4, places=5)
        self.assertEqual(result[0]["durationBeforeGestureCalibration"], 0.2)
        self.assertEqual(result[1]["durationBeforeGestureCalibration"], 0.4)


class FocusedMelodyCollisionTests(unittest.TestCase):
    @staticmethod
    def arranged(
        midi,
        time,
        source_instrument,
        *,
        role="melody",
        probability=0.5,
        velocity=0.7,
        duration=0.4,
        **extra,
    ):
        return {
            **note(midi, time, "acoustic_piano", duration, velocity),
            "sourceInstrument": source_instrument,
            "arrangementRole": role,
            "selectionProbability": probability,
            **extra,
        }

    def test_focused_voice_note_replaces_nearby_melody_leakage(self):
        focused_voice = self.arranged(
            72,
            10.0,
            "voice",
            probability=0.72,
            focusedMelodyDecoded=True,
        )
        guitar_leakage = self.arranged(
            76,
            10.04,
            "clean_electric_guitar",
            probability=0.96,
            velocity=0.95,
        )
        harmony = self.arranged(
            60,
            10.02,
            "clean_electric_guitar",
            role="harmony",
        )

        result, removed = collapse_focused_melody_collisions(
            [guitar_leakage, harmony, focused_voice]
        )

        self.assertEqual(removed, 1)
        self.assertEqual(
            [(item["midi"], item["arrangementRole"]) for item in result],
            [(72, "melody"), (60, "harmony")],
        )

    def test_ordinary_close_melody_notes_are_not_rewritten(self):
        notes = [
            self.arranged(72, 2.0, "flutes"),
            self.arranged(76, 2.04, "violin"),
        ]

        result, removed = collapse_focused_melody_collisions(notes)

        self.assertEqual(removed, 0)
        self.assertEqual([item["midi"] for item in result], [72, 76])

    def test_recovered_primary_anchor_wins_over_conflicting_focused_guess(self):
        decoded = self.arranged(
            75,
            4.0,
            "voice",
            probability=0.99,
            velocity=0.99,
            focusedMelodyDecoded=True,
        )
        anchor = self.arranged(
            67,
            4.03,
            "voice",
            probability=0.1,
            velocity=0.4,
            focusedMelodyDecoded=True,
            focusedMelodyAnchorRecovered=True,
        )

        result, removed = collapse_focused_melody_collisions([decoded, anchor])

        self.assertEqual(removed, 1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["midi"], 67)
        self.assertTrue(result[0]["focusedMelodyAnchorRecovered"])


class ExactLeftHandDuplicateTests(unittest.TestCase):
    @staticmethod
    def arranged(
        midi,
        time,
        source_instrument,
        *,
        role="harmony",
        duration=0.4,
        velocity=0.7,
        **extra,
    ):
        return {
            **note(midi, time, "acoustic_piano", duration, velocity),
            "sourceInstrument": source_instrument,
            "arrangementRole": role,
            **extra,
        }

    def test_same_left_key_and_onset_become_one_physical_strike(self):
        quiet = self.arranged(
            55,
            2.0,
            "electric_bass",
            role="bass",
            duration=0.35,
            velocity=0.52,
            selectionProbability=0.75,
            audioDuration=0.48,
        )
        long = self.arranged(
            55,
            2.0003,
            "clean_electric_guitar",
            duration=0.8,
            velocity=0.78,
            selectionProbability=0.7,
            audioDuration=1.1,
        )

        result, removed = collapse_exact_left_hand_duplicates([quiet, long])

        self.assertEqual(removed, 1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["midi"], 55)
        self.assertEqual(result[0]["duration"], 0.8)
        self.assertEqual(result[0]["audioDuration"], 1.1)
        self.assertEqual(result[0]["velocity"], 0.78)
        self.assertEqual(result[0]["collapsedSimultaneousSourceStrikes"], 2)
        self.assertEqual(
            result[0]["collapsedSourceInstruments"],
            ["clean_electric_guitar", "electric_bass"],
        )

    def test_protected_melody_wins_without_being_rewritten(self):
        melody = self.arranged(
            67,
            4.0,
            "voice",
            role="melody",
            duration=0.61,
            velocity=0.84,
            focusedMelodyDecoded=True,
        )
        accompaniment = self.arranged(
            67,
            4.0,
            "clean_electric_guitar",
            duration=0.3,
            velocity=0.95,
        )
        original_melody = copy.deepcopy(melody)

        result, removed = collapse_exact_left_hand_duplicates(
            [accompaniment, melody]
        )

        self.assertEqual(removed, 1)
        self.assertEqual(result, [original_melody])

    def test_different_pitches_at_one_onset_remain_a_chord(self):
        chord = [
            self.arranged(48, 8.0, "electric_bass", role="bass"),
            self.arranged(55, 8.0, "clean_electric_guitar"),
            self.arranged(60, 8.0, "string_ensemble"),
        ]

        result, removed = collapse_exact_left_hand_duplicates(chord)

        self.assertEqual(removed, 0)
        self.assertEqual([item["midi"] for item in result], [48, 55, 60])

class ContextSelectionFeatureTests(unittest.TestCase):
    def test_adaptive_interpolation_lifts_append_only_sparse_contract(self):
        low = {
            "featureNames": ["bias", "pitch"],
            "weights": [0.2, 0.8],
            "means": [0.0, 0.0],
            "scales": [1.0, 1.0],
            "threshold": 0.4,
        }
        high = {
            "featureNames": ["bias", "pitch", "voice_near"],
            "weights": [-0.1, 0.3, 0.9],
            "means": [0.0, 0.0, 0.0],
            "scales": [1.0, 1.0, 1.0],
            "threshold": 0.7,
        }
        result = interpolate_raw_selection_models(low, high, 0.0)
        self.assertEqual(result["featureNames"], high["featureNames"])
        self.assertEqual(result["weights"], [0.2, 0.8, 0.0])
        self.assertEqual(result["threshold"], 0.4)

    def test_v1_feature_contract_is_unchanged(self):
        notes = normalize_source_notes(
            [
                note(60, 0.0, "voice"),
                note(48, 0.02, "acoustic_guitar"),
            ]
        )
        self.assertEqual(selection_feature_rows(notes, FEATURE_NAMES), raw_feature_rows(notes))
        self.assertEqual(len(selection_feature_rows(notes, FEATURE_NAMES)[0]), 20)

    def test_v2_features_capture_stem_and_vocal_context(self):
        notes = normalize_source_notes(
            [
                note(60, 0.0, "voice"),
                note(48, 0.02, "clean_electric_guitar"),
                note(60, 0.03, "electric_bass"),
            ]
        )
        rows = selection_feature_rows(notes, CONTEXT_SELECTION_FEATURE_NAMES)
        names = list(CONTEXT_SELECTION_FEATURE_NAMES)
        guitar_row = rows[1]
        self.assertEqual(len(guitar_row), len(CONTEXT_SELECTION_FEATURE_NAMES))
        self.assertEqual(guitar_row[names.index("is_clean_electric_guitar")], 1.0)
        self.assertEqual(guitar_row[names.index("voice_within_060ms")], 1.0)
        self.assertGreater(guitar_row[names.index("cross_family_onset_support")], 0.0)

    def test_v2_lite_excludes_track_specific_stem_flags(self):
        notes = normalize_source_notes(
            [
                note(60, 0.0, "voice"),
                note(48, 0.02, "clean_electric_guitar"),
            ]
        )
        rows = selection_feature_rows(notes, ROBUST_CONTEXT_SELECTION_FEATURE_NAMES)
        self.assertEqual(len(rows[0]), len(ROBUST_CONTEXT_SELECTION_FEATURE_NAMES))
        self.assertNotIn(
            "is_clean_electric_guitar", ROBUST_CONTEXT_SELECTION_FEATURE_NAMES
        )
        self.assertIn("voice_within_060ms", ROBUST_CONTEXT_SELECTION_FEATURE_NAMES)

    def test_zero_weight_context_extension_preserves_v1_probabilities(self):
        notes = normalize_source_notes(
            [
                note(60, 0.0, "voice"),
                note(48, 0.02, "clean_electric_guitar"),
                note(40, 0.50, "electric_bass"),
            ]
        )
        context_profile = copy.deepcopy(LEARNED_PROFILE)
        extension = len(CONTEXT_SELECTION_FEATURE_NAMES) - len(FEATURE_NAMES)
        context_profile["selectionModel"]["featureNames"] = list(
            CONTEXT_SELECTION_FEATURE_NAMES
        )
        context_profile["selectionModel"]["weights"].extend([0.0] * extension)
        context_profile["selectionModel"]["means"].extend([0.0] * extension)
        context_profile["selectionModel"]["scales"].extend([1.0] * extension)
        old_scores = selection_scores(notes, LEARNED_PROFILE)
        context_scores = selection_scores(notes, context_profile)
        for old, new in zip(old_scores, context_scores):
            self.assertAlmostEqual(old, new, places=12)

    def test_harmonic_features_keep_relative_chord_evidence_after_octave_shift(self):
        original = normalize_source_notes(
            [
                note(43, 0.0, "electric_bass"),
                note(55, 0.0, "clean_electric_guitar"),
                note(58, 0.0, "string_ensemble"),
            ]
        )
        transposed = normalize_source_notes(
            [
                note(55, 0.0, "electric_bass"),
                note(67, 0.0, "clean_electric_guitar"),
                note(70, 0.0, "string_ensemble"),
            ]
        )
        original_rows = selection_feature_rows(
            original, HARMONIC_LEFT_HAND_SELECTION_FEATURE_NAMES
        )
        transposed_rows = selection_feature_rows(
            transposed, HARMONIC_LEFT_HAND_SELECTION_FEATURE_NAMES
        )
        names = list(HARMONIC_LEFT_HAND_SELECTION_FEATURE_NAMES)
        relative_names = [
            name
            for name in names
            if name.startswith("interval_above_")
            or name.startswith("estimated_chord_")
            or name in {
                "wide_pitch_class_support",
                "wide_pitch_class_rank",
                "wide_pitch_class_dominance",
                "estimated_bass_root_match",
            }
        ]
        for original_row, transposed_row in zip(original_rows, transposed_rows):
            for feature_name in relative_names:
                index = names.index(feature_name)
                self.assertAlmostEqual(
                    original_row[index], transposed_row[index], places=12
                )

    def test_small_mlp_selection_model_is_supported_without_runtime_numpy(self):
        notes = normalize_source_notes(
            [
                note(60, 0.0, "clean_electric_guitar", velocity=0.25),
                note(64, 1.0, "clean_electric_guitar", velocity=0.90),
            ]
        )
        velocity_index = list(FEATURE_NAMES).index("velocity")
        hidden_weights = [[0.0] for _name in FEATURE_NAMES]
        hidden_weights[velocity_index][0] = 4.0
        profile = {
            "selectionModel": {
                "type": "standardized-mlp-left-hand-ranker-v1",
                "featureNames": list(FEATURE_NAMES),
                "means": [0.0] * len(FEATURE_NAMES),
                "scales": [1.0] * len(FEATURE_NAMES),
                "hiddenWeights": hidden_weights,
                "hiddenBiases": [0.0],
                "outputWeights": [2.0],
                "outputBias": 0.0,
            }
        }
        scores = selection_scores(notes, profile)
        self.assertGreater(scores[1], scores[0])
        self.assertEqual(len(LEFT_HAND_SELECTION_FEATURE_NAMES), 54)

    def test_conditional_selector_changes_only_requested_family_and_register(self):
        notes = normalize_source_notes(
            [
                note(64, 0.0, "clean_electric_guitar", velocity=0.9),
                note(64, 0.1, "voice", velocity=0.9),
                note(55, 0.2, "clean_electric_guitar", velocity=0.9),
            ]
        )
        velocity_index = list(FEATURE_NAMES).index("velocity")
        base_weights = [0.0] * len(FEATURE_NAMES)
        alternate_weights = [0.0] * len(FEATURE_NAMES)
        alternate_weights[velocity_index] = 4.0
        model = lambda weights: {
            "featureNames": list(FEATURE_NAMES),
            "weights": weights,
            "means": [0.0] * len(FEATURE_NAMES),
            "scales": [1.0] * len(FEATURE_NAMES),
            "threshold": 0.5,
        }
        profile = {
            "selectionModel": model(base_weights),
            "decoder": {
                "conditionalSelectionBlend": {
                    "enabled": True,
                    "alternativeShare": 1.0,
                    "minimumSourceMidi": 60,
                    "maximumSourceMidi": 84,
                    "sourceFamilies": ["guitar"],
                    "selectionModel": model(alternate_weights),
                }
            },
        }
        base = selection_scores(notes, profile)
        blended, diagnostics = contextual_selection_scores(notes, profile)
        self.assertGreater(blended[0], base[0])
        self.assertEqual(blended[1:], base[1:])
        self.assertEqual(diagnostics["eligibleNotes"], 1)
        self.assertEqual(diagnostics["changedScores"], 1)

    def test_conditional_selector_centers_each_model_on_its_own_threshold(self):
        notes = normalize_source_notes([note(64, 0.0, "clean_electric_guitar")])
        base_weights = [0.0] * len(FEATURE_NAMES)
        alternate_weights = [0.0] * len(FEATURE_NAMES)
        alternate_weights[0] = 1.38629436112
        profile = {
            "selectionModel": {
                "featureNames": list(FEATURE_NAMES),
                "weights": base_weights,
                "means": [0.0] * len(FEATURE_NAMES),
                "scales": [1.0] * len(FEATURE_NAMES),
                "threshold": 0.5,
            },
            "decoder": {
                "conditionalSelectionBlend": {
                    "enabled": True,
                    "alternativeShare": 1.0,
                    "minimumSourceMidi": 60,
                    "sourceFamilies": ["guitar"],
                    "selectionModel": {
                        "featureNames": list(FEATURE_NAMES),
                        "weights": alternate_weights,
                        "means": [0.0] * len(FEATURE_NAMES),
                        "scales": [1.0] * len(FEATURE_NAMES),
                        "threshold": 0.8,
                    },
                }
            },
        }
        blended, _diagnostics = contextual_selection_scores(notes, profile)
        self.assertAlmostEqual(blended[0], 0.5, places=6)

    def test_conditional_reranker_preserves_frozen_notes_and_slot_count(self):
        notes = normalize_source_notes(
            [
                note(62, 0.0, "clean_electric_guitar", velocity=0.8),
                note(64, 0.1, "clean_electric_guitar", velocity=0.7),
                note(67, 0.2, "voice", velocity=0.9),
            ]
        )
        base_selected = [(notes[0], 0.8), (notes[2], 0.9)]
        profile = {
            "decoder": {
                "windowSeconds": 0.5,
                "conditionalSelectionBlend": {
                    "enabled": True,
                    "minimumSourceMidi": 60,
                    "maximumSourceMidi": 84,
                    "sourceFamilies": ["guitar"],
                },
            }
        }
        selected, diagnostics = _rerank_conditional_selection_slots(
            notes, base_selected, [0.2, 0.95, 0.1], profile, "full"
        )
        selected_indices = {item[0]["sourceIndex"] for item in selected}
        self.assertEqual(len(selected), 2)
        self.assertIn(notes[1]["sourceIndex"], selected_indices)
        self.assertIn(notes[2]["sourceIndex"], selected_indices)
        self.assertNotIn(notes[0]["sourceIndex"], selected_indices)
        self.assertEqual(diagnostics["eligibleSlots"], 1)
        self.assertEqual(diagnostics["replacedSlots"], 1)

    def test_conditional_onset_reranker_cannot_move_a_slot_to_another_beat(self):
        notes = normalize_source_notes(
            [
                note(62, 0.0, "clean_electric_guitar", velocity=0.8),
                note(64, 0.02, "clean_electric_guitar", velocity=0.7),
                note(67, 0.3, "clean_electric_guitar", velocity=0.9),
                note(69, 0.0, "voice", velocity=0.9),
            ]
        )
        base_guitar = next(
            item for item in notes if item["instrument"] == "clean_electric_guitar" and item["midi"] == 62
        )
        nearby_guitar = next(item for item in notes if item["midi"] == 64)
        later_guitar = next(item for item in notes if item["midi"] == 67)
        voice = next(item for item in notes if item["instrument"] == "voice")
        base_selected = [(base_guitar, 0.8), (voice, 0.9)]
        profile = {
            "decoder": {
                "conditionalSelectionBlend": {
                    "enabled": True,
                    "minimumSourceMidi": 60,
                    "maximumSourceMidi": 84,
                    "sourceFamilies": ["guitar"],
                    "onsetSlotWindowSeconds": 0.035,
                }
            }
        }
        scores = [
            0.95 if item is nearby_guitar else 1.0 if item is later_guitar else 0.2
            for item in notes
        ]
        selected, diagnostics = _rerank_conditional_onset_slots(
            notes, base_selected, scores, profile, "full"
        )
        selected_indices = {item[0]["sourceIndex"] for item in selected}
        self.assertIn(nearby_guitar["sourceIndex"], selected_indices)
        self.assertNotIn(later_guitar["sourceIndex"], selected_indices)
        self.assertIn(voice["sourceIndex"], selected_indices)
        self.assertEqual(len(selected), len(base_selected))
        self.assertTrue(diagnostics["preservedBaseOnsetCounts"])


class LeftHandAccompanimentTests(unittest.TestCase):
    @staticmethod
    def linear_midi_model() -> dict:
        weights = [0.0] * len(FEATURE_NAMES)
        weights[list(FEATURE_NAMES).index("midi_centered")] = 5.0
        return {
            "type": "standardized-logistic-left-hand-ranker-v2",
            "featureNames": list(FEATURE_NAMES),
            "weights": weights,
            "means": [0.0] * len(FEATURE_NAMES),
            "scales": [1.0] * len(FEATURE_NAMES),
            "threshold": 0.5,
        }

    def test_chord_completion_cannot_interrupt_frozen_melody_sustain(self):
        source = {
            "notes": [
                note(67, 0.30, "clean_electric_guitar", duration=0.4),
                note(70, 0.30, "string_ensemble", duration=0.4),
            ]
        }
        melody = {
            **note(70, 0.0, "acoustic_piano", duration=1.0),
            "sourceInstrument": "voice",
            "arrangementRole": "melody",
            "hand": "right",
            "scoreDuration": 1.0,
            "sourceIndex": 99,
        }
        accompaniment = {
            **note(67, 0.30, "acoustic_piano", duration=0.4),
            "sourceInstrument": "clean_electric_guitar",
            "arrangementRole": "harmony",
            "hand": "left",
            "sourceIndex": 0,
            "selectionProbability": 0.5,
        }
        profile = {
            "decoder": {
                "leftHandAccompaniment": {
                    "enabled": True,
                    "sourceMode": "frozen-baseline",
                    "handSplitMidi": 72,
                    "targetKeepRatio": 1.0,
                    "windowSeconds": 4.0,
                    "minimumNotesPerWindow": 1,
                    "maximumNotesPerOnset": 1,
                    "minimumSamePitchGapSeconds": 0.18,
                    "fastRetriggerKeepShare": 1.0,
                    "removeVoiceBelowSplit": False,
                    "preserveMelodyBelowSplit": True,
                    "chordCompletionModel": self.linear_midi_model(),
                    "chordCompletionRadiusSeconds": 0.08,
                }
            }
        }

        result, diagnostics = refine_left_hand_accompaniment(
            [melody, accompaniment], source, profile
        )

        protected = next(item for item in result if item["arrangementRole"] == "melody")
        completed = next(item for item in result if item["arrangementRole"] == "harmony")
        self.assertEqual(protected["midi"], 70)
        self.assertEqual(protected["duration"], 1.0)
        self.assertNotEqual(completed["midi"], 70)
        self.assertTrue(diagnostics["chordCompletionModelApplied"])

    def test_chord_completion_preserves_deliberate_octave_doubling(self):
        source = {
            "notes": [
                note(48, 0.0, "electric_bass", duration=0.4),
                note(60, 0.0, "clean_electric_guitar", duration=0.4),
                note(70, 0.0, "string_ensemble", duration=0.4),
            ]
        }
        accompaniment = [
            {
                **note(midi, 0.0, "acoustic_piano", duration=0.4),
                "sourceInstrument": "clean_electric_guitar",
                "arrangementRole": "harmony",
                "hand": "left",
                "sourceIndex": index,
                "selectionProbability": 0.5,
            }
            for index, midi in enumerate((48, 60))
        ]
        profile = {
            "decoder": {
                "leftHandAccompaniment": {
                    "enabled": True,
                    "sourceMode": "frozen-baseline",
                    "handSplitMidi": 72,
                    "targetKeepRatio": 1.0,
                    "windowSeconds": 4.0,
                    "minimumNotesPerWindow": 1,
                    "maximumNotesPerOnset": 2,
                    "minimumSamePitchGapSeconds": 0.18,
                    "fastRetriggerKeepShare": 1.0,
                    "removeVoiceBelowSplit": False,
                    "preserveMelodyBelowSplit": True,
                    "chordCompletionModel": self.linear_midi_model(),
                    "chordCompletionRadiusSeconds": 0.08,
                    "chordCompletionPreserveOctaveDoublings": True,
                }
            }
        }

        result, diagnostics = refine_left_hand_accompaniment(
            accompaniment, source, profile
        )

        self.assertEqual(sorted(item["midi"] for item in result), [48, 60])
        self.assertEqual(diagnostics["chordCompletionChangedNotes"], 0)
        self.assertEqual(diagnostics["chordCompletionPreservedOctaveOnsets"], 1)

    def test_supporting_tone_ranker_changes_only_the_non_bass_slot(self):
        source = {
            "notes": [
                note(48, 0.0, "clean_electric_guitar", duration=0.4),
                note(52, 0.0, "clean_electric_guitar", duration=0.4),
                note(67, 0.0, "clean_electric_guitar", duration=0.4),
            ]
        }
        accompaniment = [
            {
                **note(midi, 0.0, "acoustic_piano", duration=0.4),
                "sourceInstrument": "clean_electric_guitar",
                "arrangementRole": "harmony",
                "hand": "left",
                "sourceIndex": index,
                "selectionProbability": 0.5 + 0.1 * index,
            }
            for index, midi in enumerate((48, 52, 67))
        ]
        protected = {
            **note(76, 0.0, "acoustic_piano", duration=0.4),
            "sourceInstrument": "voice",
            "arrangementRole": "melody",
            "hand": "right",
            "sourceIndex": 99,
            "selectionProbability": 0.9,
        }
        supporting_model = self.linear_midi_model()
        midi_index = supporting_model["featureNames"].index("midi_centered")
        supporting_model["weights"][midi_index] = -5.0
        profile = {
            "decoder": {
                "leftHandAccompaniment": {
                    "enabled": True,
                    "sourceMode": "frozen-baseline",
                    "handSplitMidi": 72,
                    "targetKeepRatio": 1.0,
                    "windowSeconds": 4.0,
                    "minimumNotesPerWindow": 1,
                    "maximumNotesPerOnset": 2,
                    "minimumSamePitchGapSeconds": 0.18,
                    "fastRetriggerKeepShare": 1.0,
                    "removeVoiceBelowSplit": False,
                    "preserveMelodyBelowSplit": True,
                    "supportingToneSelectionModel": supporting_model,
                    "supportingToneAlternativeShare": 1.0,
                }
            }
        }

        result, diagnostics = refine_left_hand_accompaniment(
            [*accompaniment, protected], source, profile
        )

        self.assertEqual(sorted(item["midi"] for item in result), [48, 52, 76])
        self.assertEqual(next(item for item in result if item["midi"] == 76)["sourceIndex"], 99)
        self.assertTrue(diagnostics["supportingToneSelectionModelApplied"])
        self.assertEqual(diagnostics["outputLeftHandNotes"], 2)
        self.assertEqual(diagnostics["outputRightHandNotes"], 1)

    def test_absent_supporting_tone_model_does_not_add_fallback_ranking_fields(self):
        source = {"notes": [note(48, 0.0, "clean_electric_guitar")]}
        accompaniment = {
            **note(48, 0.0, "acoustic_piano"),
            "sourceInstrument": "clean_electric_guitar",
            "arrangementRole": "harmony",
            "hand": "left",
            "sourceIndex": 0,
            "selectionProbability": 0.5,
        }
        profile = {
            "decoder": {
                "leftHandAccompaniment": {
                    "enabled": True,
                    "sourceMode": "frozen-baseline",
                    "handSplitMidi": 72,
                    "targetKeepRatio": 1.0,
                    "windowSeconds": 4.0,
                    "maximumNotesPerOnset": 2,
                    "minimumSamePitchGapSeconds": 0.18,
                    "fastRetriggerKeepShare": 1.0,
                }
            }
        }

        result, diagnostics = refine_left_hand_accompaniment(
            [accompaniment], source, profile
        )

        self.assertNotIn("leftHandSupportingToneRankingScore", result[0])
        self.assertFalse(diagnostics["supportingToneSelectionModelApplied"])


class PianoLegatoTests(unittest.TestCase):
    def test_authored_register_defaults_keep_notes_in_pianella_hand_bands(self):
        source = [
            {**note(39, 0.0, 'electric_bass'), 'arrangementRole': 'bass', 'sourceIndex': 0},
            {**note(58, 0.2, 'guitar'), 'arrangementRole': 'harmony', 'sourceIndex': 1},
            {**note(60, 0.0, 'voice'), 'arrangementRole': 'melody', 'sourceIndex': 2},
            {**note(84, 0.2, 'voice'), 'arrangementRole': 'melody', 'sourceIndex': 3},
        ]

        compacted, diagnostics = compact_authored_two_hand_register(
            source,
            {'targetUpperNoteShare': 0.5},
        )

        self.assertEqual([item['midi'] for item in compacted], [39, 58, 60, 84])
        self.assertEqual(
            diagnostics['preferredOctaveShiftsSemitones'],
            {
                'melodyUpper': 0,
                'bassLower': 0,
                'harmonyUpper': 0,
                'harmonyLower': 0,
            },
        )
        self.assertTrue(all(34 <= item['midi'] <= 58 for item in compacted if item['hand'] == 'left'))
        self.assertTrue(all(60 <= item['midi'] <= 84 for item in compacted if item['hand'] == 'right'))

    def test_authored_register_can_preserve_native_harmony_without_target_share_forcing(self):
        source = [
            {**note(39, 0.0, 'electric_bass'), 'arrangementRole': 'bass', 'sourceIndex': 0},
            {**note(51, 0.0, 'guitar'), 'arrangementRole': 'harmony', 'sourceIndex': 1},
            {**note(67, 0.0, 'guitar'), 'arrangementRole': 'harmony', 'sourceIndex': 2},
            {**note(72, 0.0, 'voice'), 'arrangementRole': 'melody', 'sourceIndex': 3},
        ]

        compacted, diagnostics = compact_authored_two_hand_register(
            source,
            {
                'targetUpperNoteShare': 0.75,
                'harmonyHandAssignment': 'preserve-native-register',
                'lower': {'minimumMidi': 34, 'maximumMidi': 58},
                'upper': {'minimumMidi': 60, 'maximumMidi': 84},
            },
        )

        self.assertEqual([item['midi'] for item in compacted], [39, 51, 67, 72])
        self.assertEqual([item['hand'] for item in compacted], ['left', 'left', 'right', 'right'])
        self.assertEqual(
            diagnostics['harmonyHandAssignment'], 'preserve-native-register'
        )
        self.assertEqual(diagnostics['actualUpperNoteShare'], 0.5)

    def test_authored_two_hand_register_preserves_pitch_classes_and_staff_share(self):
        source = [
            {**note(56, 0.0, 'voice'), 'arrangementRole': 'melody', 'sourceIndex': 0},
            {**note(66, 0.2, 'voice'), 'arrangementRole': 'melody', 'sourceIndex': 1},
            {**note(35, 0.0, 'electric_bass'), 'arrangementRole': 'bass', 'sourceIndex': 2},
            {**note(47, 0.2, 'electric_bass'), 'arrangementRole': 'bass', 'sourceIndex': 3},
            {**note(58, 0.0, 'guitar'), 'arrangementRole': 'harmony', 'sourceIndex': 4},
            {**note(64, 0.2, 'guitar'), 'arrangementRole': 'harmony', 'sourceIndex': 5},
        ]
        compacted, diagnostics = compact_authored_two_hand_register(
            source,
            {
                'targetUpperNoteShare': 0.5,
                'lower': {'minimumMidi': 40, 'maximumMidi': 64},
                'upper': {'minimumMidi': 59, 'maximumMidi': 81},
                'preferredOctaveShiftsSemitones': {
                    'melodyUpper': 12,
                    'bassLower': 12,
                    'harmonyUpper': 0,
                    'harmonyLower': 12,
                },
            },
        )

        self.assertEqual([item['midi'] % 12 for item in compacted], [item['midi'] % 12 for item in source])
        self.assertEqual(sum(item['hand'] == 'right' for item in compacted), 3)
        self.assertEqual(sum(item['hand'] == 'left' for item in compacted), 3)
        self.assertTrue(all(59 <= item['midi'] <= 81 for item in compacted if item['hand'] == 'right'))
        self.assertTrue(all(40 <= item['midi'] <= 64 for item in compacted if item['hand'] == 'left'))
        self.assertEqual(diagnostics['actualUpperNoteShare'], 0.5)

    def test_authored_duration_style_transfers_short_and_long_hold_vocabulary(self):
        source = [
            {**note(48, 0.0, 'acoustic_piano', duration=0.10), 'hand': 'left'},
            {**note(52, 0.2, 'acoustic_piano', duration=0.40), 'hand': 'left'},
            {**note(64, 0.4, 'acoustic_piano', duration=0.12), 'hand': 'right'},
            {**note(67, 0.6, 'acoustic_piano', duration=0.50), 'hand': 'right'},
        ]
        styled, diagnostics = apply_authored_duration_style(
            source,
            {
                'durationGrid': {
                    'enabled': True,
                    'writtenReleaseRatio': 0.95,
                    'maximumUnits': 8,
                    'lowerHistogram': [
                        {'units': 1, 'share': 0.5},
                        {'units': 4, 'share': 0.5},
                    ],
                    'upperHistogram': [
                        {'units': 1, 'share': 0.5},
                        {'units': 4, 'share': 0.5},
                    ],
                }
            },
        )

        durations = sorted(item['duration'] for item in styled)
        self.assertEqual(durations, [0.19, 0.19, 0.76, 0.76])
        self.assertTrue(diagnostics['applied'])
        self.assertAlmostEqual(diagnostics['estimatedSubdivisionSeconds'], 0.2)

    def test_preserves_the_score_register_and_folds_only_the_low_edge(self):
        compacted, diagnostics = compact_pianella_register(
            [
                {**note(21, 0, 'acoustic_piano'), 'note': 'A0'},
                {**note(36, 1, 'acoustic_piano'), 'note': 'C2'},
                {**note(60, 2, 'acoustic_piano'), 'note': 'C4'},
            ]
        )

        self.assertEqual([item['midi'] for item in compacted], [33, 36, 60])
        self.assertTrue(all(
            COMPACT_PIANO_MIN_MIDI <= item['midi'] <= COMPACT_PIANO_MAX_MIDI
            for item in compacted
        ))
        self.assertEqual(diagnostics['globalShiftSemitones'], 0)
        self.assertEqual(diagnostics['shiftedNotes'], 1)
        self.assertEqual(diagnostics['edgeFoldedNotes'], 1)
        self.assertEqual(
            diagnostics['semitoneShiftCounts'],
            {'12': 1},
        )

    def test_keeps_mid_and_upper_register_notes_in_their_authored_octaves(self):
        compacted, diagnostics = compact_pianella_register(
            [
                {**note(48, 0, 'acoustic_piano'), 'note': 'C3'},
                {**note(84, 1, 'acoustic_piano'), 'note': 'C6'},
            ]
        )

        self.assertEqual(diagnostics['globalShiftSemitones'], 0)
        self.assertEqual([item['midi'] for item in compacted], [48, 84])

    def test_folds_unavoidable_edges_instead_of_dropping_them(self):
        compacted, diagnostics = compact_pianella_register(
            [
                {**note(21, 0, 'acoustic_piano'), 'note': 'A0'},
                {**note(60, 1, 'acoustic_piano'), 'note': 'C4'},
                {**note(108, 2, 'acoustic_piano'), 'note': 'C8'},
            ]
        )

        self.assertEqual(diagnostics['globalShiftSemitones'], 0)
        self.assertEqual(diagnostics['edgeFoldedNotes'], 1)
        self.assertEqual([item['midi'] for item in compacted], [33, 60, 108])

    def test_can_disable_global_lift_without_losing_a0(self):
        compacted, diagnostics = compact_pianella_register(
            [
                {**note(21, 0, 'acoustic_piano'), 'note': 'A0'},
                {**note(60, 1, 'acoustic_piano'), 'note': 'C4'},
                {**note(84, 2, 'acoustic_piano'), 'note': 'C6'},
            ],
            preferred_global_shift_semitones=0,
        )

        self.assertEqual([item['midi'] for item in compacted], [33, 60, 84])
        self.assertEqual(diagnostics['preferredShiftSemitones'], 0)
        self.assertEqual(diagnostics['globalShiftSemitones'], 0)
        self.assertEqual(diagnostics['edgeFoldedNotes'], 1)

    def test_legacy_positive_shift_request_cannot_restore_the_blanket_lift(self):
        compacted, diagnostics = compact_pianella_register(
            [
                {**note(33, 0, 'acoustic_piano'), 'note': 'A1'},
                {**note(60, 1, 'acoustic_piano'), 'note': 'C4'},
                {**note(108, 2, 'acoustic_piano'), 'note': 'C8'},
            ],
            preferred_global_shift_semitones=24,
        )

        self.assertEqual([item['midi'] for item in compacted], [33, 60, 108])
        self.assertEqual(diagnostics['preferredShiftSemitones'], 0)
        self.assertEqual(diagnostics['globalShiftSemitones'], 0)

    def test_shapes_connected_harmony_with_a_long_release(self):
        notes = [note(72, 0, 'voice', duration=0.2), note(43, 0, 'electric_bass')]
        for index, onset in enumerate((0.0, 0.3, 0.6, 0.9)):
            for midi in (60, 64, 67, 71):
                notes.append(
                    note(midi + index, onset, 'clean_electric_guitar', duration=0.12)
                )

        result = arrange_payload({'title': 'Legato fixture', 'notes': notes}, 'full')
        harmony = [
            arranged
            for arranged in result['notes']
            if arranged['arrangementRole'] == 'harmony'
        ]

        self.assertEqual(result['pianoArrangement']['version'], 5)
        self.assertGreater(result['pianoArrangement']['legatoExtendedNotes'], 0)
        self.assertEqual(result['performance']['defaultAutoplayReleaseSeconds'], 0.62)
        self.assertTrue(any(arranged['duration'] >= 0.4 for arranged in harmony))
        self.assertGreaterEqual(len(harmony), 8)

    def test_places_melody_in_front_and_softens_left_hand(self):
        notes = []
        for index in range(36):
            onset = index * 0.24
            notes.extend(
                [
                    note(43, onset, 'electric_bass', duration=0.5, velocity=0.78),
                    note(55, onset, 'clean_electric_guitar', duration=0.35, velocity=0.78),
                    note(64, onset, 'clean_electric_guitar', duration=0.35, velocity=0.78),
                    note(72 + index % 3, onset, 'voice', duration=0.42, velocity=0.78),
                ]
            )

        result = arrange_payload({'title': 'Melody balance fixture', 'notes': notes}, 'full')
        melody = [item for item in result['notes'] if item['arrangementRole'] == 'melody']
        bass = [item for item in result['notes'] if item['arrangementRole'] == 'bass']
        left = [item for item in result['notes'] if item['midi'] < 60]
        right = [item for item in result['notes'] if item['midi'] >= 60]
        expression = result['pianoArrangement']['expression']

        self.assertTrue(melody)
        self.assertTrue(bass)
        self.assertGreater(min(item['velocity'] for item in melody), max(item['velocity'] for item in bass))
        self.assertGreater(
            sum(item['velocity'] for item in right) / len(right),
            sum(item['velocity'] for item in left) / len(left),
        )
        self.assertGreater(expression['rightToLeftVelocityRatio'], 1.2)
        self.assertTrue(result['performance']['melodyForwardDynamics'])
        self.assertEqual(result['performance']['profile'], 'polymath-piano-arranger-v5')
        self.assertEqual(result['arrangementProfile'], 'piano-reduction-with-physical-performance-v5')

    def test_separates_written_duration_key_hold_release_and_inferred_pedal(self):
        notes = []
        for index, onset in enumerate((0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)):
            notes.extend(
                [
                    note(48 + index % 4, onset, 'electric_bass', duration=0.45),
                    note(60 + index % 5, onset, 'clean_electric_guitar', duration=0.45),
                    note(72 + index % 3, onset, 'voice', duration=0.45),
                ]
            )

        result = arrange_payload(
            {
                'title': 'Physical performance fixture',
                'instrument': 'band',
                'bpm': 120,
                'notes': notes,
            },
            'full',
        )

        self.assertEqual(result['instrument'], 'piano')
        self.assertTrue(result['pedals'])
        self.assertEqual(result['pedals'], result['pedalEvents'])
        self.assertTrue(all(event['inferred'] for event in result['pedals']))
        self.assertTrue(all('scoreDuration' in item for item in result['notes']))
        self.assertTrue(all('audioDuration' in item for item in result['notes']))
        self.assertTrue(all('releaseSeconds' in item for item in result['notes']))
        self.assertTrue(all(item['scoreDuration'] == item['duration'] for item in result['notes']))
        self.assertTrue(all(item['visualDuration'] == item['audioDuration'] for item in result['notes']))
        self.assertEqual(result['performance']['visualDurationPolicy'], 'physical-key-hold')
        self.assertTrue(result['pianoArrangement']['physicalPerformance']['writtenAndPhysicalDurationsSeparated'])
        self.assertTrue(all(item['articulation'] == 'legato' for item in result['notes']))
        self.assertTrue(all('maximumPhysicalHoldSeconds' in item for item in result['notes']))

    def test_preserved_piano_gets_register_balance_without_clipping(self):
        notes = []
        for index in range(60):
            onset = index * 0.18
            notes.extend(
                [
                    note(48 + index % 4, onset, 'acoustic_piano', duration=0.45, velocity=0.99),
                    note(67 + index % 5, onset, 'acoustic_piano', duration=0.38, velocity=0.99),
                ]
            )

        result = arrange_payload({'title': 'Piano register fixture', 'notes': notes}, 'full')
        left = [item for item in result['notes'] if item['hand'] == 'left']
        right = [item for item in result['notes'] if item['hand'] == 'right']

        self.assertEqual(result['pianoArrangement']['profile'], 'acoustic-piano-preserve')
        self.assertGreater(min(item['velocity'] for item in right), max(item['velocity'] for item in left))
        self.assertLess(max(item['velocity'] for item in right), 1.0)
        self.assertGreater(
            result['pianoArrangement']['expression']['rightToLeftVelocityRatio'],
            1.2,
        )

    def test_profile_can_duck_accompaniment_while_a_vocal_melody_is_active(self):
        notes = [
            {
                **note(72, 1.0, "voice", duration=1.0, velocity=0.76),
                "arrangementRole": "melody",
                "hand": "right",
            },
            {
                **note(43, 1.0, "electric_bass", duration=0.7, velocity=0.76),
                "arrangementRole": "bass",
                "hand": "left",
            },
            {
                **note(64, 1.0, "acoustic_guitar", duration=0.7, velocity=0.76),
                "arrangementRole": "harmony",
                "hand": "right",
            },
            {
                **note(43, 4.0, "electric_bass", duration=0.7, velocity=0.76),
                "arrangementRole": "bass",
                "hand": "left",
            },
        ]
        default, _ = shape_melody_forward_expression(notes)
        balanced, diagnostics = shape_melody_forward_expression(
            notes,
            {
                "decoder": {
                    "melodyForwardBalance": {
                        "enabled": True,
                        "melodyVelocityRange": [0.88, 0.99],
                        "bassVelocityRange": [0.28, 0.42],
                        "melodyGain": 1.04,
                        "leftGainDuringMelody": 0.75,
                        "rightGainDuringMelody": 0.9,
                    }
                }
            },
        )

        self.assertGreater(balanced[0]["velocity"], default[0]["velocity"])
        self.assertLess(balanced[1]["velocity"], balanced[3]["velocity"])
        self.assertLess(balanced[2]["velocity"], default[2]["velocity"])
        self.assertEqual(diagnostics["melodyBalance"]["duckedLeftNotes"], 1)
        self.assertEqual(diagnostics["melodyBalance"]["duckedRightNotes"], 1)


class LearnedDurationTests(unittest.TestCase):
    def test_extension_only_duration_model_never_shortens_source_hold(self):
        source = normalize_source_notes(
            [note(64, 0.0, "voice", duration=0.5)]
        )[0]
        profile = {
            "durationModel": {"predictionWeight": 1.0},
            "decoder": {
                "sourceDurationWeight": 1.0,
                "durationExtensionOnly": True,
            },
        }

        rendered = _render_note(source, 0.9, profile, predicted_duration=0.1)

        self.assertEqual(rendered["duration"], 0.5)

    def test_register_model_uses_supported_group_and_falls_back_for_unseen_group(self):
        profile = {
            "roles": {
                "melody": {"minimumMidi": 55, "maximumMidi": 88},
                "harmony": {"minimumMidi": 45, "maximumMidi": 84},
            },
            "registerModel": {
                "type": "hierarchical-categorical-octave-shift-v1",
                "fallbackShiftSemitones": 12,
                "uncertainShiftSemitones": 0,
                "minimumSpecificWeightedSupport": 18,
                "minimumRoleWeightedSupport": 36,
                "minimumConfidence": 0.55,
                "groups": {
                    "melody|voice|high": {
                        "shiftSemitones": 0,
                        "weightedSupport": 50,
                        "confidence": 0.8,
                    },
                    "harmony|guitar|high": {
                        "shiftSemitones": 12,
                        "weightedSupport": 50,
                        "confidence": 0.51,
                    },
                },
            },
        }
        voice = _render_note(
            normalize_source_notes([note(67, 0.0, "voice")])[0],
            0.9,
            profile,
        )
        guitar = _render_note(
            normalize_source_notes([note(60, 0.5, "acoustic_guitar")])[0],
            0.9,
            profile,
        )
        piano = _render_note(
            normalize_source_notes([note(60, 1.0, "acoustic_piano")])[0],
            0.9,
            profile,
        )

        adjusted, diagnostics = apply_learned_register_model(
            [voice, guitar, piano], profile
        )

        self.assertEqual([item["midi"] for item in adjusted], [67, 60, 72])
        self.assertEqual(
            adjusted[0]["learnedRegisterEvidenceKey"], "melody|voice|high"
        )
        self.assertEqual(adjusted[1]["learnedRegisterEvidenceKey"], "uncertain")
        self.assertEqual(adjusted[2]["learnedRegisterEvidenceKey"], "fallback")
        self.assertEqual(diagnostics["shiftCounts"], {"0": 2, "12": 1})

    def test_minimum_rendered_duration_only_extends_abnormally_short_notes(self):
        source = normalize_source_notes(
            [note(64, 0.0, "voice", duration=0.04)]
        )[0]
        profile = json.loads(json.dumps(LEARNED_PROFILE))
        profile.setdefault("decoder", {})["minimumRenderedDurationSeconds"] = 0.18

        rendered = _render_note(source, 0.9, profile)

        self.assertEqual(rendered["duration"], 0.18)

    def test_minimum_rendered_duration_can_be_scoped_to_a_role(self):
        profile = json.loads(json.dumps(LEARNED_PROFILE))
        profile.setdefault("decoder", {})[
            "minimumRenderedDurationByRole"
        ] = {"bass": 0.28}
        bass = _render_note(
            normalize_source_notes(
                [note(43, 0.0, "electric_bass", duration=0.08)]
            )[0],
            0.9,
            profile,
        )
        melody = _render_note(
            normalize_source_notes(
                [note(67, 0.0, "voice", duration=0.08)]
            )[0],
            0.9,
            profile,
        )

        self.assertEqual(bass["duration"], 0.28)
        self.assertLess(melody["duration"], 0.28)

    def test_minimum_rendered_duration_can_be_scoped_to_source_family(self):
        profile = json.loads(json.dumps(LEARNED_PROFILE))
        profile.setdefault("decoder", {})[
            "minimumRenderedDurationBySourceFamily"
        ] = {"bass": 0.4}
        bass = _render_note(
            normalize_source_notes(
                [note(43, 0.0, "electric_bass", duration=0.08)]
            )[0],
            0.9,
            profile,
        )
        guitar = _render_note(
            normalize_source_notes(
                [note(55, 0.0, "acoustic_guitar", duration=0.08)]
            )[0],
            0.9,
            profile,
        )

        self.assertEqual(bass["duration"], 0.4)
        self.assertLess(guitar["duration"], 0.4)

    def test_source_density_policy_interpolates_without_mutating_the_profile(self):
        profile = {
            'decoder': {
                'preCleanupDensityMultiplier': 1.8,
                'adaptiveSourceDensity': {
                    'enabled': True,
                    'lowSourceNotesPerSecond': 20,
                    'highSourceNotesPerSecond': 30,
                    'lowDensityMultiplier': 1.8,
                    'highDensityMultiplier': 2.0,
                    'lowDurationPredictionWeight': 0.0,
                    'highDurationPredictionWeight': 0.4,
                },
            },
            'durationModel': {'predictionWeight': 0.0},
        }

        effective, diagnostics = adapt_profile_to_source_density(
            profile, {'sourceNotesPerSecond': 25}
        )

        self.assertEqual(profile['decoder']['preCleanupDensityMultiplier'], 1.8)
        self.assertAlmostEqual(effective['decoder']['preCleanupDensityMultiplier'], 1.9)
        self.assertAlmostEqual(effective['durationModel']['predictionWeight'], 0.2)
        self.assertEqual(diagnostics['interpolationRatio'], 0.5)

    def test_sparse_vocal_physical_limits_do_not_affect_dense_or_instrumental_sources(self):
        profile = {
            "decoder": {
                "physicalPerformance": {
                    "maximumLegatoBridgeSeconds": {
                        "melody": 0.9,
                        "bass": 1.8,
                        "harmony": 2.4,
                    },
                    "maximumPhysicalHoldSeconds": {
                        "melody": 1.35,
                        "bass": 2.2,
                        "harmony": 2.6,
                    },
                    "adaptiveSourceDensity": {
                        "enabled": True,
                        "lowSourceNotesPerSecond": 5.0,
                        "highSourceNotesPerSecond": 10.0,
                        "minimumVoiceRatio": 0.05,
                        "maximumPianoRatio": 0.05,
                        "lowMaximumLegatoBridgeSeconds": {
                            "melody": 0.9,
                            "bass": 0.25,
                            "harmony": 0.8,
                        },
                        "highMaximumLegatoBridgeSeconds": {
                            "melody": 0.9,
                            "bass": 1.8,
                            "harmony": 2.4,
                        },
                        "lowMaximumPhysicalHoldSeconds": {
                            "melody": 1.35,
                            "bass": 2.2,
                            "harmony": 2.6,
                        },
                        "highMaximumPhysicalHoldSeconds": {
                            "melody": 1.35,
                            "bass": 2.2,
                            "harmony": 2.6,
                        },
                    },
                }
            }
        }
        sparse, sparse_diagnostics = adapt_physical_performance_to_source(
            profile,
            {"sourceNotesPerSecond": 5, "voiceRatio": 0.15, "pianoRatio": 0},
        )
        dense, dense_diagnostics = adapt_physical_performance_to_source(
            profile,
            {"sourceNotesPerSecond": 20, "voiceRatio": 0.15, "pianoRatio": 0},
        )
        instrumental, instrumental_diagnostics = adapt_physical_performance_to_source(
            profile,
            {"sourceNotesPerSecond": 5, "voiceRatio": 0, "pianoRatio": 0},
        )

        self.assertEqual(
            sparse["decoder"]["physicalPerformance"][
                "maximumLegatoBridgeSeconds"
            ]["bass"],
            0.25,
        )
        self.assertEqual(
            dense["decoder"]["physicalPerformance"][
                "maximumLegatoBridgeSeconds"
            ]["bass"],
            1.8,
        )
        self.assertEqual(
            instrumental["decoder"]["physicalPerformance"][
                "maximumLegatoBridgeSeconds"
            ]["bass"],
            1.8,
        )
        self.assertEqual(sparse_diagnostics["physicalPerformanceInterpolationRatio"], 0)
        self.assertEqual(dense_diagnostics["physicalPerformanceInterpolationRatio"], 1)
        self.assertTrue(
            instrumental_diagnostics["physicalPerformanceVoiceGateApplied"]
        )
        self.assertEqual(profile["decoder"]["physicalPerformance"]["maximumLegatoBridgeSeconds"]["bass"], 1.8)

    def test_selection_blend_adapts_to_factual_source_density(self):
        low_model = {
            'featureNames': ['bias', 'pitch'],
            'weights': [1.0, 3.0],
            'means': [0.0, 0.0],
            'scales': [1.0, 1.0],
            'threshold': 0.4,
        }
        high_model = {
            'featureNames': ['bias', 'pitch'],
            'weights': [3.0, 7.0],
            'means': [0.0, 0.0],
            'scales': [1.0, 1.0],
            'threshold': 0.6,
        }
        profile = {
            'selectionModel': low_model,
            'decoder': {
                'adaptiveSelectionBlend': {
                    'enabled': True,
                    'lowSourceNotesPerSecond': 20,
                    'highSourceNotesPerSecond': 30,
                    'lowSourceBaseShare': 0.95,
                    'highSourceBaseShare': 0.80,
                    'minimumVoiceRatioForAggressiveBlend': 0.10,
                    'lowSourceSelectionModel': low_model,
                    'highSourceSelectionModel': high_model,
                }
            },
        }

        effective, diagnostics = adapt_profile_to_source_density(
            profile, {'sourceNotesPerSecond': 25, 'voiceRatio': 0.20}
        )

        self.assertEqual(profile['selectionModel']['weights'], [1.0, 3.0])
        self.assertEqual(effective['selectionModel']['weights'], [2.0, 5.0])
        self.assertAlmostEqual(effective['selectionModel']['threshold'], 0.5)
        self.assertEqual(diagnostics['selectionInterpolationRatio'], 0.5)
        self.assertEqual(diagnostics['effectiveSelectionBaseShare'], 0.875)
        self.assertFalse(diagnostics['selectionVoiceGateApplied'])

        gated, gated_diagnostics = adapt_profile_to_source_density(
            profile, {'sourceNotesPerSecond': 35, 'voiceRatio': 0.01}
        )
        self.assertEqual(gated['selectionModel']['weights'], [1.0, 3.0])
        self.assertEqual(gated_diagnostics['selectionDensityInterpolationRatio'], 1.0)
        self.assertEqual(gated_diagnostics['selectionInterpolationRatio'], 0.0)
        self.assertTrue(gated_diagnostics['selectionVoiceGateApplied'])

    def test_duration_policy_can_use_source_hold_statistics_independently_of_density(self):
        profile = {
            'decoder': {
                'adaptiveSourceDensity': {
                    'enabled': True,
                    'lowSourceNotesPerSecond': 20,
                    'highSourceNotesPerSecond': 30,
                    'lowDensityMultiplier': 1.8,
                    'highDensityMultiplier': 2.0,
                    'lowDurationPredictionWeight': 0.0,
                    'highDurationPredictionWeight': 0.4,
                    'shortSourceMedianDurationSeconds': 0.17,
                    'longSourceMedianDurationSeconds': 0.20,
                },
            },
            'durationModel': {'predictionWeight': 0.0},
        }

        short, short_diagnostics = adapt_profile_to_source_density(
            profile,
            {'sourceNotesPerSecond': 25, 'sourceMedianDurationSeconds': 0.16},
        )
        long, long_diagnostics = adapt_profile_to_source_density(
            profile,
            {'sourceNotesPerSecond': 25, 'sourceMedianDurationSeconds': 0.21},
        )

        self.assertAlmostEqual(short['durationModel']['predictionWeight'], 0.4)
        self.assertAlmostEqual(long['durationModel']['predictionWeight'], 0.0)
        self.assertEqual(
            short_diagnostics['durationInterpolationSignal'],
            'non-percussive-source-median-duration',
        )
        self.assertEqual(short_diagnostics['durationInterpolationRatio'], 1.0)
        self.assertEqual(long_diagnostics['durationInterpolationRatio'], 0.0)

    def test_duration_policy_can_require_factual_voice_evidence(self):
        profile = {
            'decoder': {
                'adaptiveSourceDensity': {
                    'enabled': True,
                    'lowSourceNotesPerSecond': 20,
                    'highSourceNotesPerSecond': 30,
                    'lowDensityMultiplier': 1.8,
                    'highDensityMultiplier': 2.0,
                    'lowDurationPredictionWeight': 0.0,
                    'highDurationPredictionWeight': 0.4,
                    'shortSourceMedianDurationSeconds': 0.18,
                    'longSourceMedianDurationSeconds': 0.19,
                    'minimumVoiceRatioForDurationPrediction': 0.05,
                    'nonVocalDurationPredictionWeight': 0.0,
                    'shortSourceDurationWeight': 0.8,
                    'longSourceDurationWeight': 1.0,
                    'nonVocalSourceDurationWeight': 1.0,
                    'maximumPianoRatioForVocalAdaptation': 0.05,
                },
            },
            'durationModel': {'predictionWeight': 0.0},
        }

        vocal, vocal_diagnostics = adapt_profile_to_source_density(
            profile,
            {
                'sourceNotesPerSecond': 25,
                'sourceMedianDurationSeconds': 0.18,
                'voiceRatio': 0.12,
            },
        )
        instrumental, instrumental_diagnostics = adapt_profile_to_source_density(
            profile,
            {
                'sourceNotesPerSecond': 25,
                'sourceMedianDurationSeconds': 0.18,
                'voiceRatio': 0.0,
            },
        )
        piano_rich, piano_rich_diagnostics = adapt_profile_to_source_density(
            profile,
            {
                'sourceNotesPerSecond': 25,
                'sourceMedianDurationSeconds': 0.18,
                'voiceRatio': 0.12,
                'pianoRatio': 0.20,
            },
        )

        self.assertAlmostEqual(vocal['durationModel']['predictionWeight'], 0.4)
        self.assertAlmostEqual(instrumental['durationModel']['predictionWeight'], 0.0)
        self.assertAlmostEqual(piano_rich['durationModel']['predictionWeight'], 0.0)
        self.assertAlmostEqual(vocal['decoder']['sourceDurationWeight'], 0.8)
        self.assertAlmostEqual(instrumental['decoder']['sourceDurationWeight'], 1.0)
        self.assertAlmostEqual(piano_rich['decoder']['sourceDurationWeight'], 1.0)
        self.assertFalse(vocal_diagnostics['voiceGateApplied'])
        self.assertTrue(instrumental_diagnostics['voiceGateApplied'])
        self.assertTrue(piano_rich_diagnostics['voiceGateApplied'])

    def test_quota_backfill_tightens_only_for_dense_vocal_piano_light_mix(self):
        profile = {
            'decoder': {
                'quotaBackfillRatio': 1.0,
                'adaptiveSourceDensity': {
                    'enabled': True,
                    'lowSourceNotesPerSecond': 20,
                    'highSourceNotesPerSecond': 30,
                    'lowDensityMultiplier': 1.7,
                    'highDensityMultiplier': 1.7,
                    'lowDurationPredictionWeight': 0.0,
                    'highDurationPredictionWeight': 0.0,
                    'minimumVoiceRatioForDensityAdaptation': 0.05,
                    'nonVocalDensityMultiplier': 1.45,
                    'maximumPianoRatioForVocalAdaptation': 0.05,
                    'lowQuotaBackfillRatio': 1.0,
                    'highQuotaBackfillRatio': 0.65,
                    'nonVocalQuotaBackfillRatio': 1.0,
                },
            },
            'durationModel': {'predictionWeight': 0.0},
        }

        sparse, sparse_diagnostics = adapt_profile_to_source_density(
            profile,
            {'sourceNotesPerSecond': 20, 'voiceRatio': 0.10, 'pianoRatio': 0.0},
        )
        dense, dense_diagnostics = adapt_profile_to_source_density(
            profile,
            {'sourceNotesPerSecond': 30, 'voiceRatio': 0.10, 'pianoRatio': 0.0},
        )
        instrumental, instrumental_diagnostics = adapt_profile_to_source_density(
            profile,
            {'sourceNotesPerSecond': 30, 'voiceRatio': 0.0, 'pianoRatio': 0.0},
        )
        piano_rich, piano_rich_diagnostics = adapt_profile_to_source_density(
            profile,
            {'sourceNotesPerSecond': 30, 'voiceRatio': 0.10, 'pianoRatio': 0.20},
        )

        self.assertEqual(profile['decoder']['quotaBackfillRatio'], 1.0)
        self.assertAlmostEqual(sparse['decoder']['quotaBackfillRatio'], 1.0)
        self.assertAlmostEqual(dense['decoder']['quotaBackfillRatio'], 0.65)
        self.assertAlmostEqual(instrumental['decoder']['quotaBackfillRatio'], 1.0)
        self.assertAlmostEqual(piano_rich['decoder']['quotaBackfillRatio'], 1.0)
        self.assertFalse(sparse_diagnostics['quotaBackfillVoiceGateApplied'])
        self.assertFalse(dense_diagnostics['quotaBackfillVoiceGateApplied'])
        self.assertTrue(
            instrumental_diagnostics['quotaBackfillVoiceGateApplied']
        )
        self.assertTrue(piano_rich_diagnostics['quotaBackfillVoiceGateApplied'])

    def test_voice_gate_protects_instrumental_density_and_disables_vocal_doubling(self):
        profile = {
            'decoder': {
                'preCleanupDensityMultiplier': 1.45,
                'expandSparseHarmony': True,
                'adaptiveSourceDensity': {
                    'enabled': True,
                    'lowSourceNotesPerSecond': 20,
                    'highSourceNotesPerSecond': 30,
                    'lowDensityMultiplier': 1.8,
                    'highDensityMultiplier': 2.0,
                    'lowDurationPredictionWeight': 0.0,
                    'highDurationPredictionWeight': 0.4,
                    'minimumVoiceRatioForDensityAdaptation': 0.05,
                    'nonVocalDensityMultiplier': 1.45,
                    'minimumVoiceRatioForDisablingSparseExpansion': 0.05,
                    'maximumPianoRatioForVocalAdaptation': 0.05,
                },
            },
            'durationModel': {'predictionWeight': 0.0},
        }

        instrumental, instrumental_diagnostics = adapt_profile_to_source_density(
            profile,
            {'sourceNotesPerSecond': 35, 'voiceRatio': 0.0},
        )
        vocal, vocal_diagnostics = adapt_profile_to_source_density(
            profile,
            {'sourceNotesPerSecond': 35, 'voiceRatio': 0.10, 'pianoRatio': 0.0},
        )
        piano_rich, piano_rich_diagnostics = adapt_profile_to_source_density(
            profile,
            {'sourceNotesPerSecond': 35, 'voiceRatio': 0.10, 'pianoRatio': 0.20},
        )

        self.assertEqual(profile['decoder']['preCleanupDensityMultiplier'], 1.45)
        self.assertTrue(profile['decoder']['expandSparseHarmony'])
        self.assertAlmostEqual(
            instrumental['decoder']['preCleanupDensityMultiplier'], 1.45
        )
        self.assertTrue(instrumental['decoder']['expandSparseHarmony'])
        self.assertTrue(instrumental_diagnostics['densityVoiceGateApplied'])
        self.assertFalse(
            instrumental_diagnostics['sparseExpansionDisabledForVoice']
        )
        self.assertAlmostEqual(vocal['decoder']['preCleanupDensityMultiplier'], 2.0)
        self.assertFalse(vocal['decoder']['expandSparseHarmony'])
        self.assertFalse(vocal_diagnostics['densityVoiceGateApplied'])
        self.assertTrue(vocal_diagnostics['sparseExpansionDisabledForVoice'])
        self.assertAlmostEqual(
            piano_rich['decoder']['preCleanupDensityMultiplier'], 1.45
        )
        self.assertTrue(piano_rich['decoder']['expandSparseHarmony'])
        self.assertTrue(piano_rich_diagnostics['densityVoiceGateApplied'])
        self.assertFalse(
            piano_rich_diagnostics['sparseExpansionDisabledForVoice']
        )

    def test_duration_model_is_optional_and_decodes_without_an_ml_runtime(self):
        notes = normalize_source_notes([
            note(60, 0.0, 'voice', duration=0.2),
            note(48, 0.5, 'electric_bass', duration=0.3),
        ])
        self.assertEqual(duration_predictions(notes, {}), [None, None])

        weights = [0.0] * len(DURATION_FEATURE_NAMES)
        weights[0] = 0.7884573604  # log(1 + 1.2 seconds)
        profile = {
            'durationModel': {
                'weights': weights,
                'means': [0.0] * len(weights),
                'scales': [1.0] * len(weights),
                'minimumSeconds': 0.05,
                'maximumSeconds': 4.0,
            }
        }

        predictions = duration_predictions(notes, profile)

        self.assertEqual(len(predictions), 2)
        self.assertAlmostEqual(predictions[0], 1.2, places=4)
        self.assertAlmostEqual(predictions[1], 1.2, places=4)

    def test_threshold_can_disable_low_confidence_quota_backfill_for_research(self):
        notes = normalize_source_notes([
            note(60 + index, index * 0.02, 'clean_electric_guitar')
            for index in range(5)
        ])
        scores = [0.95, 0.80, 0.40, 0.30, 0.20]
        profile = {
            'style': {'targetNotesPerSecond': 8.0},
            'selectionModel': {'threshold': 0.75},
            'decoder': {
                'windowSeconds': 0.5,
                'preCleanupDensityMultiplier': 1.0,
                'quotaBackfillRatio': 0.0,
            },
        }

        strict = _pick_window_notes(notes, scores, profile, 'full')
        profile['decoder']['quotaBackfillRatio'] = 1.0
        filled = _pick_window_notes(notes, scores, profile, 'full')

        self.assertEqual(len(strict), 2)
        self.assertGreater(len(filled), len(strict))


class PianoArrangerTests(unittest.TestCase):
    def test_preserves_raw_source_indices_as_provenance(self):
        payload = {
            "title": "Source provenance fixture",
            "notes": [
                note(48, 0.0, "acoustic_piano"),
                note(10, 0.1, "acoustic_piano"),
                note(64, 0.2, "acoustic_piano"),
            ]
            + [
                note(52 + index % 12, 0.4 + index * 0.2, "acoustic_piano")
                for index in range(24)
            ],
        }

        result = arrange_payload(payload, "full")

        self.assertEqual(result["notes"][0]["sourceIndex"], 0)
        self.assertEqual(result["notes"][1]["sourceIndex"], 2)
        self.assertEqual(
            [item["sourceIndex"] for item in result["notes"]],
            [0, 2, *range(3, 27)],
        )

    def test_source_supported_cycle_preserves_octaves_and_varies_hand_texture(self):
        source = [
            note(midi, onset, "clean_electric_guitar", duration=0.15, velocity=0.62)
            for onset in (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05, 1.20)
            for midi in (51, 58, 63, 67)
        ]

        arranged, diagnostics = source_supported_cyclic_harmony(
            source,
            {
                "minimumRunGroups": 8,
                "minimumGuitarShare": 0.7,
                "minimumGridCoverage": 0.8,
            },
        )
        groups = _group_test_notes(arranged)

        self.assertTrue(diagnostics["applied"])
        self.assertAlmostEqual(diagnostics["estimatedPulseSeconds"], 0.15, places=3)
        self.assertEqual([item["midi"] for item in groups[0]], [39, 63, 67])
        self.assertEqual([item["midi"] for item in groups[1]], [58])
        self.assertEqual([item["midi"] for item in groups[2]], [46, 63, 67])
        self.assertEqual([item["midi"] for item in groups[3]], [51, 63, 67])
        self.assertEqual([item["midi"] for item in groups[5]], [63, 67])

    def test_source_supported_cycle_falls_back_when_tempo_gate_does_not_match(self):
        source = [
            note(midi, onset, "clean_electric_guitar", velocity=0.62)
            for onset in (0.0, 0.20, 0.40, 0.60, 0.80, 1.0, 1.20, 1.40)
            for midi in (48, 55, 60, 64)
        ]

        arranged, diagnostics = source_supported_cyclic_harmony(source)

        self.assertFalse(diagnostics["applied"])
        self.assertEqual(diagnostics["reason"], "source-texture-gate-not-satisfied")
        self.assertTrue(arranged)

    def test_source_supported_cycle_preserves_meaningful_piano_source(self):
        source = [
            note(midi, onset, "clean_electric_guitar", duration=0.15, velocity=0.62)
            for onset in (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05, 1.20)
            for midi in (51, 58, 63, 67)
        ]
        expected = compact_harmony(source)

        arranged, diagnostics = source_supported_cyclic_harmony(
            source,
            {
                "_sourcePianoRatio": 0.178,
                "maximumSourcePianoRatio": 0.08,
            },
        )

        self.assertFalse(diagnostics["applied"])
        self.assertEqual(diagnostics["reason"], "piano-source-preservation-gate")
        self.assertEqual(diagnostics["sourcePianoRatio"], 0.178)
        self.assertEqual(diagnostics["maximumSourcePianoRatio"], 0.08)
        self.assertEqual(
            [(item["time"], item["midi"], item["duration"]) for item in arranged],
            [(item["time"], item["midi"], item["duration"]) for item in expected],
        )

    def test_compact_octave_action_replaces_inner_voice_without_adding_density(self):
        source = [
            note(midi, onset, "clean_electric_guitar", duration=0.15, velocity=0.62)
            for onset in (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05, 1.20)
            for midi in (51, 58, 63, 67)
        ]

        arranged, diagnostics = source_supported_cyclic_harmony(
            source,
            {
                "onsetPolicy": "compact",
                "phasePattern": ["compact-octave-root"],
            },
        )
        groups = _group_test_notes(arranged)

        self.assertTrue(diagnostics["applied"])
        self.assertEqual({len(group) for group in groups}, {3})
        self.assertEqual([item["midi"] for item in groups[0]], [51, 63, 67])

    def test_compact_preserve_action_keeps_frozen_harmony_notes_and_holds(self):
        source = [
            note(midi, onset, "clean_electric_guitar", duration=0.17, velocity=0.62)
            for onset in (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05, 1.20)
            for midi in (51, 58, 63, 67)
        ]
        expected = compact_harmony(source)

        arranged, _diagnostics = source_supported_cyclic_harmony(
            source,
            {
                "onsetPolicy": "compact",
                "phasePattern": ["compact-preserve"],
            },
        )

        self.assertEqual(
            [(item["time"], item["midi"], item["duration"]) for item in arranged],
            [(item["time"], item["midi"], item["duration"]) for item in expected],
        )

    def test_compact_cycle_can_advance_on_the_real_pulse_clock(self):
        source = [
            note(midi, onset, "clean_electric_guitar", duration=0.15, velocity=0.62)
            for onset in (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05, 1.20)
            for midi in (51, 58, 63, 67)
        ]

        arranged, diagnostics = source_supported_cyclic_harmony(
            source,
            {
                "onsetPolicy": "compact",
                "phaseClock": "pulse-grid",
                "phasePattern": ["compact-preserve"] * 8,
            },
        )
        groups = _group_test_notes(arranged)

        self.assertEqual(diagnostics["phaseClock"], "pulse-grid")
        self.assertEqual(groups[0][0]["pianistTexturePhase"], 0)
        self.assertEqual(groups[1][0]["pianistTexturePhase"], 2)

    def test_compact_cycle_can_reset_phase_from_the_detected_bass_root(self):
        source = [
            note(midi, onset, "clean_electric_guitar", duration=0.15, velocity=0.62)
            for onset in (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05, 1.20)
            for midi in (48, 52, 55)
        ]
        bass_context = [
            note(36, onset, "electric_bass", duration=0.25, velocity=0.72)
            for onset in (0.0, 0.30, 0.60)
        ] + [
            note(38, onset, "electric_bass", duration=0.25, velocity=0.72)
            for onset in (0.90, 1.20)
        ]

        arranged, diagnostics = source_supported_cyclic_harmony(
            source,
            {
                "onsetPolicy": "compact",
                "phaseClock": "harmonic-root-event-index",
                "phasePattern": ["compact-preserve"] * 4,
                "useBassContextForRoot": True,
                "bassContextRadiusSeconds": 0.16,
                "_bassContextNotes": bass_context,
            },
        )
        groups = _group_test_notes(arranged)

        self.assertEqual(
            [group[0]["pianistTexturePhase"] for group in groups],
            [0, 1, 2, 0, 1],
        )
        self.assertEqual(diagnostics["harmonicRootResets"], 2)
        self.assertEqual(diagnostics["bassContextNotes"], 5)

    def test_compact_cycle_can_choose_texture_from_source_context(self):
        source = [
            note(midi, onset, "clean_electric_guitar", duration=0.15, velocity=0.62)
            for onset in (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05, 1.20)
            for midi in (51, 58, 63, 67)
        ]

        arranged, diagnostics = source_supported_cyclic_harmony(
            source,
            {
                "onsetPolicy": "compact",
                "phasePattern": ["compact-preserve"],
                "contextActionMap": {
                    "quiet:both:change": "compact-inner",
                    "quiet:both:stable": "compact-upper",
                },
                "_melodyOnsets": [],
            },
        )
        groups = _group_test_notes(arranged)

        self.assertEqual([item["midi"] for item in groups[0]], [58])
        self.assertTrue(all(item["midi"] >= 60 for item in groups[1]))
        self.assertEqual(
            {item["pianistTextureAction"] for item in groups[1]},
            {"compact-upper"},
        )
        self.assertGreater(diagnostics["contextCounts"]["quiet:both:stable"], 0)
        self.assertIn(
            "quiet:both:change=compact-inner",
            diagnostics["contextActionCounts"],
        )

    def test_pulse_grid_reconstructs_a_missing_source_attack_and_can_rest(self):
        source = [
            note(midi, onset, "clean_electric_guitar", duration=0.15, velocity=0.62)
            # The detector missed 0.45, but the surrounding 150 ms clock is
            # unambiguous.  The research decoder may reconstruct that timing;
            # the phase pattern still decides whether a pianist strikes it.
            for onset in (0.0, 0.15, 0.30, 0.60, 0.75, 0.90, 1.05, 1.20)
            for midi in (51, 58, 63, 67)
        ]

        arranged, diagnostics = source_supported_cyclic_harmony(
            source,
            {
                "onsetPolicy": "pulse-grid",
                "phasePattern": ["root", "rest"],
                "minimumRunGroups": 8,
                "minimumGuitarShare": 0.7,
                "minimumGridCoverage": 0.7,
            },
        )
        groups = _group_test_notes(arranged)

        self.assertTrue(diagnostics["applied"])
        self.assertAlmostEqual(diagnostics["estimatedPulseSeconds"], 0.15, places=3)
        self.assertEqual(
            [round(float(group[0]["time"]), 2) for group in groups],
            [0.0, 0.30, 0.60, 0.90, 1.20],
        )
        self.assertEqual(diagnostics["generatedGridGestures"], 5)
        self.assertEqual(diagnostics["suppressedGridGestures"], 4)
        self.assertTrue(
            all(item["midi"] % 12 in {51 % 12, 58 % 12, 63 % 12, 67 % 12} for item in arranged)
        )

    def test_pulse_grid_can_share_an_onset_with_existing_melody_or_bass(self):
        source = [
            note(midi, onset, "clean_electric_guitar", duration=0.15, velocity=0.62)
            for onset in (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05)
            for midi in (51, 58, 63, 67)
        ]

        arranged, diagnostics = source_supported_cyclic_harmony(
            source,
            {
                "onsetPolicy": "pulse-grid",
                "phasePattern": ["root"],
                "_preferredOnsets": [0.02, 0.28],
                "snapToRoleOnsetsSeconds": 0.03,
                "minimumGridCoverage": 0.7,
            },
        )

        onset_times = sorted({round(float(item["time"]), 2) for item in arranged})
        self.assertEqual(onset_times[:4], [0.02, 0.15, 0.28, 0.45])
        self.assertEqual(diagnostics["snappedGridGestures"], 2)

    def test_compact_cycle_can_fill_quiet_missing_pulses_without_target_notes(self):
        source = [
            note(midi, onset, "clean_electric_guitar", duration=0.15, velocity=0.62)
            for onset in (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05, 1.20)
            for midi in (51, 58, 63, 67)
        ]

        arranged, diagnostics = source_supported_cyclic_harmony(
            source,
            {
                "onsetPolicy": "compact",
                "phaseClock": "pulse-grid",
                "phasePattern": ["compact-preserve"],
                "supplementalPhasePattern": ["inner"],
                "supplementalMinimumOnsetDistanceSeconds": 0.08,
                "_melodyOnsets": [],
            },
        )

        onset_times = sorted({round(float(item["time"]), 2) for item in arranged})
        self.assertEqual(
            onset_times,
            [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05, 1.20],
        )
        self.assertEqual(diagnostics["supplementalGridGestures"], 4)
        self.assertEqual(diagnostics["sourceSupportedPatternNotes"], len(arranged))

    def test_research_default_pipeline_can_preserve_faster_harmony_attacks(self):
        source = [
            note(midi, onset, "clean_electric_guitar", velocity=0.7)
            # 140 ms is a common broken-chord spacing in the trusted pianist
            # references.  The legacy 300 ms quantizer collapses both attacks
            # into one bin, while a 120 ms research window preserves them.
            for onset in (0.0, 0.14)
            for midi in (48, 52, 55, 58)
        ]

        baseline = compact_harmony(source)
        candidate = compact_harmony(
            source,
            window_seconds=0.12,
            maximum_pitch_classes=2,
        )

        self.assertEqual(len({item["time"] for item in baseline}), 1)
        self.assertEqual(len({item["time"] for item in candidate}), 2)
        self.assertEqual(len(candidate), 4)

        payload = {"title": "Research structure fixture", "notes": source}
        rendered = arrange_payload(
            payload,
            "full",
            style_profile={
                "schema": "polymath-piano-arranger-profile-v1",
                "id": "research-structure-fixture",
                "decoder": {
                    "defaultPipeline": True,
                    "defaultFullMix": {
                        "harmonyWindowSeconds": 0.12,
                        "maximumHarmonyPitchClasses": 2,
                    },
                },
            },
        )
        self.assertEqual(rendered["pianoArrangement"]["maximumHarmonyPitchClasses"], 2)
        self.assertEqual(
            rendered["pianoArrangement"]["defaultFullMixTuning"],
            {"harmonyWindowSeconds": 0.12, "maximumHarmonyPitchClasses": 2},
        )
        self.assertEqual(
            rendered["pianoArrangement"]["learnedProfileBypassReason"],
            "default-pipeline-profile-applied",
        )

    def test_performance_only_profile_keeps_default_note_selection_and_timing(self):
        payload_notes = []
        for index in range(120):
            onset = index * 0.12
            payload_notes.extend(
                [
                    note(43 + index % 5, onset, "electric_bass", velocity=0.42),
                    note(
                        58 + index % 12,
                        onset + 0.01,
                        "clean_electric_guitar",
                        velocity=0.58,
                    ),
                ]
            )
            if index % 3 == 0:
                payload_notes.append(note(64 + index % 7, onset, "voice", velocity=0.82))
        payload = {"title": "Performance-only route fixture", "notes": payload_notes}
        baseline = arrange_payload(payload, "full")
        profile = {
            "schema": "polymath-piano-arranger-profile-v1",
            "id": "performance-only-test-v001",
            "decoder": {
                "performanceOnly": True,
                "gestureDynamics": {
                    "enabled": True,
                    "preserveInputGestureVelocity": True,
                    "minimumVelocity": 0.05,
                    "maximumVelocity": 1.0,
                    "pairedCalibration": {
                        "enabled": True,
                        "id": "performance-only-bias-test",
                        "featureNames": ["bias"],
                        "weights": [0.05],
                        "means": [0.0],
                        "scales": [1.0],
                        "blend": 1.0,
                        "maximumCorrection": 0.05,
                    },
                },
            },
        }

        candidate = arrange_payload(payload, "full", style_profile=profile)

        baseline_score = [
            (item["time"], item["midi"], item["duration"])
            for item in baseline["notes"]
        ]
        candidate_score = [
            (item["time"], item["midi"], item["duration"])
            for item in candidate["notes"]
        ]
        self.assertEqual(candidate_score, baseline_score)
        self.assertTrue(candidate["pianoArrangement"]["performanceOnlyProfileApplied"])
        self.assertEqual(
            candidate["pianoArrangement"]["learnedProfileBypassReason"],
            "selection-frozen-performance-only-profile",
        )
        self.assertEqual(
            candidate["pianoArrangement"]["profile"],
            baseline["pianoArrangement"]["profile"],
        )
        self.assertNotEqual(
            [item["velocity"] for item in candidate["notes"]],
            [item["velocity"] for item in baseline["notes"]],
        )

    def test_default_pipeline_profile_preserves_meaningful_mixed_piano_source(self):
        payload_notes = []
        for index in range(80):
            onset = index * 0.15
            payload_notes.extend(
                [
                    note(43 + index % 5, onset, "electric_bass", velocity=0.42),
                    note(58 + index % 12, onset + 0.01, "clean_electric_guitar", velocity=0.58),
                ]
            )
            if index % 3 == 0:
                payload_notes.append(
                    note(64 + index % 7, onset, "acoustic_piano", velocity=0.74)
                )
        payload = {"title": "Mixed piano preservation fixture", "notes": payload_notes}
        baseline = arrange_payload(payload, "full")
        profile = {
            "schema": "polymath-piano-arranger-profile-v1",
            "id": "mixed-piano-preservation-test-v001",
            "decoder": {
                "defaultPipeline": True,
                "maximumSourcePianoRatioForProfile": 0.08,
                "defaultFullMix": {
                    "sourceSupportedCyclicHarmony": {"enabled": True},
                    "adaptiveMelodyRegisterSeparation": {
                        "enabled": True,
                        "octaveShiftSemitones": 12,
                    },
                },
                "gestureDynamics": {
                    "enabled": True,
                    "minimumVelocity": 0.9,
                    "maximumVelocity": 0.95,
                },
            },
        }

        candidate = arrange_payload(payload, "full", style_profile=profile)

        self.assertEqual(candidate["notes"], baseline["notes"])
        diagnostics = candidate["pianoArrangement"]
        self.assertFalse(diagnostics["defaultPipelineProfileApplied"])
        self.assertEqual(
            diagnostics["learnedProfileBypassReason"],
            "meaningful-piano-source-preserved",
        )
        self.assertTrue(
            diagnostics["defaultPipelinePianoPreservationGate"]["applied"]
        )

    def test_learned_profile_can_bound_sparse_accompaniment_key_holds(self):
        profile = copy.deepcopy(LEARNED_PROFILE)
        profile.setdefault("decoder", {})["physicalPerformance"] = {
            "maximumLegatoBridgeSeconds": {
                "melody": 0.9,
                "bass": 0.7,
                "harmony": 0.5,
            },
            "maximumPhysicalHoldSeconds": {
                "melody": 1.35,
                "bass": 1.2,
                "harmony": 1.0,
            },
        }
        limits = learned_physical_performance_limits(profile)
        self.assertEqual(limits["maximumLegatoBridgeSeconds"]["harmony"], 0.5)

        payload = {
            "title": "Physical hold policy fixture",
            "notes": [
                note(48 + index % 12, index * 0.18, "acoustic_guitar")
                for index in range(100)
            ]
            + [
                note(68 + index % 5, index * 0.54, "voice")
                for index in range(34)
            ],
        }
        result = arrange_payload(payload, "full", style_profile=profile)

        self.assertEqual(
            result["pianoArrangement"]["physicalPerformanceRoleLimits"], limits
        )
        for item in result["notes"]:
            role = item["arrangementRole"]
            self.assertEqual(
                item["maximumLegatoBridgeSeconds"],
                limits["maximumLegatoBridgeSeconds"][role],
            )
            self.assertEqual(
                item["maximumPhysicalHoldSeconds"],
                limits["maximumPhysicalHoldSeconds"][role],
            )

    def test_learned_profile_is_used_only_for_a_full_mix(self):
        notes = []
        for index in range(120):
            onset = index * 0.12
            notes.extend(
                [
                    note(36, onset, "drums", duration=0.05),
                    note(43 + index % 5, onset, "electric_bass"),
                    note(58 + index % 12, onset + 0.01, "clean_electric_guitar"),
                ]
            )
            if index % 3 == 0:
                notes.append(note(64 + index % 7, onset, "voice"))

        result = arrange_payload(
            {"title": "Learned route fixture", "notes": notes},
            "full",
            style_profile=LEARNED_PROFILE,
        )

        self.assertEqual(
            result["pianoArrangement"]["learnedProfileId"],
            "pianella-supervised-v006",
        )
        self.assertEqual(
            result["pianoArrangement"]["routingContract"],
            "instrument-aware-transcription-then-piano-only-arrangement",
        )
        self.assertTrue(result["vocalMelodyIncluded"])
        self.assertTrue(
            all(item["instrument"] == "acoustic_piano" for item in result["notes"])
        )

    def test_learned_instrumental_route_drops_voice_before_piano_rendering(self):
        payload = {
            "title": "Learned instrumental fixture",
            "notes": [
                note(60 + index % 5, index * 0.12, "voice")
                if index % 2
                else note(48 + index % 12, index * 0.12, "acoustic_guitar")
                for index in range(100)
            ],
        }

        result = arrange_payload(payload, "instrumental", style_profile=LEARNED_PROFILE)

        self.assertFalse(result["vocalMelodyIncluded"])
        self.assertTrue(
            all(item.get("sourceInstrument") != "voice" for item in result["notes"])
        )

    def test_mixed_source_does_not_bypass_the_learned_arranger_because_of_its_piano_ratio(self):
        notes = [
            note(48 + index % 24, index * 0.14, "acoustic_piano")
            for index in range(80)
        ]
        notes.extend(
            note(72 + index % 5, index * 0.56, "flutes")
            for index in range(20)
        )

        result = arrange_payload(
            {"title": "Mostly piano fixture", "notes": notes},
            "full",
            style_profile=LEARNED_PROFILE,
        )

        self.assertEqual(
            result["pianoArrangement"]["learnedProfileId"],
            "pianella-supervised-v006",
        )
        self.assertEqual(result["pianoArrangement"]["version"], 6)
        self.assertTrue(result["performance"]["melodyForwardDynamics"])
        self.assertTrue(result["pedals"])
        self.assertTrue(all("audioDuration" in item for item in result["notes"]))
        self.assertTrue(
            all(item["articulation"] == "legato" for item in result["notes"])
        )
        self.assertTrue(
            all("maximumLegatoBridgeSeconds" in item for item in result["notes"])
        )

    def test_clean_solo_piano_still_bypasses_the_learned_arranger(self):
        notes = [
            note(48 + index % 24, index * 0.18, "acoustic_piano", duration=0.42)
            for index in range(100)
        ]

        result = arrange_payload(
            {"title": "Clean solo piano", "notes": notes},
            "full",
            style_profile=LEARNED_PROFILE,
        )

        self.assertEqual(result["pianoArrangement"]["profile"], "acoustic-piano-preserve")
        self.assertEqual(
            result["pianoArrangement"]["requestedLearnedProfileId"],
            "pianella-supervised-v006",
        )
        self.assertEqual(
            result["pianoArrangement"]["learnedProfileBypassReason"],
            "genuine-solo-piano-preserved",
        )

    def test_preserves_clean_acoustic_piano_inside_88_key_range(self):
        notes = [
            note(48 + index % 24, index * 0.125, "acoustic_piano")
            for index in range(120)
        ]
        payload = {"title": "Acoustic fixture", "notes": notes}

        result = arrange_payload(payload, "full")

        self.assertEqual(
            result["pianoArrangement"]["profile"],
            "acoustic-piano-preserve",
        )
        self.assertTrue(
            result["pianoArrangement"]["detectedAcousticPianoPerformance"]
        )
        self.assertEqual(len(result["notes"]), len(notes))
        self.assertTrue(
            all(
                PIANO_MIN_MIDI <= arranged["midi"] <= PIANO_MAX_MIDI
                for arranged in result["notes"]
            )
        )

    def test_preserves_high_density_pure_acoustic_piano(self):
        # A virtuoso performance can legitimately exceed the full-mix
        # arranger's 12-note/second budget. Instrument purity, bounded onset
        # clusters, and low cleanup pressure distinguish it from a mislabeled
        # separator output.
        notes = [
            note(48 + index % 24, index * 0.055, "acoustic_piano", duration=0.18)
            for index in range(180)
        ]

        result = arrange_payload({"title": "Virtuoso piano", "notes": notes}, "full")

        self.assertEqual(
            result["pianoArrangement"]["profile"],
            "acoustic-piano-preserve",
        )
        self.assertTrue(
            result["pianoArrangement"]["pureAcousticPianoDensityOverride"]
        )
        self.assertGreater(
            result["pianoArrangement"]["densityLimitNotesPerSecond"], 12
        )
        self.assertEqual(len(result["notes"]), len(notes))
        self.assertTrue(
            all(
                float(item["audioDuration"]) >= 0.16
                for item in result["notes"]
            )
        )
        self.assertGreater(
            result["pianoPerformance"]["independentPianoHoldsPreserved"],
            0,
        )

        frozen_baseline = arrange_payload(
            {"title": "Virtuoso piano", "notes": notes},
            "full",
            allow_pure_piano_density_override=False,
        )
        self.assertEqual(
            frozen_baseline["pianoArrangement"]["profile"],
            "full-mix-piano-reduction",
        )
        self.assertFalse(
            frozen_baseline["pianoArrangement"][
                "pureAcousticPianoDensityOverride"
            ]
        )

    def test_mislabeled_full_mix_uses_cleanup_pressure_instead_of_trusting_piano_tag(self):
        notes = [
            note(38 + index % 34, index * 0.12, "acoustic_piano", duration=1.1)
            for index in range(180)
        ]
        payload = {
            "title": "Mislabeled full mix",
            "notes": notes,
            "transcriptionCleanup": {
                "inputNotes": 230,
                "removedDuplicateNotes": 28,
                "shortenedSameKeyOverlaps": 48,
            },
        }

        result = arrange_payload(payload, "instrumental")

        self.assertEqual(
            result["pianoArrangement"]["profile"],
            "full-mix-piano-reduction",
        )
        self.assertFalse(
            result["pianoArrangement"]["detectedAcousticPianoPerformance"]
        )
        self.assertGreater(
            result["pianoArrangement"]["cleanupArtifactPressure"],
            result["pianoArrangement"]["maximumDirectCleanupPressure"],
        )
        self.assertTrue(
            any(item["arrangementRole"] == "melody" for item in result["notes"])
        )

    def test_piano_only_source_survives_acoustic_electric_label_drift(self):
        notes = [
            note(
                45 + index % 36,
                index * 0.09,
                "acoustic_piano" if index % 2 == 0 else "electric_piano",
                duration=0.24,
            )
            for index in range(160)
        ]
        payload = {
            "title": "One piano with classifier label drift",
            "notes": notes,
            "transcriptionCleanup": {
                "inputNotes": 200,
                "removedDuplicateNotes": 18,
                "shortenedSameKeyOverlaps": 22,
            },
        }

        result = arrange_payload(payload, "full")

        self.assertEqual(
            result["pianoArrangement"]["profile"],
            "acoustic-piano-preserve",
        )
        self.assertTrue(result["pianoArrangement"]["purePianoSource"])
        self.assertEqual(
            result["pianoArrangement"]["maximumDirectCleanupPressure"],
            0.25,
        )

    def test_preserved_piano_stabilizes_only_implausibly_short_score_holds(self):
        notes = [
            note(48, 0.0, "acoustic_piano", duration=0.08),
            note(72, 0.2, "acoustic_piano", duration=0.08),
            note(47, 0.4, "acoustic_piano", duration=0.15),
            note(76, 0.6, "acoustic_piano", duration=0.30),
        ] + [
            note(60 + index % 12, 1.0 + index * 0.2, "acoustic_piano")
            for index in range(24)
        ]

        result = arrange_payload({"title": "Short hold fixture", "notes": notes})
        by_onset = {round(float(item["time"]), 1): item for item in result["notes"]}

        self.assertAlmostEqual(by_onset[0.0]["duration"], 0.1595)
        self.assertAlmostEqual(by_onset[0.2]["duration"], 0.145)
        self.assertAlmostEqual(by_onset[0.4]["duration"], 0.2175)
        self.assertAlmostEqual(by_onset[0.6]["duration"], 0.30)
        diagnostics = result["pianoArrangement"]["directPianoHoldStabilization"]
        self.assertEqual(diagnostics["profile"], "direct-piano-short-hold-stabilizer-v2")
        self.assertEqual(diagnostics["minimumHoldAdjustedNotes"], 2)
        self.assertEqual(diagnostics["shortBassAdjustedNotes"], 1)
        self.assertEqual(diagnostics["calibratedShortHoldNotes"], 3)
        self.assertEqual(diagnostics["shortHoldScale"], 1.45)
        self.assertEqual(diagnostics["shortHoldScaleMaximumSeconds"], 0.25)

    def test_full_mix_removes_drums_and_prioritizes_vocal_melody(self):
        notes = []
        for index in range(120):
            time = index * 0.08
            notes.extend(
                [
                    note(36, time, "drums", duration=0.05),
                    note(45 + index % 5, time, "electric_bass"),
                    note(55 + index % 12, time, "clean_electric_guitar"),
                    note(60 + index % 8, time + 0.01, "clean_electric_guitar"),
                ]
            )
            if index % 6 == 0:
                notes.append(note(67 + index % 5, time, "voice", velocity=0.82))
        notes.extend([note(10, 0, "voice"), note(120, 1, "voice")])

        result = arrange_payload({"title": "Full mix", "notes": notes}, "full")

        self.assertEqual(
            result["pianoArrangement"]["profile"],
            "full-mix-piano-reduction",
        )
        self.assertGreater(result["pianoArrangement"]["removedPercussionNotes"], 0)
        self.assertGreater(result["pianoArrangement"]["vocalMelodyNotes"], 0)
        self.assertTrue(result["vocalMelodyIncluded"])
        self.assertTrue(
            all(
                arranged.get("sourceInstrument") != "drums"
                for arranged in result["notes"]
            )
        )
        self.assertTrue(
            all(
                PIANO_MIN_MIDI <= arranged["midi"] <= PIANO_MAX_MIDI
                for arranged in result["notes"]
            )
        )
        self.assertLessEqual(
            result["pianoArrangement"]["outputNotesPerSecond"],
            MAX_ARRANGED_NOTES_PER_SECOND,
        )
        self.assertLessEqual(
            result["pianoArrangement"]["outputMaximumOnsetCluster"],
            6,
        )

    def test_instrumental_mode_excludes_voice(self):
        payload = {
            "title": "Instrumental fixture",
            "notes": [
                note(67, 0, "voice"),
                note(48, 0, "electric_bass"),
                note(60, 0, "acoustic_guitar"),
                note(64, 0.02, "acoustic_guitar"),
                note(67, 0.04, "acoustic_guitar"),
            ],
        }

        result = arrange_payload(payload, "instrumental")

        self.assertFalse(result["vocalMelodyIncluded"])
        self.assertEqual(result["pianoArrangement"]["vocalMelodyNotes"], 0)
        self.assertTrue(
            all(
                arranged.get("sourceInstrument") != "voice"
                for arranged in result["notes"]
            )
        )

    def test_suppresses_same_key_machine_gun_retriggers(self):
        notes = [
            note(60, 1.0, "clean_electric_guitar", duration=0.04),
            note(60, 1.04, "clean_electric_guitar", duration=0.04),
            note(60, 1.07, "clean_electric_guitar", duration=0.04),
            note(48, 1.0, "electric_bass"),
        ]

        result = arrange_payload({"title": "Retrigger fixture", "notes": notes}, "full")
        c_notes = [
            arranged
            for arranged in result["notes"]
            if arranged["midi"] == 60
            and arranged.get("sourceInstrument") == "clean_electric_guitar"
        ]

        self.assertEqual([arranged["time"] for arranged in c_notes], [1.0, 1.07])
        self.assertGreater(
            result["pianoArrangement"]["removedRapidRetriggers"],
            0,
        )

    def test_full_mix_merges_cross_role_same_key_collisions(self):
        notes = []
        for index in range(36):
            onset = index * 0.24
            notes.extend(
                [
                    note(48, onset, "electric_bass", duration=0.5),
                    note(60, onset, "voice", duration=0.35),
                    note(60, onset + 0.12, "clean_electric_guitar", duration=0.5),
                    note(64, onset + 0.12, "clean_electric_guitar", duration=0.5),
                ]
            )

        result = arrange_payload({"title": "Role collision", "notes": notes}, "full")
        c4_onsets = sorted(
            item["time"] for item in result["notes"] if item["midi"] == 60
        )

        self.assertTrue(
            all(
                current - previous >= 0.18
                for previous, current in zip(c4_onsets, c4_onsets[1:])
            )
        )


if __name__ == "__main__":
    unittest.main()
