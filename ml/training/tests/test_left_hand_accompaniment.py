import unittest

import numpy as np

from ml.training.analyze_left_hand_accompaniment import (
    accompaniment_summary,
    candidate_left_notes,
    onset_groups,
    reference_left_notes,
)
from ml.training.fit_chord_completion_ranker import match_onset_groups
from ml.training.fit_left_hand_accompaniment import (
    build_pairwise_ranking_dataset,
    nearest_target_onset,
    notes_in_ranges,
    target_onset_groups,
)
from ml.training.fit_onset_gesture_ranker import (
    map_time as map_onset_gesture_time,
    one_to_one_positive_keys,
)
from server.piano_arranger_adapter import (
    ONSET_GESTURE_SEQUENCE_FEATURE_NAMES,
    onset_gesture_selection_feature_rows,
    onset_gesture_sequence_feature_rows,
)


class LeftHandAccompanimentAnalysisTests(unittest.TestCase):
    def test_summary_measures_chords_holds_and_fast_retriggers(self):
        notes = [
            {"midi": 48, "time": 0.00, "duration": 0.10},
            {"midi": 52, "time": 0.02, "duration": 0.20},
            {"midi": 48, "time": 0.10, "duration": 0.30},
        ]

        summary = accompaniment_summary(notes)

        self.assertEqual(summary["notes"], 3)
        self.assertEqual(summary["onsets"], 2)
        self.assertEqual(summary["onsetSizeHistogram"], {"1": 1, "2": 1})
        self.assertEqual(summary["fastSamePitchRetriggers65To180ms"], 1)
        self.assertAlmostEqual(summary["durationSeconds"]["median"], 0.2)

    def test_candidate_filter_excludes_melody_voice_and_untrusted_time(self):
        payload = {
            "notes": [
                {"midi": 48, "time": 1.0, "duration": 0.2, "arrangementRole": "bass"},
                {"midi": 50, "time": 1.1, "duration": 0.2, "arrangementRole": "melody"},
                {"midi": 52, "time": 1.2, "duration": 0.2, "sourceInstrument": "voice"},
                {"midi": 54, "time": 4.0, "duration": 0.2, "arrangementRole": "bass"},
                {"midi": 74, "time": 1.3, "duration": 0.2, "arrangementRole": "bass"},
            ]
        }

        selected = candidate_left_notes(payload, [(0.0, 2.0)], hand_split=72)

        self.assertEqual([note["midi"] for note in selected], [48])

    def test_reference_filter_transposes_only_training_eligible_left_hand(self):
        payload = {
            "notes": [
                {"midi": 47, "time": 0.0, "duration": 0.2, "trainingEligible": True},
                {"midi": 58, "time": 0.1, "duration": 0.2, "trainingEligible": False},
                {"midi": 60, "time": 0.2, "duration": 0.2, "trainingEligible": True},
            ]
        }

        selected = reference_left_notes(payload, reference_split=60, transpose=12)

        self.assertEqual([note["midi"] for note in selected], [59])


class LeftHandTrainingHelperTests(unittest.TestCase):
    def test_decoder_statistics_cannot_read_notes_outside_trusted_ranges(self):
        notes = [
            {"midi": 48, "time": 1.0, "duration": 0.4},
            {"midi": 50, "time": 5.0, "duration": 0.4},
            {"midi": 52, "time": 12.0, "duration": 9.0},
            {"midi": 53, "time": "bad", "duration": 1.0},
        ]

        selected = notes_in_ranges(notes, [(0.0, 2.0), (4.0, 6.0)])

        self.assertEqual([note["midi"] for note in selected], [48, 50])

    def test_onset_gesture_mapping_and_labels_are_monotonic_one_to_one(self):
        anchors = [
            {"referenceTime": 0.0, "observedTime": 2.0},
            {"referenceTime": 10.0, "observedTime": 13.0},
        ]
        self.assertAlmostEqual(map_onset_gesture_time(5.0, anchors), 7.5)

        positives = one_to_one_positive_keys(
            [1000, 1040, 2000], [1.02, 2.01], tolerance=0.05
        )
        self.assertEqual(positives, {1000, 2000})

    def test_onset_gesture_features_are_shared_and_transposition_invariant(self):
        notes = [
            {"midi": 40, "time": 1.0, "duration": 0.4, "velocity": 0.7, "instrument": "electric_bass"},
            {"midi": 47, "time": 1.0, "duration": 0.3, "velocity": 0.6, "instrument": "clean_electric_guitar"},
            {"midi": 52, "time": 1.5, "duration": 0.2, "velocity": 0.5, "instrument": "clean_electric_guitar"},
        ]
        shifted = [{**note, "midi": note["midi"] + 5} for note in notes]

        rows = onset_gesture_selection_feature_rows(notes)
        shifted_rows = onset_gesture_selection_feature_rows(shifted)

        self.assertEqual(rows[0], rows[1])
        self.assertNotEqual(rows[0], rows[2])
        np.testing.assert_allclose(rows, shifted_rows)

    def test_sequence_features_are_shared_transposition_invariant_and_phase_aware(self):
        notes = [
            {
                "midi": midi,
                "time": time,
                "duration": 0.22,
                "velocity": 0.62,
                "instrument": "clean_electric_guitar",
            }
            for time in (1.0, 1.2, 1.4, 1.6, 2.5)
            for midi in (48, 55, 60)
        ]
        shifted = [{**note, "midi": note["midi"] + 5} for note in notes]

        rows = onset_gesture_sequence_feature_rows(notes)
        shifted_rows = onset_gesture_sequence_feature_rows(shifted)
        phase_zero = ONSET_GESTURE_SEQUENCE_FEATURE_NAMES.index("harmonic_run_mod4_0")
        phase_one = ONSET_GESTURE_SEQUENCE_FEATURE_NAMES.index("harmonic_run_mod4_1")

        self.assertEqual(rows[0], rows[1])
        self.assertEqual(rows[0], rows[2])
        self.assertEqual(rows[0][phase_zero], 1.0)
        self.assertEqual(rows[3][phase_one], 1.0)
        self.assertEqual(rows[12][phase_zero], 1.0)
        np.testing.assert_allclose(rows, shifted_rows)

    def test_target_onsets_group_nearby_notes_and_find_nearest(self):
        groups = target_onset_groups(
            [
                {"midi": 48, "time": 1.00},
                {"midi": 52, "time": 1.02},
                {"midi": 55, "time": 1.20},
            ]
        )

        self.assertEqual([group["size"] for group in groups], [2, 1])
        nearest, distance = nearest_target_onset(
            groups, [group["time"] for group in groups], 1.18
        )
        self.assertEqual(nearest["pitches"], {55})
        self.assertAlmostEqual(distance, 0.02)

    def test_pairwise_dataset_is_symmetric_and_never_crosses_split(self):
        features = np.asarray([[2.0, 0.0], [0.0, 0.0], [3.0, 1.0], [1.0, 1.0]])
        labels = np.asarray([1.0, 0.0, 1.0, 0.0])
        weights = np.asarray([2.0, 1.0, 3.0, 1.0])
        training = np.asarray([True, True, False, False])
        times = np.asarray([0.00, 0.01, 1.00, 1.01])

        rows, pair_labels, pair_weights, pair_training = build_pairwise_ranking_dataset(
            features,
            labels,
            weights,
            training,
            times,
            maximum_negatives_per_positive=1,
        )

        self.assertEqual(rows.shape, (4, 2))
        np.testing.assert_allclose(rows[0], -rows[1])
        np.testing.assert_allclose(rows[2], -rows[3])
        np.testing.assert_array_equal(pair_labels, [1.0, 0.0, 1.0, 0.0])
        np.testing.assert_array_equal(pair_weights, [2.0, 2.0, 3.0, 3.0])
        np.testing.assert_array_equal(pair_training, [True, True, False, False])

    def test_onset_matching_is_one_to_one(self):
        reference = onset_groups(
            [
                {"midi": 48, "time": 1.00},
                {"midi": 50, "time": 1.20},
            ]
        )
        candidate = onset_groups(
            [
                {"midi": 47, "time": 1.03},
                {"midi": 49, "time": 1.19},
            ]
        )

        matches = match_onset_groups(reference, candidate, tolerance=0.05)

        self.assertEqual(len(matches), 2)
        self.assertEqual([group[0]["midi"] for group, _ in matches], [47, 49])
        self.assertEqual([group[0]["midi"] for _, group in matches], [48, 50])


if __name__ == "__main__":
    unittest.main()
