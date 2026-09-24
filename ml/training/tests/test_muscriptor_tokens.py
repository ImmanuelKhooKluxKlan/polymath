import math
import unittest

from ml.training.muscriptor_tokens import (
    EOS_ID,
    INITIAL_TOKEN_ID,
    MODEL_CARDINALITY,
    PITCH_BASE,
    PROGRAM_BASE,
    SHIFT_BASE,
    TIE_ID,
    VELOCITY_BASE,
    VOCAB_SIZE,
    encode_piano_clip,
    encode_instrument_clip,
    teacher_forcing_pair,
)
from ml.training.train_muscriptor_piano import (
    TrainingError,
    decoded_selection_rank,
    deterministic_seed,
    target_token_weights,
)


class MuScriptorTokenTests(unittest.TestCase):
    def test_decoded_f1_outranks_lower_teacher_forced_loss(self):
        musically_better = {
            "100ms": {"microF1": 0.901, "recall": 0.89},
            "250ms": {"microF1": 0.92, "recall": 0.91},
        }
        lower_loss_but_worse_notes = {
            "100ms": {"microF1": 0.899, "recall": 0.90},
            "250ms": {"microF1": 0.93, "recall": 0.92},
        }
        self.assertGreater(
            decoded_selection_rank(musically_better, 1.20),
            decoded_selection_rank(lower_loss_but_worse_notes, 1.10),
        )

    def test_decoded_recall_breaks_an_exact_f1_tie(self):
        higher_recall = {
            "100ms": {"microF1": 0.90, "recall": 0.89},
            "250ms": {"microF1": 0.91, "recall": 0.90},
        }
        lower_recall = {
            "100ms": {"microF1": 0.90, "recall": 0.88},
            "250ms": {"microF1": 0.95, "recall": 0.94},
        }
        self.assertGreater(
            decoded_selection_rank(higher_recall, 1.20),
            decoded_selection_rank(lower_recall, 1.10),
        )

    def test_decoded_selection_rejects_non_finite_metrics(self):
        invalid = {
            "100ms": {"microF1": math.nan, "recall": 0.89},
            "250ms": {"microF1": 0.92, "recall": 0.91},
        }
        with self.assertRaisesRegex(TrainingError, "must be finite"):
            decoded_selection_rank(invalid, 1.20)

    def test_training_seed_is_stable_and_label_specific(self):
        first = deterministic_seed("phase94-v002")
        self.assertEqual(first, deterministic_seed("phase94-v002"))
        self.assertNotEqual(first, deterministic_seed("phase94-v003"))
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 2**32)

    def test_simple_note_matches_public_mt3_event_layout(self):
        tokens = encode_piano_clip([
            {"midi": 60, "time": 0.25, "duration": 0.5},
        ])
        self.assertEqual(tokens, [
            TIE_ID,
            SHIFT_BASE + 25,
            PROGRAM_BASE,
            VELOCITY_BASE + 1,
            PITCH_BASE + 60,
            SHIFT_BASE + 75,
            VELOCITY_BASE,
            PITCH_BASE + 60,
            EOS_ID,
        ])

    def test_boundary_sustain_uses_tie_and_omits_right_edge_note_off(self):
        tokens = encode_piano_clip([
            {
                "midi": 48,
                "time": 0,
                "duration": 2,
                "continuedFromPreviousClip": True,
            },
            {
                "midi": 64,
                "time": 4.75,
                "duration": 0.25,
                "continuesIntoNextClip": True,
            },
        ])
        self.assertEqual(tokens[:3], [PROGRAM_BASE, PITCH_BASE + 48, TIE_ID])
        self.assertIn(PITCH_BASE + 64, tokens)
        self.assertEqual(tokens[-1], EOS_ID)
        self.assertTrue(all(0 <= token < MODEL_CARDINALITY for token in tokens))

    def test_teacher_forcing_shifts_targets_by_one(self):
        targets = [TIE_ID, EOS_ID]
        inputs, labels = teacher_forcing_pair(targets)
        self.assertEqual(inputs, [INITIAL_TOKEN_ID, TIE_ID])
        self.assertEqual(labels, targets)
        self.assertEqual(VOCAB_SIZE, 1393)
        self.assertEqual(MODEL_CARDINALITY, 1395)

    def test_teacher_forcing_accepts_checkpoint_specific_initial_token(self):
        inputs, labels = teacher_forcing_pair([TIE_ID, EOS_ID], initial_token_id=1393)
        self.assertEqual(inputs, [1393, TIE_ID])
        self.assertEqual(labels, [TIE_ID, EOS_ID])

    def test_individual_guitar_uses_its_representative_program(self):
        tokens = encode_instrument_clip([
            {"midi": 52, "time": 0.1, "duration": 0.3, "instrument": "acoustic_guitar"},
        ], instrument="acoustic_guitar")
        self.assertIn(PROGRAM_BASE + 24, tokens)
        self.assertNotIn(PROGRAM_BASE, tokens)

    def test_overlapping_same_key_is_trimmed_at_next_strike(self):
        tokens = encode_piano_clip([
            {"midi": 60, "time": 0.1, "duration": 1.0},
            {"midi": 60, "time": 0.5, "duration": 0.4},
        ])
        # At 0.50 seconds the first strike must turn off immediately before
        # the second turns on; there must be no stale 1.10-second note-off.
        shift_50 = tokens.index(SHIFT_BASE + 50)
        self.assertEqual(tokens[shift_50 + 1:shift_50 + 5], [
            VELOCITY_BASE,
            PITCH_BASE + 60,
            VELOCITY_BASE + 1,
            PITCH_BASE + 60,
        ])
        self.assertNotIn(SHIFT_BASE + 110, tokens)

    def test_loss_weights_emphasize_timing_note_off_and_eos(self):
        tokens = [
            TIE_ID,
            SHIFT_BASE + 10,
            VELOCITY_BASE + 1,
            PITCH_BASE + 60,
            SHIFT_BASE + 50,
            VELOCITY_BASE,
            PITCH_BASE + 60,
            EOS_ID,
        ]
        weights = target_token_weights(tokens)
        self.assertAlmostEqual(sum(weights) / len(weights), 1.0)
        self.assertGreater(weights[1], weights[3])
        self.assertGreater(weights[5], weights[3])
        self.assertGreater(weights[6], weights[3])
        self.assertGreater(weights[7], weights[3])

    def test_loss_weights_can_emphasize_note_on_velocity_and_pitch(self):
        tokens = [
            SHIFT_BASE + 10,
            VELOCITY_BASE + 1,
            PITCH_BASE + 60,
            VELOCITY_BASE,
            PITCH_BASE + 60,
            EOS_ID,
        ]
        weights = target_token_weights(tokens, note_on_weight=1.35)
        self.assertAlmostEqual(sum(weights) / len(weights), 1.0)
        self.assertAlmostEqual(weights[1], weights[2])
        self.assertGreater(weights[1], weights[3])
        self.assertGreater(weights[2], weights[4])


if __name__ == "__main__":
    unittest.main()
