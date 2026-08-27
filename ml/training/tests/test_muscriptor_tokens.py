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
    teacher_forcing_pair,
)


class MuScriptorTokenTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

