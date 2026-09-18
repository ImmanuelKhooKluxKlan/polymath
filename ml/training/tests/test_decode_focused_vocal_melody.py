import copy
import unittest

from ml.training.decode_focused_vocal_melody import decode_focused_vocal_melody


class FocusedVocalMelodyDecoderTests(unittest.TestCase):
    def test_selects_one_continuous_pitch_and_joins_model_frames(self):
        focused = [
            {"midi": 60, "time": 1.00, "duration": 0.18, "instrument": "voice"},
            {"midi": 64, "time": 1.00, "duration": 0.18, "instrument": "voice"},
            {"midi": 67, "time": 1.00, "duration": 0.18, "instrument": "voice"},
            {"midi": 60, "time": 1.18, "duration": 0.18, "instrument": "voice"},
            {"midi": 64, "time": 1.18, "duration": 0.18, "instrument": "voice"},
            {"midi": 67, "time": 1.18, "duration": 0.18, "instrument": "voice"},
        ]
        primary = [
            {"midi": 64, "time": 1.01, "duration": 0.4, "instrument": "voice"},
            {"midi": 48, "time": 1.00, "duration": 0.4, "instrument": "bass"},
        ]
        original = copy.deepcopy(focused)
        decoded, diagnostics = decode_focused_vocal_melody(focused, primary)

        self.assertEqual(focused, original)
        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoded[0]["midi"], 64)
        self.assertAlmostEqual(decoded[0]["duration"], 0.36)
        self.assertTrue(decoded[0]["focusedMelodyDecoded"])
        self.assertEqual(diagnostics["inputFrames"], 2)
        self.assertEqual(diagnostics["samePitchFragmentsJoined"], 1)

    def test_preserves_a_real_rest_and_same_pitch_retrigger(self):
        focused = [
            {"midi": 60, "time": 1.00, "duration": 0.18, "instrument": "voice"},
            {"midi": 60, "time": 1.18, "duration": 0.18, "instrument": "voice"},
            {"midi": 60, "time": 1.80, "duration": 0.18, "instrument": "voice"},
        ]
        decoded, diagnostics = decode_focused_vocal_melody(focused, [])

        self.assertEqual(len(decoded), 2)
        self.assertEqual([note["time"] for note in decoded], [1.0, 1.8])
        self.assertEqual(diagnostics["phrases"], 1)

    def test_rejects_an_unanchored_chord_like_frame(self):
        focused = [
            {"midi": midi, "time": 1.0, "duration": 0.2, "instrument": "voice"}
            for midi in (48, 52, 55, 60, 64)
        ] + [
            {"midi": 62, "time": 2.0, "duration": 0.2, "instrument": "voice"}
        ]
        decoded, diagnostics = decode_focused_vocal_melody(focused, [])

        self.assertEqual([(note["midi"], note["time"]) for note in decoded], [(62, 2.0)])
        self.assertEqual(diagnostics["rejectedChordLikeFrames"], 1)

    def test_keeps_a_polyphonic_frame_when_the_broad_voice_anchors_it(self):
        focused = [
            {"midi": midi, "time": 1.0, "duration": 0.2, "instrument": "voice"}
            for midi in (48, 52, 55, 60, 64)
        ]
        primary = [
            {"midi": 60, "time": 1.02, "duration": 0.5, "instrument": "voice"}
        ]
        decoded, diagnostics = decode_focused_vocal_melody(focused, primary)

        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoded[0]["midi"], 60)
        self.assertEqual(diagnostics["anchoredFrames"], 1)

    def test_density_gate_leaves_an_ordinary_pass_bit_for_bit_unchanged(self):
        focused = [
            {"midi": 60, "time": 1.0, "duration": 0.2, "instrument": "voice"},
            {"midi": 62, "time": 3.0, "duration": 0.2, "instrument": "voice"},
        ]
        decoded, diagnostics = decode_focused_vocal_melody(
            focused,
            [],
            config={"minimum_input_notes_per_second": 4.0},
        )

        self.assertEqual(decoded, focused)
        self.assertFalse(diagnostics["applied"])
        self.assertEqual(
            diagnostics["skipReason"],
            "input-density-below-fragmentation-threshold",
        )


if __name__ == "__main__":
    unittest.main()
