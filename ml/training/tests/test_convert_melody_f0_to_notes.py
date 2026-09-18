from __future__ import annotations

import math
import unittest

from ml.training.convert_melody_f0_to_notes import (
    bridge_same_pitch_gaps,
    convert_f0_reference,
    frequency_to_midi,
    quantize_with_hysteresis,
)


def hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


class MelodyF0ConversionTests(unittest.TestCase):
    def test_frequency_to_midi_is_calibrated_at_a4(self) -> None:
        self.assertAlmostEqual(frequency_to_midi(440.0), 69.0)
        self.assertAlmostEqual(frequency_to_midi(220.0), 57.0)
        with self.assertRaises(ValueError):
            frequency_to_midi(0.0)

    def test_hysteresis_prevents_semitone_boundary_chatter(self) -> None:
        labels = quantize_with_hysteresis([69.0, 69.52, 69.48, 69.51, 69.7], 0.55)
        self.assertEqual(labels, [69, 69, 69, 69, 70])

    def test_bridges_only_short_same_pitch_unvoiced_gaps(self) -> None:
        self.assertEqual(
            bridge_same_pitch_gaps([60, 60, None, 60, 60], 0.02, 0.08),
            [60, 60, 60, 60, 60],
        )
        self.assertEqual(
            bridge_same_pitch_gaps([60, 60, None, 61, 61], 0.02, 0.08),
            [60, 60, None, 61, 61],
        )
        self.assertEqual(
            bridge_same_pitch_gaps([60, None, None, None, None, None, 60], 0.02, 0.08),
            [60, None, None, None, None, None, 60],
        )

    def test_conversion_ignores_short_pitch_spike_and_keeps_phrase(self) -> None:
        hop = 0.01
        frames = []
        for index in range(40):
            midi = 69.0 + 0.12 * math.sin(index)
            if 18 <= index < 20:
                midi = 72.0
            frames.append((index * hop, hz(midi)))
        notes, policy = convert_f0_reference(
            frames,
            median_filter_frames=5,
            hysteresis_semitones=0.55,
            bridge_gap_seconds=0.08,
            minimum_stable_run_seconds=0.035,
            minimum_note_seconds=0.08,
        )
        self.assertEqual([note["midi"] for note in notes], [69])
        self.assertAlmostEqual(notes[0]["duration"], 0.4)
        self.assertEqual(policy["outputNotes"], 1)


if __name__ == "__main__":
    unittest.main()
