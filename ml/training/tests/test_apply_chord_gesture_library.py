import unittest

from ml.training.apply_chord_gesture_library import (
    closest_variable_voicing,
    minimum_note_assignment,
)


class ApplyChordGestureLibraryTests(unittest.TestCase):
    def test_variable_voicing_can_reduce_a_dense_gesture(self):
        voiced = closest_variable_voicing([48, 55, 60], {0, 7})
        self.assertIsNotNone(voiced)
        self.assertEqual(len(voiced), 2)
        self.assertEqual({midi % 12 for midi in voiced}, {0, 7})

    def test_assignment_reports_one_added_pitch(self):
        mapping, additions = minimum_note_assignment([48], [48, 55])
        self.assertEqual(mapping, {0: 48})
        self.assertEqual(additions, [55])


if __name__ == "__main__":
    unittest.main()
