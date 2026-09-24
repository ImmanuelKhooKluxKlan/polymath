import unittest

from ml.training.prepare_overlap_validation import clip_song_notes, offset_starts


class PrepareOverlapValidationTests(unittest.TestCase):
    def test_offset_starts_do_not_reopen_zero_aligned_pass(self):
        self.assertEqual(offset_starts(12.0, 5.0, 2.5), [2.5, 7.5])

    def test_note_crossing_shifted_boundary_gets_continuation_flags(self):
        notes = [{
            "midi": 60,
            "time": 2.0,
            "duration": 6.0,
            "velocity": 0.5,
            "instrument": "acoustic_piano",
        }]
        clipped = clip_song_notes(notes, 2.5, 7.5)
        self.assertEqual(len(clipped), 1)
        self.assertEqual(clipped[0]["time"], 0.0)
        self.assertEqual(clipped[0]["duration"], 5.0)
        self.assertTrue(clipped[0]["continuedFromPreviousClip"])
        self.assertTrue(clipped[0]["continuesIntoNextClip"])

    def test_short_overlap_is_preserved_with_local_time(self):
        notes = [{
            "midi": 64,
            "time": 6.0,
            "duration": 0.2,
            "velocity": 0.4,
            "instrument": "acoustic_piano",
        }]
        clipped = clip_song_notes(notes, 2.5, 7.5)
        self.assertEqual(clipped[0]["time"], 3.5)
        self.assertAlmostEqual(clipped[0]["duration"], 0.2)
        self.assertFalse(clipped[0]["continuedFromPreviousClip"])
        self.assertFalse(clipped[0]["continuesIntoNextClip"])

    def test_sub_10ms_fragment_before_boundary_is_not_discarded(self):
        notes = [{
            "midi": 66,
            "time": 4.995,
            "duration": 0.1,
            "velocity": 0.5,
            "instrument": "acoustic_piano",
        }]
        left = clip_song_notes(notes, 0.0, 5.0)
        right = clip_song_notes(notes, 5.0, 10.0)
        self.assertAlmostEqual(left[0]["duration"], 0.005)
        self.assertTrue(left[0]["continuesIntoNextClip"])
        self.assertEqual(right[0]["time"], 0.0)
        self.assertTrue(right[0]["continuedFromPreviousClip"])


if __name__ == "__main__":
    unittest.main()
