import unittest

from ml.training.compare_pianist_candidates import compare


def note(midi, time, velocity, **extra):
    return {
        "_index": extra.pop("_index", int(time * 100) + midi),
        "midi": midi,
        "time": time,
        "duration": extra.pop("duration", 0.3),
        "velocity": velocity,
        "hand": "left" if midi < 60 else "right",
        **extra,
    }


class PianistCandidateComparisonTests(unittest.TestCase):
    def test_separates_key_selection_from_common_note_velocity(self):
        reference = [note(48, 0, 0.4), note(60, 0, 0.4), note(64, 1, 0.8)]
        baseline = [note(48, 0.01, 0.5), note(60, 0.01, 0.5)]
        candidate = [note(48, 0.01, 0.41), note(60, 0.01, 0.41), note(64, 1, 0.6)]

        result = compare(reference, baseline, candidate, 0.1, 0.035)

        self.assertEqual(result["headline"]["candidateMinusBaselineExactMatches"], 1)
        self.assertEqual(result["headline"]["commonExactMatches"], 2)
        self.assertEqual(result["headline"]["candidateCloserVelocity"], 2)
        self.assertEqual(
            result["byReference"]["referenceChordSize"][0]["referenceNotes"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
