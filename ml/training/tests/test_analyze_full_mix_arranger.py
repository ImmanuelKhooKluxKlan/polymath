import importlib.util
import sys
import unittest
from pathlib import Path


TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

SPEC = importlib.util.spec_from_file_location(
    "analyze_full_mix_arranger",
    TRAINING_DIR / "analyze_full_mix_arranger.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class AnalyzeFullMixArrangerTests(unittest.TestCase):
    def test_greedy_match_indices_matches_evaluator_pitch_rules(self):
        reference = [
            {"midi": 60, "time": 1.0, "duration": 0.4},
            {"midi": 64, "time": 2.0, "duration": 0.4},
        ]
        observed = [
            {"midi": 72, "time": 1.04, "duration": 0.4},
            {"midi": 64, "time": 2.08, "duration": 0.4},
        ]
        exact = MODULE.greedy_match_indices(reference, observed, 0.1)
        pitch_class = MODULE.greedy_match_indices(
            reference, observed, 0.1, octave_equivalent=True
        )
        self.assertEqual(exact, [(1, 1)])
        self.assertEqual(pitch_class, [(0, 0), (1, 1)])

    def test_upstream_ceiling_splits_model_and_arranger_misses(self):
        reference = [
            {"midi": 60, "time": 1.0, "duration": 0.4},
            {"midi": 62, "time": 2.0, "duration": 0.4},
            {"midi": 64, "time": 3.0, "duration": 0.4},
        ]
        raw = [
            {"midi": 48, "time": 1.01, "duration": 0.4},
            {"midi": 50, "time": 2.01, "duration": 0.4},
        ]
        candidate = [{"midi": 60, "time": 1.01, "duration": 0.4}]
        result = MODULE.upstream_recall_ceiling(reference, raw, candidate)
        self.assertEqual(result["rawPitchClassMatches250ms"], 2)
        self.assertEqual(result["candidatePitchClassMatches250ms"], 1)
        self.assertEqual(result["arrangerMissedDespiteRawSupport"], 1)
        self.assertEqual(result["upstreamMissingAt250ms"], 1)

    def test_upstream_ceiling_does_not_treat_drum_midi_as_melody(self):
        reference = [{"midi": 62, "time": 2.0, "duration": 0.4}]
        raw = [
            {
                "midi": 38,
                "time": 2.01,
                "duration": 0.04,
                "instrument": "drums",
            }
        ]
        result = MODULE.upstream_recall_ceiling(reference, raw, [])
        self.assertEqual(result["rawPitchClassMatches250ms"], 0)
        self.assertEqual(result["rawNotes"], 1)
        self.assertEqual(result["rawPitchedNotes"], 0)
        self.assertEqual(result["excludedPercussionNotes"], 1)
        self.assertEqual(result["arrangerMissedDespiteRawSupport"], 0)
        self.assertEqual(result["upstreamMissingAt250ms"], 1)

    def test_group_rows_preserve_metadata_and_report_false_notes(self):
        notes = [
            {
                "midi": 60,
                "sourceInstrument": "voice",
                "arrangementRole": "melody",
                "selectionProbability": 0.9,
            },
            {
                "midi": 40,
                "sourceInstrument": "bass",
                "arrangementRole": "bass",
                "selectionProbability": 0.2,
            },
        ]
        rows = MODULE.group_precision_rows(notes, {0}, {0}, "sourceInstrument")
        by_name = {row["name"]: row for row in rows}
        self.assertEqual(by_name["voice"]["pitchClass250Precision"], 1.0)
        self.assertEqual(by_name["voice"]["matchedMedianSelectionProbability"], 0.9)
        self.assertEqual(by_name["bass"]["unmatchedAtPitchClass250"], 1)
        self.assertEqual(by_name["bass"]["unmatchedMedianSelectionProbability"], 0.2)


if __name__ == "__main__":
    unittest.main()
