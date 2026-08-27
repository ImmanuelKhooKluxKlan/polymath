import json
import tempfile
import unittest
from pathlib import Path

from ml.training.dataset_builder import DatasetError, build_dataset, deterministic_split
from ml.training.evaluate_predictions import evaluate


class DatasetBuilderTests(unittest.TestCase):
    def make_files(self, root: Path, *, rights=True):
        media = root / "song.wav"
        media.write_bytes(b"not-real-audio-but-hashable")
        package = {
            "schema": "polymath-supervision-package-v1",
            "timeline": {"sourceDurationSeconds": 10.0},
            "alignment": {"qualityWindows": [
                {"id": "w0001", "sourceStart": 0.0, "sourceEnd": 5.0, "status": "trusted"},
                {"id": "w0002", "sourceStart": 5.0, "sourceEnd": 10.0, "status": "rejected"},
            ]},
            "notes": [
                {"midi": 60, "time": 1.0, "duration": 0.5, "velocity": 0.8, "instrument": "piano", "qualityStatus": "trusted", "qualityWindowId": "w0001", "trainingEligible": True},
                {"midi": 64, "time": 6.0, "duration": 0.5, "velocity": 0.8, "instrument": "piano", "qualityStatus": "rejected", "qualityWindowId": "w0002", "trainingEligible": False},
            ],
        }
        package_path = root / "package.json"
        package_path.write_text(json.dumps(package), encoding="utf-8")
        index = {
            "songs": [{
                "songId": "song-a",
                "sourceMedia": str(media),
                "supervisionPackage": str(package_path),
                "split": "train",
                "rights": {"allowedForTraining": rights, "note": "Owned recording and arrangement"},
            }]
        }
        index_path = root / "index.json"
        index_path.write_text(json.dumps(index), encoding="utf-8")
        return index_path

    def test_build_excludes_every_clip_touching_a_rejected_window(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = build_dataset(self.make_files(root), root / "out")
            self.assertEqual(summary["counts"]["train"], 1)
            self.assertEqual(summary["rejectedClips"], 1)
            accepted = json.loads((root / "out" / "train.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(accepted["clipId"], "song-a-00000")
            self.assertEqual(len(accepted["notes"]), 1)
            rejected = json.loads((root / "out" / "rejected-clips.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(rejected["reason"], "overlaps-unapproved-window")

    def test_rights_gate_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(DatasetError, "allowedForTraining"):
                build_dataset(self.make_files(root, rights=False), root / "out")

    def test_song_split_is_deterministic(self):
        first = deterministic_split("same-song", "seed", 0.8, 0.1)
        self.assertEqual(first, deterministic_split("same-song", "seed", 0.8, 0.1))

    def test_note_evaluation_reports_false_positive_and_false_negative(self):
        reference = [
            {"index": 0, "midi": 60, "time": 1.0, "duration": 0.5},
            {"index": 1, "midi": 64, "time": 2.0, "duration": 0.5},
        ]
        predicted = [
            {"index": 0, "midi": 60, "time": 1.02, "duration": 0.48},
            {"index": 1, "midi": 67, "time": 3.0, "duration": 0.2},
        ]
        result = evaluate(reference, predicted, onset_tolerance=0.05)
        self.assertEqual(result["matchedNotes"], 1)
        self.assertEqual(result["falsePositiveNotes"], 1)
        self.assertEqual(result["falseNegativeNotes"], 1)
        self.assertAlmostEqual(result["f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
