import unittest

from ml.training.audit_trusted_excerpt import audit


class TrustedExcerptAuditTests(unittest.TestCase):
    def fixtures(self):
        target = {
            "trainingExcerpt": {"endSeconds": 150, "rule": "[start,end)"},
            "notes": [{"time": 149.8, "duration": 0.2, "midi": 60}],
        }
        aligned = {
            "notes": [
                {
                    "time": 141.36,
                    "duration": 0.2,
                    "originalTime": 149.8,
                    "trainingEligible": True,
                }
            ]
        }
        alignment = {
            "anchors": [
                {"referenceTime": 0, "observedTime": 0},
                {"referenceTime": 150, "observedTime": 141.56},
            ],
            "qualityWindows": [{"referenceStart": 145, "referenceEnd": 150}],
        }
        library = {"prototypes": [{"songId": "kiss-me", "time": 141.3}]}
        return target, aligned, alignment, library

    def test_passes_a_strict_pre_cutoff_dataset(self):
        target, aligned, alignment, library = self.fixtures()
        report = audit(
            trimmed_target=target,
            aligned_target=aligned,
            alignment=alignment,
            cutoff=150,
            song_id="kiss-me",
            chord_libraries=[library],
            scanned_artifacts=[("profile.json", {"target": "trusted/aligned.json"})],
            forbidden_fragments=["old/full-target"],
        )
        self.assertEqual(report["decision"], "PASS")

    def test_fails_if_a_tail_label_or_old_target_path_leaks_back(self):
        target, aligned, alignment, library = self.fixtures()
        aligned["notes"].append(
            {"time": 142.0, "duration": 0.1, "originalTime": 150.0}
        )
        report = audit(
            trimmed_target=target,
            aligned_target=aligned,
            alignment=alignment,
            cutoff=150,
            song_id="kiss-me",
            chord_libraries=[library],
            scanned_artifacts=[("profile.json", {"target": "OLD\\FULL-TARGET\\x.json"})],
            forbidden_fragments=["old/full-target"],
        )
        self.assertEqual(report["decision"], "FAIL")


if __name__ == "__main__":
    unittest.main()
