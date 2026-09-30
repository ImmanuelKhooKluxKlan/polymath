from __future__ import annotations

import unittest

from ml.training.augment_texture_audit_multiview import local_pitch_class_rows


class AugmentTextureAuditMultiviewTests(unittest.TestCase):
    def test_counts_agreement_without_reference_data(self) -> None:
        vocals = [
            {"midi": 60, "time": 1.02, "velocity": 0.8},
            {"midi": 64, "time": 1.20, "velocity": 0.7},
        ]
        backing = [{"midi": 72, "time": 0.97, "velocity": 0.6}]
        rows = local_pitch_class_rows(
            {
                "vocals": (vocals, [note["time"] for note in vocals]),
                "backing": (backing, [note["time"] for note in backing]),
            },
            1.0,
            attack_window=0.05,
            context_window=0.25,
        )

        c = rows[0]
        e = rows[4]
        self.assertEqual(c["viewAgreementCount"], 2)
        self.assertEqual(c["attackAgreementCount"], 2)
        self.assertEqual(e["viewAgreementCount"], 1)
        self.assertEqual(e["attackAgreementCount"], 0)
        self.assertFalse(c["views"]["fullMix"]["available"])


if __name__ == "__main__":
    unittest.main()
