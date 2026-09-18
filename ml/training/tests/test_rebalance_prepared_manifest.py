import unittest

from ml.training.rebalance_prepared_manifest import equalize_song_weights, rebalance_records


class RebalancePreparedManifestTests(unittest.TestCase):
    def test_downweights_background_without_inflating_reviewed_weights(self) -> None:
        records = [
            {"clipId": "a", "songId": "synthetic", "exampleWeight": 1.0},
            {"clipId": "b", "songId": "synthetic", "exampleWeight": 0.35},
            {"clipId": "c", "songId": "reviewed", "exampleWeight": 1.0},
        ]
        output, audit = rebalance_records(
            records, priority_song_ids={"reviewed"}, background_weight=0.1
        )
        self.assertEqual([record["exampleWeight"] for record in output], [0.1, 0.1, 1.0])
        self.assertEqual(audit["changedRecords"], 2)
        self.assertAlmostEqual(audit["effectiveWeightBySong"]["synthetic"], 0.2)

    def test_fails_when_a_priority_song_is_missing(self) -> None:
        with self.assertRaisesRegex(ValueError, "not present"):
            rebalance_records(
                [{"songId": "synthetic"}],
                priority_song_ids={"missing"},
                background_weight=0.1,
            )

    def test_equalizes_total_song_influence_without_destroying_relative_weights(self) -> None:
        records = [
            {"clipId": "a-1", "songId": "a", "exampleWeight": 1.0},
            {"clipId": "a-2", "songId": "a", "exampleWeight": 0.5},
            {"clipId": "b-1", "songId": "b", "exampleWeight": 0.1},
            {"clipId": "b-2", "songId": "b", "exampleWeight": 0.1},
            {"clipId": "b-3", "songId": "b", "exampleWeight": 0.1},
        ]
        output, audit = equalize_song_weights(records, target_effective_weight=0.3)
        totals = {
            song_id: sum(
                record["exampleWeight"]
                for record in output
                if record["songId"] == song_id
            )
            for song_id in {"a", "b"}
        }
        self.assertAlmostEqual(totals["a"], 0.3)
        self.assertAlmostEqual(totals["b"], 0.3)
        self.assertAlmostEqual(output[0]["exampleWeight"] / output[1]["exampleWeight"], 2.0)
        self.assertEqual(audit["songs"], 2)

    def test_equalizer_rejects_an_impossible_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "above 1.0"):
            equalize_song_weights(
                [{"clipId": "a-1", "songId": "a", "exampleWeight": 1.0}],
                target_effective_weight=2.0,
            )


if __name__ == "__main__":
    unittest.main()
