import unittest

from ml.training.search_focused_fusion_policy import (
    aggregate_song_metrics,
    promotion_gates,
    subset_trial,
)


def metrics(exact_f1=0.2, exact_recall=0.2, pitch_f1=0.4, pitch_recall=0.4):
    payload = {"rapidRetriggersUnder100ms": 0}
    for tolerance in (50, 100, 250):
        payload[f"exactPitchOnset{tolerance}ms"] = {
            "precision": exact_f1,
            "recall": exact_recall,
            "f1": exact_f1,
        }
        payload[f"pitchClassOnset{tolerance}ms"] = {
            "precision": pitch_f1,
            "recall": pitch_recall,
            "f1": pitch_f1,
        }
    for name in ("duration", "visualDuration", "physicalDuration"):
        payload[name] = {
            "medianAbsoluteErrorSeconds": 0.1,
            "severeCutoffRate": 0.1,
        }
    return payload


class FocusedFusionSearchTests(unittest.TestCase):
    def test_weighted_aggregate_respects_song_weights(self):
        result = aggregate_song_metrics([(1.0, metrics(0.2)), (3.0, metrics(0.4))])
        self.assertEqual(result["exactF1_100ms"], 0.35)

    def test_precision_only_gain_cannot_hide_recall_regression(self):
        baseline = aggregate_song_metrics([(1.0, metrics(0.2, 0.2, 0.4, 0.4))])
        candidate = aggregate_song_metrics([(1.0, metrics(0.21, 0.19, 0.4, 0.4))])
        gates = promotion_gates(baseline, candidate, [])
        self.assertTrue(gates["exactF1_100ms_improves"])
        self.assertFalse(gates["exactRecall_100ms_improves"])

    def test_subset_trial_aggregates_values_instead_of_consuming_generator(self):
        baseline = {
            name: value
            for name, value in aggregate_song_metrics([(1.0, metrics(0.2))]).items()
        }
        candidate = {
            name: value
            for name, value in aggregate_song_metrics([(1.0, metrics(0.3))]).items()
        }
        row = {
            "policy": {"id": "fixture"},
            "songs": {
                "a": {
                    "weight": 1.0,
                    "baselineFlattened": baseline,
                    "candidateFlattened": candidate,
                }
            },
        }
        result = subset_trial(row, {"a"})
        self.assertEqual(result["baseline"]["exactF1_100ms"], 0.2)
        self.assertEqual(result["candidate"]["exactF1_100ms"], 0.3)
        self.assertEqual(result["deltas"]["exactF1_100ms"], 0.1)


if __name__ == "__main__":
    unittest.main()
