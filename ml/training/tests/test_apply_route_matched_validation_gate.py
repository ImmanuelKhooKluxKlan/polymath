import unittest

from ml.training.apply_route_matched_validation_gate import evaluate_gate


def comparison():
    def side(f1_100, f1_250):
        return {
            "100ms": {"microF1": f1_100},
            "250ms": {"microF1": f1_250},
        }

    return {
        "schema": "polymath-checkpoint-comparison-v1",
        "instrumentConstraint": [],
        "baseline": side(0.61, 0.72),
        "candidate": side(0.62, 0.73),
    }


def timeline(f1=0.62, precision=0.64, recall=0.60, song_b=0.61, frame=0.25, duration=0.37):
    per_song = {"song-a": f1, "song-b": song_b, "song-c": f1}
    return {
        "schema": "polymath-stitched-song-timeline-score-v1",
        "songs": 3,
        "songIds": list(per_song),
        "metrics": {
            "100ms": {"microF1": f1, "precision": precision, "recall": recall},
            "250ms": {"microF1": f1 + 0.10, "precision": precision + 0.10, "recall": recall + 0.10},
        },
        "perSong": {
            song_id: {"100ms": {"microF1": value}}
            for song_id, value in per_song.items()
        },
        "songClusterBootstrap95": {
            "100ms": {"lower95": min(per_song.values())}
        },
        "diagnostics100ms": {
            "matchedNotes": 600,
            "predictedNotes": 700,
            "cutOffNotes": 20,
            "overlongNotes": 25,
            "rapidRetriggers": 2,
            "onsetAndOffset": {"f1": duration},
            "frame": {"f1": frame},
        },
    }


def training(mode="unconditioned", baseline_loss=0.8, candidate_loss=0.79):
    return {
        "schema": "polymath-muscriptor-training-run-v1",
        "conditioningMode": mode,
        "baselineValidationLoss": baseline_loss,
        "selectedValidationLoss": candidate_loss,
    }


def gate():
    return {
        "schema": "polymath-route-matched-validation-gate-v1",
        "candidateVersion": "test",
        "route": {"conditioningMode": "unconditioned", "instrumentConstraint": []},
        "requiredChecks": {
            "minimumCompleteSongs": 3,
            "clipLocal100msF1MaximumRegression": 0.0,
            "clipLocal250msF1MaximumRegression": 0.0,
            "continuous100msF1MinimumImprovement": 0.0,
            "continuous100msPrecisionMaximumRegression": 0.005,
            "continuous100msRecallMaximumRegression": 0.0,
            "continuous250msF1MaximumRegression": 0.0,
            "songBootstrapLower95MaximumRegression": 0.0,
            "maximumPerSong100msF1Regression": 0.005,
            "onsetAndOffsetF1MaximumRegression": 0.005,
            "frameF1MaximumRegression": 0.005,
            "cutOffPer1000MatchedMaximumIncrease": 2.0,
            "overlongPer1000MatchedMaximumIncrease": 2.0,
            "rapidRetriggersPer1000PredictedMaximumIncrease": 0.25,
        },
        "manualGatesStillRequired": ["listening"],
    }


class RouteMatchedValidationGateTests(unittest.TestCase):
    def test_route_matched_balanced_improvement_passes_research_gate(self):
        baseline = timeline(f1=0.61, precision=0.63, recall=0.59, song_b=0.60)
        candidate = timeline(f1=0.62, precision=0.64, recall=0.60, song_b=0.61)
        result = evaluate_gate(comparison(), baseline, candidate, training(), gate())
        self.assertTrue(result["automatedResearchGatePassed"])
        self.assertTrue(result["canProceedToListeningCanary"])
        self.assertFalse(result["productionEligible"])

    def test_wrong_conditioning_mode_fails_closed(self):
        baseline = timeline(f1=0.61, precision=0.63, recall=0.59, song_b=0.60)
        candidate = timeline(f1=0.62, precision=0.64, recall=0.60, song_b=0.61)
        result = evaluate_gate(
            comparison(), baseline, candidate, training(mode="instrument"), gate()
        )
        self.assertFalse(result["automatedResearchGatePassed"])
        self.assertIn("conditioning_mode_matches_live_route", result["failedChecks"])

    def test_aggregate_gain_cannot_hide_one_song_regression(self):
        baseline = timeline(f1=0.61, precision=0.63, recall=0.59, song_b=0.60)
        candidate = timeline(f1=0.62, precision=0.64, recall=0.60, song_b=0.58)
        result = evaluate_gate(comparison(), baseline, candidate, training(), gate())
        self.assertFalse(result["automatedResearchGatePassed"])
        self.assertIn("per_song_100ms_f1", result["failedChecks"])
        self.assertIn("song_bootstrap_lower95", result["failedChecks"])

    def test_onset_gain_cannot_hide_duration_regression(self):
        baseline = timeline(
            f1=0.61, precision=0.63, recall=0.59, song_b=0.60,
            frame=0.25, duration=0.37,
        )
        candidate = timeline(
            f1=0.62, precision=0.64, recall=0.60, song_b=0.61,
            frame=0.23, duration=0.34,
        )
        result = evaluate_gate(comparison(), baseline, candidate, training(), gate())
        self.assertFalse(result["automatedResearchGatePassed"])
        self.assertIn("onset_and_offset_f1", result["failedChecks"])
        self.assertIn("frame_f1", result["failedChecks"])


if __name__ == "__main__":
    unittest.main()
