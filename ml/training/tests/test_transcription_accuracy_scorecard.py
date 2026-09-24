import json
import tempfile
import unittest
from pathlib import Path

from ml.training.transcription_accuracy_scorecard import (
    aggregate_rows,
    bottleneck_breakdown,
    build_scorecard,
    f1_from_counts,
)


def note(midi, time, duration=0.4, velocity=0.7):
    return {
        "midi": midi,
        "time": time,
        "duration": duration,
        "audioDuration": duration,
        "visualDuration": duration,
        "velocity": velocity,
    }


class TranscriptionAccuracyScorecardTests(unittest.TestCase):
    def test_f1_is_computed_from_event_counts(self):
        result = f1_from_counts(reference=10, observed=8, matches=7)
        self.assertEqual(result["precision"], 0.875)
        self.assertEqual(result["recall"], 0.7)
        self.assertEqual(result["f1"], 0.777778)

    def test_bottleneck_decomposes_exact_accuracy_headroom(self):
        result = bottleneck_breakdown(
            {
                "exactPitchOnset100ms": {"f1": 0.40},
                "pitchClassOnset100ms": {"f1": 0.55},
                "pitchClassOnset250ms": {"f1": 0.70},
            }
        )
        self.assertEqual(result["registerPlacementGap"], 0.15)
        self.assertEqual(result["coarseTimingGap"], 0.15)
        self.assertEqual(result["missingOrExtraEventGap"], 0.30)
        self.assertEqual(result["dominant"], "missingOrExtraEventGap")

    def test_aggregate_uses_micro_not_macro_f1(self):
        rows = [
            self._aggregate_row(100, 100, 90),
            self._aggregate_row(10, 10, 0),
        ]
        result = aggregate_rows(rows)
        self.assertEqual(result["accuracy"]["exactPitchOnset100ms"]["f1"], 0.818182)

    def test_development_data_cannot_certify_ninety_percent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.json"
            candidate = root / "candidate.json"
            alignment = root / "alignment.json"
            payload = {"notes": [note(60, 0.0), note(64, 1.0)]}
            reference.write_text(json.dumps(payload), encoding="utf-8")
            candidate.write_text(json.dumps(payload), encoding="utf-8")
            alignment.write_text(
                json.dumps({"anchors": [{"referenceTime": 0, "observedTime": 0}, {"referenceTime": 2, "observedTime": 2}]}),
                encoding="utf-8",
            )
            target = {
                "exactPitchOnset100msF1": 0.9,
                "pitchClassOnset250msF1": 0.95,
                "physicalOnsetAndOffset250msF1": 0.85,
                "maximumPhysicalSevereCutoffRate": 0.03,
                "maximumRapidRetriggersPer1000Notes": 2.0,
            }
            result = build_scorecard(
                {
                    "id": "test",
                    "songs": [
                        {
                            "id": "perfect-but-opened",
                            "route": "direct-piano-transcription",
                            "evidenceRole": "development",
                            "reference": str(reference),
                            "alignment": str(alignment),
                            "candidate": str(candidate),
                            "referenceAlreadyAligned": True,
                        }
                    ],
                    "objectives": {"direct-piano-transcription": target},
                }
            )
            objective = result["objectives"]["direct-piano-transcription"]
            self.assertTrue(all(objective["gates"].values()))
            self.assertFalse(objective["claimEligible"])
            self.assertEqual(
                objective["status"], "DEVELOPMENT_ONLY_NEEDS_NEW_SEALED_HOLDOUT"
            )
            self.assertEqual(result["overallClaim"], "NOT_YET_VERIFIED")

    def test_source_coverage_separates_upstream_and_arranger_loss(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.json"
            source = root / "source.json"
            candidate = root / "candidate.json"
            alignment = root / "alignment.json"
            reference.write_text(
                json.dumps({"notes": [note(60, 0.0), note(64, 1.0)]}),
                encoding="utf-8",
            )
            source.write_text(
                json.dumps({"notes": [note(60, 0.0), note(64, 1.0)]}),
                encoding="utf-8",
            )
            candidate.write_text(
                json.dumps({"notes": [note(60, 0.0)]}), encoding="utf-8"
            )
            alignment.write_text(
                json.dumps({"anchors": [{"referenceTime": 0, "observedTime": 0}, {"referenceTime": 2, "observedTime": 2}]}),
                encoding="utf-8",
            )
            result = build_scorecard(
                {
                    "id": "source-ceiling",
                    "songs": [
                        {
                            "id": "song",
                            "route": "full-mix-piano-reduction",
                            "evidenceRole": "development",
                            "reference": str(reference),
                            "source": str(source),
                            "alignment": str(alignment),
                            "candidate": str(candidate),
                            "referenceAlreadyAligned": True,
                        }
                    ],
                    "objectives": {},
                }
            )
            coverage = result["songs"][0]["sourceCoverage"]
            self.assertEqual(coverage["pitchClassOnset100msRecall"], 1.0)
            self.assertEqual(coverage["upstreamPitchClassMissingAt100ms"], 0.0)
            self.assertEqual(coverage["arrangerPitchClassRecallLossAt100ms"], 0.5)

    @staticmethod
    def _aggregate_row(reference, observed, matches):
        metric = {"matches": matches, "precision": 0, "recall": 0, "f1": 0}
        return {
            "referenceNotes": reference,
            "observedNotes": observed,
            "accuracy": {
                "exactPitchOnset50ms": metric,
                "exactPitchOnset100ms": metric,
                "exactPitchOnset250ms": metric,
                "pitchClassOnset100ms": metric,
                "pitchClassOnset250ms": metric,
            },
            "performance": {
                "physicalDurationMedianAbsoluteErrorSeconds": 0.1,
                "physicalDurationMedianRelativeError": 0.1,
                "physicalSevereCutoffs": 0,
                "velocityMeanAbsoluteError": 0.1,
                "rapidRetriggersUnder100ms": 0,
            },
            "_aggregation": {
                "physicalMatchedNotes": matches,
                "physicalOffsetMatches": matches,
                "velocityMatchedNotes": matches,
                "sourceObservedNotes": None,
                "sourceExact100Matches": None,
                "sourceExact250Matches": None,
                "sourcePitchClass100Matches": None,
                "sourcePitchClass250Matches": None,
            },
        }


if __name__ == "__main__":
    unittest.main()
