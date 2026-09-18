import unittest
from unittest.mock import patch

from ml.training.apply_pianist_pitch_class_decoder import (
    apply_decoder,
    validate_probability_override,
)


def candidate_note(midi, time=1.0, role="harmony"):
    return {
        "midi": midi,
        "time": time,
        "duration": 0.4,
        "scoreDuration": 0.4,
        "audioDuration": 0.55,
        "velocity": 0.6,
        "arrangementRole": role,
    }


class ApplyPitchClassDecoderTest(unittest.TestCase):
    def test_validates_external_probability_contract(self):
        audit_song = {"id": "test", "cells": [{}, {}]}
        payload = {
            "songId": "test",
            "cells": [
                {"cellIndex": 0, "probabilities": [0.1] * 12},
                {"cellIndex": 1, "probabilities": [0.9] * 12},
            ],
        }
        result = validate_probability_override(audit_song, payload)
        self.assertEqual(result[0][0], 0.1)
        self.assertEqual(result[1][11], 0.9)

        payload["cells"][1]["probabilities"] = [0.5] * 11
        with self.assertRaisesRegex(ValueError, "12 values"):
            validate_probability_override(audit_song, payload)

    def test_applies_external_scores_without_loading_estimator(self):
        candidate = {"notes": [candidate_note(60), candidate_note(67)]}
        source = {
            "notes": [
                {"midi": 60, "time": 1.0, "duration": 0.3, "velocity": 0.5},
                {"midi": 64, "time": 1.0, "duration": 0.3, "velocity": 0.5},
            ]
        }
        audit_song = {
            "id": "test",
            "cells": [
                {
                    "sourceTime": 1.0,
                    "candidatePitchClasses": [0, 7],
                    "sourcePitchClasses": [0, 4],
                    "referencePitchClasses": [],
                }
            ],
        }
        model = {
            "id": "boosted-research",
            "addThreshold": 0.65,
            "removeThreshold": 0.125,
            "sourceSupportedAdditionsOnly": True,
        }
        scores = {
            "songId": "test",
            "cells": [
                {
                    "cellIndex": 0,
                    "probabilities": [
                        0.95,
                        0.01,
                        0.01,
                        0.01,
                        0.95,
                        0.01,
                        0.01,
                        0.01,
                        0.01,
                        0.01,
                        0.01,
                        0.01,
                    ],
                }
            ],
        }
        output, report = apply_decoder(
            candidate,
            source,
            audit_song,
            model,
            register_strategy="nearest-candidate",
            addition_timing="candidate",
            probabilities_override=scores,
        )
        self.assertEqual({note["midi"] % 12 for note in output["notes"]}, {0, 4})
        self.assertEqual(report["probabilitySource"], "external-frozen-scores")

    @patch(
        "ml.training.apply_pianist_pitch_class_decoder.cell_probabilities"
    )
    def test_applies_supported_addition_and_confident_removal(self, mocked):
        mocked.return_value = {
            0: {pitch_class: (0.95 if pitch_class in (0, 4) else 0.01)
                for pitch_class in range(12)}
        }
        candidate = {"notes": [candidate_note(60), candidate_note(67)]}
        source = {
            "notes": [
                {"midi": 60, "time": 1.0, "duration": 0.3, "velocity": 0.5},
                {"midi": 64, "time": 1.0, "duration": 0.3, "velocity": 0.5},
            ]
        }
        audit_song = {
            "id": "test",
            "cells": [
                {
                    "sourceTime": 1.0,
                    "candidatePitchClasses": [0, 7],
                    "sourcePitchClasses": [0, 4],
                    "referencePitchClasses": [],
                }
            ],
        }
        model = {
            "id": "model",
            "addThreshold": 0.65,
            "removeThreshold": 0.125,
            "sourceSupportedAdditionsOnly": True,
        }
        output, report = apply_decoder(
            candidate,
            source,
            audit_song,
            model,
            register_strategy="nearest-candidate",
            addition_timing="candidate",
        )
        self.assertEqual({note["midi"] % 12 for note in output["notes"]}, {0, 4})
        self.assertEqual(report["acceptedAdditions"], 1)
        self.assertEqual(report["removedNotes"], 1)

    @patch(
        "ml.training.apply_pianist_pitch_class_decoder.cell_probabilities"
    )
    def test_never_removes_the_last_note_of_a_gesture(self, mocked):
        mocked.return_value = {
            0: {pitch_class: 0.01 for pitch_class in range(12)}
        }
        candidate = {"notes": [candidate_note(60)]}
        source = {"notes": [candidate_note(60)]}
        audit_song = {
            "id": "test",
            "cells": [
                {
                    "sourceTime": 1.0,
                    "candidatePitchClasses": [0],
                    "sourcePitchClasses": [0],
                    "referencePitchClasses": [],
                }
            ],
        }
        model = {
            "addThreshold": 0.65,
            "removeThreshold": 0.125,
            "sourceSupportedAdditionsOnly": True,
        }
        output, report = apply_decoder(candidate, source, audit_song, model)
        self.assertEqual(len(output["notes"]), 1)
        self.assertEqual(report["removedNotes"], 0)
        self.assertEqual(report["protectedLastNote"], 1)

    @patch(
        "ml.training.apply_pianist_pitch_class_decoder.cell_probabilities"
    )
    def test_threshold_overrides_do_not_mutate_the_model(self, mocked):
        mocked.return_value = {
            0: {
                pitch_class: (0.6 if pitch_class == 4 else 0.5)
                for pitch_class in range(12)
            }
        }
        candidate = {"notes": [candidate_note(60)]}
        source = {"notes": [candidate_note(60), candidate_note(64)]}
        audit_song = {
            "id": "test",
            "cells": [{"sourceTime": 1.0, "sourcePitchClasses": [0, 4]}],
        }
        model = {
            "addThreshold": 0.65,
            "removeThreshold": 0.125,
            "sourceSupportedAdditionsOnly": True,
        }
        output, report = apply_decoder(
            candidate,
            source,
            audit_song,
            model,
            add_threshold_override=0.55,
            remove_threshold_override=0.4,
        )
        self.assertEqual({note["midi"] % 12 for note in output["notes"]}, {0, 4})
        self.assertEqual(report["addThreshold"], 0.55)
        self.assertEqual(report["removeThreshold"], 0.4)
        self.assertEqual(model["addThreshold"], 0.65)

    @patch(
        "ml.training.apply_pianist_pitch_class_decoder.cell_probabilities"
    )
    def test_generated_only_scope_preserves_foundation_notes(self, mocked):
        mocked.return_value = {
            0: {pitch_class: 0.01 for pitch_class in range(12)}
        }
        foundation = candidate_note(60)
        generated = candidate_note(64)
        generated["generatedBy"] = "pianist-pitch-class-decoder-v1"
        candidate = {"notes": [foundation, generated]}
        source = {"notes": [candidate_note(60), candidate_note(64)]}
        audit_song = {
            "id": "test",
            "cells": [
                {
                    "sourceTime": 1.0,
                    "candidatePitchClasses": [0, 4],
                    "sourcePitchClasses": [0, 4],
                }
            ],
        }
        model = {
            "addThreshold": 0.65,
            "removeThreshold": 0.125,
            "sourceSupportedAdditionsOnly": True,
        }

        output, report = apply_decoder(
            candidate,
            source,
            audit_song,
            model,
            remove_generated_only=True,
        )

        self.assertEqual([note["midi"] for note in output["notes"]], [60])
        self.assertEqual(report["removedNotes"], 1)
        self.assertEqual(report["protectedNonGeneratedPitchClasses"], 1)

    @patch(
        "ml.training.apply_pianist_pitch_class_decoder.cell_probabilities"
    )
    def test_upstream_confidence_ceiling_protects_high_confidence_generated_note(
        self, mocked
    ):
        mocked.return_value = {
            0: {pitch_class: 0.01 for pitch_class in range(12)}
        }
        foundation = candidate_note(60)
        generated = candidate_note(64)
        generated.update(
            {
                "generatedBy": "pianist-pitch-class-decoder-v1",
                "pitchClassProbability": 0.8,
            }
        )
        candidate = {"notes": [foundation, generated]}
        source = {"notes": [candidate_note(60), candidate_note(64)]}
        audit_song = {
            "id": "test",
            "cells": [
                {
                    "sourceTime": 1.0,
                    "candidatePitchClasses": [0, 4],
                    "sourcePitchClasses": [0, 4],
                }
            ],
        }
        model = {
            "addThreshold": 0.65,
            "removeThreshold": 0.125,
            "sourceSupportedAdditionsOnly": True,
        }

        output, report = apply_decoder(
            candidate,
            source,
            audit_song,
            model,
            remove_generated_only=True,
            maximum_generated_probability_for_removal=0.7,
        )

        self.assertEqual([note["midi"] for note in output["notes"]], [60, 64])
        self.assertEqual(report["removedNotes"], 0)
        self.assertEqual(report["protectedUpstreamConfidencePitchClasses"], 1)

    @patch(
        "ml.training.apply_pianist_pitch_class_decoder.cell_probabilities"
    )
    def test_exact_source_register_gate_rejects_octave_folded_removal(self, mocked):
        mocked.return_value = {
            0: {pitch_class: 0.01 for pitch_class in range(12)}
        }
        foundation = candidate_note(60)
        generated = candidate_note(64)
        generated.update(
            {
                "generatedBy": "pianist-pitch-class-decoder-v1",
                "pitchClassProbability": 0.6,
            }
        )
        candidate = {"notes": [foundation, generated]}
        source = {"notes": [candidate_note(60), candidate_note(76)]}
        evidence = [
            {"pitchClass": pitch_class, "exactCandidateMidiMatch": pitch_class == 0}
            for pitch_class in range(12)
        ]
        audit_song = {
            "id": "test",
            "cells": [
                {
                    "sourceTime": 1.0,
                    "candidatePitchClasses": [0, 4],
                    "sourcePitchClasses": [0, 4],
                    "sourcePitchClassEvidence": evidence,
                }
            ],
        }
        model = {
            "addThreshold": 0.65,
            "removeThreshold": 0.125,
            "sourceSupportedAdditionsOnly": True,
        }

        output, report = apply_decoder(
            candidate,
            source,
            audit_song,
            model,
            remove_generated_only=True,
            require_exact_source_midi_for_removal=True,
        )

        self.assertEqual([note["midi"] for note in output["notes"]], [60, 64])
        self.assertEqual(report["protectedRegisterEvidencePitchClasses"], 1)


if __name__ == "__main__":
    unittest.main()
