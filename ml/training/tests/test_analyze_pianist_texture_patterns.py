import unittest

from ml.training.analyze_pianist_texture_patterns import (
    assign_groups_to_cells,
    candidate_cells,
    cell_example,
    summarize_labels,
)


def note(midi, time, *, duration=0.3, instrument="clean_electric_guitar", role="harmony"):
    return {
        "midi": midi,
        "time": time,
        "duration": duration,
        "scoreDuration": duration,
        "velocity": 0.7,
        "instrument": instrument,
        "arrangementRole": role,
    }


class PianistTexturePatternTests(unittest.TestCase):
    def test_voronoi_cells_assign_two_reference_attacks_to_one_candidate(self):
        candidate = [[note(60, 0.0)], [note(62, 0.4)], [note(64, 0.8)]]
        reference = [
            [note(60, 0.02)],
            [note(67, 0.18)],
            [note(62, 0.41)],
            [note(64, 0.79)],
        ]
        assigned = assign_groups_to_cells(reference, candidate_cells(candidate))

        self.assertEqual([len(groups) for groups in assigned], [2, 1, 1])

    def test_cell_example_labels_unfold_and_measures_source_support(self):
        candidate = [note(48, 1.0), note(60, 1.0), note(64, 1.0)]
        references = [
            [note(48, 1.0)],
            [note(60, 1.15), note(64, 1.15)],
        ]
        source = [
            note(48, 1.0, duration=0.4),
            note(60, 1.0, duration=0.4),
            note(64, 1.0, duration=0.4),
        ]

        row = cell_example("song", 0, candidate, references, source, 0.3, 3.0)

        self.assertEqual(row["textureLabel"], "unfold-2")
        self.assertEqual(row["referenceSizeSignature"], "1-2")
        self.assertEqual(row["sourcePitchClassSupport"], 1.0)
        self.assertEqual(row["referenceWithinCellIoiSeconds"], [0.15])
        evidence = row["sourcePitchClassEvidence"][0]
        self.assertEqual(evidence["pitchClass"], 0)
        self.assertEqual(evidence["noteCount"], 2)
        self.assertEqual(evidence["attackCount"], 2)
        self.assertEqual(evidence["familyShares"]["guitar"], 1.0)

    def test_pitch_class_evidence_separates_sustained_distant_guitar(self):
        candidate = [note(43, 4.0, instrument="electric_bass", role="bass")]
        source = [
            note(74, 1.0, duration=4.0, instrument="clean_electric_guitar"),
            note(43, 4.0, duration=0.3, instrument="electric_bass"),
        ]

        row = cell_example("song", 0, candidate, [], source, 0.3, 2.0)
        d_evidence = row["sourcePitchClassEvidence"][2]

        self.assertEqual(d_evidence["noteCount"], 1)
        self.assertEqual(d_evidence["attackCount"], 0)
        self.assertEqual(d_evidence["sustainOnlyCount"], 1)
        self.assertEqual(d_evidence["familyShares"]["guitar"], 1.0)
        self.assertEqual(d_evidence["minimumRegisterDistanceSemitones"], 31.0)

    def test_candidate_evidence_marks_decoder_generated_notes(self):
        generated = note(62, 2.0)
        generated.update(
            {
                "generatedBy": "pianist-pitch-class-decoder-v1",
                "pitchClassProbability": 0.71,
                "sourceInstrument": "clean_electric_guitar",
            }
        )
        row = cell_example(
            "song", 0, [note(43, 2.0, instrument="electric_bass", role="bass"), generated], [], [generated], 0.3, 2.0
        )

        evidence = row["candidatePitchClassEvidence"][2]
        self.assertEqual(evidence["decoderGeneratedCount"], 1)
        self.assertEqual(evidence["decoderGeneratedShare"], 1.0)
        self.assertEqual(evidence["medianDecoderProbability"], 0.71)
        self.assertEqual(evidence["sourceFamilyShares"]["guitar"], 1.0)

    def test_summary_separates_keep_unfold_and_drop(self):
        rows = [
            {"textureLabel": "keep-vertical", "targetGestureCount": 1, "sourcePitchClassSupport": 1.0},
            {"textureLabel": "unfold-2", "targetGestureCount": 2, "sourcePitchClassSupport": 0.8},
            {"textureLabel": "drop", "targetGestureCount": 0, "sourcePitchClassSupport": 0.0},
        ]

        summary = summarize_labels(rows)

        self.assertEqual(summary["cells"], 3)
        self.assertAlmostEqual(summary["expansionRate"], 1 / 3, places=6)
        self.assertAlmostEqual(summary["dropRate"], 1 / 3, places=6)


if __name__ == "__main__":
    unittest.main()
