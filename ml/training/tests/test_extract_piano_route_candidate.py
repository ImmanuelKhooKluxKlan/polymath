import unittest

from ml.training.extract_piano_route_candidate import extract_direct_piano


class ExtractPianoRouteCandidateTests(unittest.TestCase):
    def test_extracts_dominant_piano_stream_and_preserves_coordinates(self) -> None:
        payload = {
            "notes": [
                {"midi": 64, "time": 1.0, "duration": 0.5, "instrument": "piano"},
                {"midi": 60, "time": 0.5, "duration": 0.4, "instrument": "acoustic_piano"},
                {"midi": 40, "time": 0.5, "duration": 0.2, "instrument": "electric_bass"},
            ]
        }
        output, diagnostics = extract_direct_piano(
            payload, minimum_share=0.6, minimum_notes=2
        )
        self.assertEqual([note["midi"] for note in output["notes"]], [60, 64])
        self.assertEqual(output["notes"][0]["time"], 0.5)
        self.assertEqual(output["instrumentGroups"], ["acoustic_piano"])
        self.assertEqual(diagnostics["decision"], "DIRECT_PIANO_ROUTE")
        self.assertAlmostEqual(diagnostics["pianoShare"], 2 / 3, places=6)

    def test_rejects_a_sparse_residual_piano_stream(self) -> None:
        payload = {
            "notes": [
                {"midi": 60, "time": 0.0, "instrument": "acoustic_piano"},
                {"midi": 48, "time": 0.0, "instrument": "electric_bass"},
                {"midi": 36, "time": 0.0, "instrument": "drums"},
            ]
        }
        with self.assertRaisesRegex(ValueError, "not dominant enough"):
            extract_direct_piano(payload, minimum_share=0.5, minimum_notes=1)

    def test_optionally_unifies_acoustic_and_electric_piano_streams(self) -> None:
        payload = {
            "notes": [
                {"midi": 48, "time": 0.0, "instrument": "acoustic_piano"},
                {"midi": 72, "time": 0.0, "instrument": "electric_piano"},
                {"midi": 36, "time": 0.0, "instrument": "electric_bass"},
            ]
        }

        output, diagnostics = extract_direct_piano(
            payload,
            minimum_share=0.6,
            minimum_notes=2,
            include_electric_piano=True,
        )

        self.assertEqual([note["midi"] for note in output["notes"]], [48, 72])
        self.assertEqual(
            diagnostics["includedPianoFamilies"],
            ["acoustic_piano", "electric_piano"],
        )


if __name__ == "__main__":
    unittest.main()
