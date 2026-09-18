import unittest

from ml.training.route_pianist_pitch_class_decoder import route_decoder


def payload(profile, midis, source_instrument=None):
    return {
        "pianoArrangement": {"profile": profile},
        "notes": [
            {
                "midi": midi,
                "time": 0.0,
                **(
                    {"sourceInstrument": source_instrument}
                    if source_instrument
                    else {}
                ),
            }
            for midi in midis
        ],
    }


class RoutePianistPitchClassDecoderTest(unittest.TestCase):
    def test_bypasses_decoder_for_preserved_acoustic_piano(self):
        pre = payload("acoustic-piano-preserve", [60, 64])
        decoded = payload("acoustic-piano-preserve", [60, 64, 67])

        output, report = route_decoder(pre, decoded)

        self.assertEqual([note["midi"] for note in output["notes"]], [60, 64])
        self.assertTrue(report["bypassedPitchClassDecoder"])
        self.assertEqual(report["selectedCandidate"], "pre-decoder")
        self.assertEqual(report["avoidedGeneratedNotes"], 1)
        self.assertFalse(report["targetOrReferenceDataUsed"])

    def test_keeps_decoder_for_full_mix_reduction(self):
        pre = payload("full-mix-piano-reduction", [60, 64])
        decoded = payload("full-mix-piano-reduction", [60, 64, 67])

        output, report = route_decoder(pre, decoded)

        self.assertEqual([note["midi"] for note in output["notes"]], [60, 64, 67])
        self.assertFalse(report["bypassedPitchClassDecoder"])
        self.assertEqual(report["selectedCandidate"], "decoded")

    def test_bypasses_dense_piano_misrouted_as_full_mix_from_provenance(self):
        pre = payload(
            "full-mix-piano-reduction",
            list(range(48, 78)),
            source_instrument="acoustic_piano",
        )
        decoded = payload(
            "full-mix-piano-reduction",
            list(range(48, 82)),
            source_instrument="acoustic_piano",
        )

        output, report = route_decoder(pre, decoded)

        self.assertEqual(len(output["notes"]), 30)
        self.assertTrue(report["bypassedPitchClassDecoder"])
        self.assertEqual(report["decisionReason"], "source-instrument-provenance")
        self.assertEqual(
            report["decisionInputs"]["sourceAcousticPianoShare"], 1.0
        )

    def test_does_not_confuse_rendered_piano_with_non_piano_source(self):
        pre = payload(
            "full-mix-piano-reduction",
            list(range(48, 78)),
            source_instrument="acoustic_guitar",
        )
        decoded = payload(
            "full-mix-piano-reduction",
            list(range(48, 82)),
            source_instrument="acoustic_guitar",
        )

        output, report = route_decoder(pre, decoded)

        self.assertEqual(len(output["notes"]), 34)
        self.assertFalse(report["bypassedPitchClassDecoder"])
        self.assertEqual(report["decisionReason"], "decoder-required")

    def test_rejects_missing_notes(self):
        with self.assertRaisesRegex(ValueError, "notes list"):
            route_decoder({"pianoArrangement": {}}, payload("mixed", [60]))

    def test_supports_explicit_future_profile_without_changing_default(self):
        pre = payload("isolated-piano-v2", [57])
        decoded = payload("isolated-piano-v2", [57, 60])

        output, report = route_decoder(
            pre, decoded, bypass_profiles=("isolated-piano-v2",)
        )

        self.assertEqual(len(output["notes"]), 1)
        self.assertTrue(report["bypassedPitchClassDecoder"])


if __name__ == "__main__":
    unittest.main()
