import copy
import unittest

from ml.training.merge_focused_transcription import fuse_focused_transcription


class FocusedTranscriptionFusionTests(unittest.TestCase):
    def test_replaces_only_requested_family(self):
        primary = {
            "title": "same recording",
            "notes": [
                {"midi": 40, "time": 1.0, "duration": 0.4, "instrument": "electric_bass"},
                {"midi": 60, "time": 1.0, "duration": 0.2, "instrument": "voice"},
                {"midi": 64, "time": 2.0, "duration": 0.2, "instrument": "voice"},
                {"midi": 67, "time": 2.1, "duration": 0.3, "instrument": "guitar"},
            ],
        }
        focused = {
            "notes": [
                {"midi": 62, "time": 1.0, "duration": 0.6, "instrument": "voice"},
                {"midi": 65, "time": 2.0, "duration": 0.4, "instrument": "voice"},
            ]
        }
        original = copy.deepcopy(primary)
        result = fuse_focused_transcription(primary, focused)
        self.assertEqual(primary, original)
        self.assertEqual(len(result["notes"]), 4)
        self.assertEqual(
            [note["midi"] for note in result["notes"] if note["instrument"] == "voice"],
            [62, 65],
        )
        self.assertEqual(result["focusedTranscriptionFusion"]["primaryTargetNotesRemoved"], 2)
        self.assertEqual(result["focusedTranscriptionFusion"]["focusedNotesInserted"], 2)

    def test_fails_closed_when_focused_family_is_absent(self):
        primary = {"notes": [{"midi": 60, "time": 1.0, "duration": 0.2, "instrument": "voice"}]}
        focused = {"notes": [{"midi": 40, "time": 1.0, "duration": 0.2, "instrument": "bass"}]}
        with self.assertRaisesRegex(ValueError, "contains no valid notes"):
            fuse_focused_transcription(primary, focused)

    def test_rejects_mismatched_timeline(self):
        primary = {"notes": [{"midi": 60, "time": 100.0, "duration": 1.0, "instrument": "voice"}]}
        focused = {"notes": [{"midi": 60, "time": 10.0, "duration": 1.0, "instrument": "voice"}]}
        with self.assertRaisesRegex(ValueError, "share a timeline"):
            fuse_focused_transcription(primary, focused)

    def test_exact_same_audio_can_have_a_long_instrumental_outro(self):
        primary = {"notes": [
            {"midi": 60, "time": 10.0, "duration": 0.5, "instrument": "voice"},
            {"midi": 40, "time": 100.0, "duration": 1.0, "instrument": "electric_bass"},
        ]}
        focused = {"notes": [
            {"midi": 62, "time": 10.0, "duration": 0.5, "instrument": "voice"},
        ]}
        result = fuse_focused_transcription(
            primary,
            focused,
            known_shared_audio=True,
        )
        self.assertEqual(result["focusedTranscriptionFusion"]["timelineValidation"], "same-audio-call")
        self.assertEqual(len(result["notes"]), 2)

    def test_normalizes_independent_beat_grid_offsets_to_the_primary_clock(self):
        primary = {
            "diagnostics": {"onsetDelayAppliedSeconds": 0.01},
            "notes": [{"midi": 60, "time": 2.0, "duration": 0.5, "instrument": "voice"}],
        }
        focused = {
            "diagnostics": {"onsetDelayAppliedSeconds": -0.02},
            "notes": [{"midi": 62, "time": 2.03, "duration": 0.5, "instrument": "voice"}],
        }
        result = fuse_focused_transcription(primary, focused)
        self.assertEqual(result["notes"][0]["time"], 2.0)
        self.assertEqual(result["focusedTranscriptionFusion"]["timebaseShiftSeconds"], -0.03)

    def test_corroborated_union_filters_unsupported_notes_and_keeps_unmatched_primary(self):
        primary = {"notes": [
            {"midi": 60, "time": 1.0, "duration": 0.4, "instrument": "voice"},
            {"midi": 64, "time": 2.0, "duration": 0.4, "instrument": "voice"},
            {"midi": 67, "time": 3.0, "duration": 0.4, "instrument": "guitar"},
        ]}
        focused = {"notes": [
            {"midi": 60, "time": 1.02, "duration": 0.5, "instrument": "voice"},
            {"midi": 67, "time": 3.02, "duration": 0.5, "instrument": "voice"},
            {"midi": 71, "time": 4.0, "duration": 0.5, "instrument": "voice"},
        ]}
        result = fuse_focused_transcription(
            primary,
            focused,
            strategy="corroborated-union",
            minimum_pass_support_ratio=0.5,
            known_shared_audio=True,
        )
        voices = [note for note in result["notes"] if note["instrument"] == "voice"]
        self.assertEqual([(note["midi"], note["time"]) for note in voices], [
            (60, 1.02),
            (64, 2.0),
            (67, 3.02),
        ])
        self.assertTrue(result["focusedTranscriptionFusion"]["applied"])
        self.assertAlmostEqual(result["focusedTranscriptionFusion"]["focusedSupportRatio"], 2 / 3, places=6)

    def test_low_corroboration_falls_back_to_the_unchanged_primary(self):
        primary = {"notes": [
            {"midi": 40, "time": 2.0, "duration": 0.4, "instrument": "bass"},
            {"midi": 60, "time": 1.0, "duration": 0.4, "instrument": "voice"},
        ]}
        focused = {"notes": [
            {"midi": 71, "time": 5.0, "duration": 0.5, "instrument": "voice"},
        ]}
        result = fuse_focused_transcription(
            primary,
            focused,
            strategy="corroborated-union",
            minimum_pass_support_ratio=0.65,
            known_shared_audio=True,
        )
        self.assertEqual(result["notes"], primary["notes"])
        self.assertFalse(result["focusedTranscriptionFusion"]["applied"])

    def test_sparse_accepted_voice_pass_can_fail_closed_to_primary(self):
        primary = {"notes": [
            {"midi": 60, "time": 1.0, "duration": 0.4, "instrument": "voice"},
            {"midi": 48, "time": 10.0, "duration": 0.4, "instrument": "bass"},
        ]}
        focused = {"notes": [
            {"midi": 60, "time": 1.01, "duration": 0.2, "instrument": "voice"},
        ]}
        result = fuse_focused_transcription(
            primary,
            focused,
            strategy="corroborated-union",
            support_scope="primary-pitched",
            support_tolerance_seconds=0.05,
            minimum_pass_support_ratio=0.5,
            known_shared_audio=True,
            focused_melody_decoder={
                "enabled": True,
                "minimum_input_notes_per_second": 0.5,
                "reject_pass_below_minimum_input_density": True,
            },
        )
        diagnostics = result["focusedTranscriptionFusion"]
        self.assertEqual(result["notes"], primary["notes"])
        self.assertFalse(diagnostics["applied"])
        self.assertTrue(diagnostics["melodyDecoder"]["focusedPassRejected"])

    def test_secondary_pitched_scope_recovers_a_mislabeled_voice(self):
        primary = {"notes": [
            {"midi": 60, "time": 1.0, "duration": 0.4, "instrument": "guitar"},
            {"midi": 64, "time": 2.0, "duration": 0.4, "instrument": "guitar"},
        ]}
        focused = {"notes": [
            {"midi": 60, "time": 1.01, "duration": 0.5, "instrument": "voice"},
            {"midi": 64, "time": 2.01, "duration": 0.5, "instrument": "voice"},
        ]}
        result = fuse_focused_transcription(
            primary,
            focused,
            known_shared_audio=True,
            strategy="corroborated-union",
            support_scope="primary-target",
            fallback_support_scope="primary-pitched",
            minimum_pass_support_ratio=0.5,
            retain_unmatched_primary=False,
        )
        diagnostics = result["focusedTranscriptionFusion"]
        self.assertTrue(diagnostics["applied"])
        self.assertTrue(diagnostics["fallbackSupportScopeUsed"])
        self.assertEqual(diagnostics["effectiveSupportScope"], "primary-pitched")

    def test_secondary_pitched_scope_requires_primary_vocal_evidence_when_configured(self):
        primary = {"notes": [
            {"midi": 60, "time": 1.0, "duration": 0.4, "instrument": "alto_sax"},
            {"midi": 64, "time": 2.0, "duration": 0.4, "instrument": "alto_sax"},
        ]}
        focused = {"notes": [
            {"midi": 60, "time": 1.01, "duration": 0.5, "instrument": "voice"},
            {"midi": 64, "time": 2.01, "duration": 0.5, "instrument": "voice"},
        ]}
        result = fuse_focused_transcription(
            primary,
            focused,
            known_shared_audio=True,
            strategy="corroborated-union",
            support_scope="primary-target",
            fallback_support_scope="primary-pitched",
            minimum_pass_support_ratio=0.5,
            minimum_primary_target_notes_for_fallback=1,
            retain_unmatched_primary=False,
        )
        diagnostics = result["focusedTranscriptionFusion"]
        self.assertFalse(diagnostics["applied"])
        self.assertFalse(diagnostics["fallbackSupportScopeUsed"])
        self.assertFalse(diagnostics["fallbackEvidenceSufficient"])
        self.assertEqual(result["notes"], primary["notes"])

    def test_accepted_voice_pass_can_be_decoded_to_one_melody_path(self):
        primary = {"notes": [
            {"midi": 60, "time": 1.0, "duration": 0.4, "instrument": "voice"},
            {"midi": 64, "time": 1.0, "duration": 0.4, "instrument": "piano"},
        ]}
        focused = {"notes": [
            {"midi": 60, "time": 1.01, "duration": 0.18, "instrument": "voice"},
            {"midi": 64, "time": 1.01, "duration": 0.18, "instrument": "voice"},
            {"midi": 60, "time": 1.19, "duration": 0.18, "instrument": "voice"},
            {"midi": 64, "time": 1.19, "duration": 0.18, "instrument": "voice"},
        ]}
        result = fuse_focused_transcription(
            primary,
            focused,
            strategy="corroborated-union",
            support_scope="primary-pitched",
            support_tolerance_seconds=0.20,
            minimum_pass_support_ratio=0.5,
            retain_unmatched_primary=False,
            known_shared_audio=True,
            focused_melody_decoder={"enabled": True},
        )
        voices = [note for note in result["notes"] if note["instrument"] == "voice"]
        decoder = result["focusedTranscriptionFusion"]["melodyDecoder"]

        self.assertEqual(len(voices), 1)
        self.assertEqual(voices[0]["midi"], 60)
        self.assertEqual(decoder["inputFrames"], 2)
        self.assertEqual(decoder["outputNotes"], 1)

    def test_applied_decoder_recovers_primary_voice_anchors_it_did_not_select(self):
        primary = {"notes": [
            {"midi": 60, "time": 1.0, "duration": 0.4, "instrument": "voice"},
            {"midi": 67, "time": 3.0, "duration": 0.4, "instrument": "voice"},
            {"midi": 60, "time": 1.0, "duration": 0.4, "instrument": "piano"},
        ]}
        focused = {"notes": [
            {"midi": 60, "time": 1.01, "duration": 0.18, "instrument": "voice"},
            {"midi": 60, "time": 1.19, "duration": 0.18, "instrument": "voice"},
        ]}
        result = fuse_focused_transcription(
            primary,
            focused,
            strategy="corroborated-union",
            support_scope="primary-pitched",
            support_tolerance_seconds=0.20,
            minimum_pass_support_ratio=0.5,
            retain_unmatched_primary=False,
            known_shared_audio=True,
            focused_melody_decoder={"enabled": True},
        )
        voices = [note for note in result["notes"] if note["instrument"] == "voice"]
        decoder = result["focusedTranscriptionFusion"]["melodyDecoder"]

        self.assertEqual([(note["midi"], note["time"]) for note in voices], [
            (60, 1.01),
            (67, 3.0),
        ])
        self.assertEqual(decoder["primaryAnchorNotesRecovered"], 1)

    def test_primary_voice_anchor_replaces_a_conflicting_near_onset_guess(self):
        primary = {"notes": [
            {"midi": 60, "time": 1.0, "duration": 0.4, "instrument": "voice"},
            {"midi": 64, "time": 1.0, "duration": 0.4, "instrument": "piano"},
        ]}
        focused = {"notes": [
            {"midi": 64, "time": 1.01, "duration": 0.18, "instrument": "voice"},
            {"midi": 64, "time": 1.19, "duration": 0.18, "instrument": "voice"},
        ]}
        result = fuse_focused_transcription(
            primary,
            focused,
            strategy="corroborated-union",
            support_scope="primary-pitched",
            support_tolerance_seconds=0.20,
            minimum_pass_support_ratio=0.5,
            retain_unmatched_primary=False,
            known_shared_audio=True,
            focused_melody_decoder={"enabled": True},
        )
        voices = [note for note in result["notes"] if note["instrument"] == "voice"]
        decoder = result["focusedTranscriptionFusion"]["melodyDecoder"]

        self.assertEqual([(note["midi"], note["time"]) for note in voices], [(60, 1.0)])
        self.assertTrue(voices[0]["focusedMelodyAnchorRecovered"])
        self.assertEqual(decoder["primaryAnchorsOverridingFocusedCandidates"], 1)

    def test_accepted_fragmented_pass_can_decode_candidates_beyond_exact_support(self):
        primary = {"notes": [
            {"midi": 60, "time": 1.0, "duration": 0.4, "instrument": "voice"},
            {"midi": 48, "time": 4.0, "duration": 0.4, "instrument": "bass"},
        ]}
        focused = {"notes": [
            {"midi": 60, "time": 1.01, "duration": 0.18, "instrument": "voice"},
            {"midi": 62, "time": 2.00, "duration": 0.18, "instrument": "voice"},
        ]}
        result = fuse_focused_transcription(
            primary,
            focused,
            strategy="corroborated-union",
            support_scope="primary-target",
            support_tolerance_seconds=0.04,
            minimum_pass_support_ratio=0.5,
            retain_unmatched_primary=False,
            known_shared_audio=True,
            focused_melody_decoder={
                "enabled": True,
                "minimum_input_notes_per_second": 0.0,
                "decode_all_candidates_after_acceptance": True,
            },
        )
        voices = [note for note in result["notes"] if note["instrument"] == "voice"]
        decoder = result["focusedTranscriptionFusion"]["melodyDecoder"]

        self.assertEqual([(note["midi"], note["time"]) for note in voices], [
            (60, 1.01),
            (62, 2.0),
        ])
        self.assertEqual(decoder["activationNotes"], 1)
        self.assertTrue(decoder["decodedAllFocusedCandidatesAfterAcceptance"])


if __name__ == "__main__":
    unittest.main()
