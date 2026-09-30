import copy
import unittest

from exact_song_intro import apply_matching_exact_song_intro, evaluate_exact_song_match
from piano_arranger_pipeline import (
    DEFAULT_ARRANGER_PROFILE,
    DEFAULT_EXACT_SONG_PROFILE,
    DEFAULT_RECOVERY_PROFILE,
    DEFAULT_REGISTER_PROFILE,
    load_json,
    run_pipeline,
)


def kiss_me_source(*, title="Kiss Me original", moving=None):
    transition_times = [4.05, 6.10, 8.65, 11.04, 13.45, 15.84, 18.24, 20.60]
    moving_midis = list(moving or [63, 62, 61, 62, 63, 62, 61, 62])
    notes = []
    time = 4.05
    while time < 23.31:
        cycle = max(index for index, anchor in enumerate(transition_times) if anchor <= time + 1e-9)
        for midi in (51, 58, moving_midis[cycle], 67):
            notes.append(
                {
                    "midi": midi,
                    "time": round(time, 6),
                    "duration": 0.25,
                    "velocity": 0.5,
                    "instrument": "clean_electric_guitar",
                }
            )
        time = round(time + 0.15, 6)
    # Preserve the source-derived harmonic transition times between grid points.
    for index, time in enumerate(transition_times[1:], start=1):
        for midi in (51, 58, moving_midis[index], 67):
            notes.append(
                {
                    "midi": midi,
                    "time": time,
                    "duration": 0.25,
                    "velocity": 0.5,
                    "instrument": "clean_electric_guitar",
                }
            )
    notes.extend(
        [
            {
                "midi": 64,
                "time": 23.46,
                "duration": 0.3,
                "velocity": 0.7,
                "instrument": "voice",
            },
            {
                "midi": 27,
                "time": 193.12,
                "duration": 4.65,
                "velocity": 0.5,
                "instrument": "electric_bass",
            },
        ]
    )
    return {"title": title, "notes": notes}


def generic_candidate():
    return {
        "arrangementProfile": "polymath-pianella-v003",
        "notes": [
            {"midi": 48, "time": 5.0, "duration": 0.2, "velocity": 0.6},
            {"midi": 64, "time": 24.0, "duration": 0.2, "velocity": 0.7},
        ],
        "pianoArrangement": {},
    }


class ExactSongIntroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = load_json(DEFAULT_EXACT_SONG_PROFILE)
        cls.arranger_profile = load_json(DEFAULT_ARRANGER_PROFILE)
        cls.recovery_profile = load_json(DEFAULT_RECOVERY_PROFILE)
        cls.register_profile = load_json(DEFAULT_REGISTER_PROFILE)

    def test_all_identity_signals_match_the_intended_source(self):
        result = evaluate_exact_song_match(kiss_me_source(), "full", self.profile)

        self.assertTrue(result["matched"], result["failedChecks"])
        self.assertEqual(result["checks"]["movingPitchClassSequence"]["observed"]["sequence"], [3, 2, 1, 2, 3, 2, 1, 2])
        self.assertTrue(all(check["passed"] for check in result["checks"].values()))

    def test_research_profile_is_off_by_default_and_does_not_mutate_score(self):
        candidate = generic_candidate()
        before = copy.deepcopy(candidate)

        output, diagnostics = apply_matching_exact_song_intro(
            kiss_me_source(), candidate, "full", [self.profile]
        )

        self.assertEqual(output, before)
        self.assertFalse(diagnostics["applied"])
        self.assertEqual(diagnostics["reason"], "matched-research-profile-disabled")

    def test_explicit_research_gate_applies_only_the_intro(self):
        output, diagnostics = apply_matching_exact_song_intro(
            kiss_me_source(),
            generic_candidate(),
            "full",
            [self.profile],
            allow_research_profiles=True,
        )

        self.assertTrue(diagnostics["applied"])
        self.assertEqual(diagnostics["matchedProfileId"], "pianella-intro-boundary-aware-v004")
        self.assertNotIn(48, [note["midi"] for note in output["notes"] if note["time"] < 23.16])
        self.assertIn(64, [note["midi"] for note in output["notes"] if note["time"] >= 23.16])
        self.assertEqual(output["pianoArrangement"]["pianistMotifGrammar"]["generatedNotes"], 169)

    def test_title_only_spoof_fails_harmonic_fingerprint(self):
        result = evaluate_exact_song_match(
            kiss_me_source(moving=[63, 63, 63, 63, 63, 63, 63, 63]),
            "full",
            self.profile,
        )

        self.assertFalse(result["matched"])
        self.assertIn("movingPitchClassSequence", result["failedChecks"])

    def test_correct_audio_shape_with_another_title_fails_closed(self):
        result = evaluate_exact_song_match(
            kiss_me_source(title="Mean - original"), "full", self.profile
        )

        self.assertFalse(result["matched"])
        self.assertEqual(result["failedChecks"], ["titleTokens"])

    def test_instrumental_mode_never_uses_full_song_adapter(self):
        result = evaluate_exact_song_match(kiss_me_source(), "instrumental", self.profile)

        self.assertFalse(result["matched"])
        self.assertIn("mode", result["failedChecks"])

    def test_profile_tampering_is_detected(self):
        tampered = copy.deepcopy(self.profile)
        tampered["application"]["sourceCycleAnchorBlend"] = 0.1

        result = evaluate_exact_song_match(kiss_me_source(), "full", tampered)

        self.assertFalse(result["matched"])
        self.assertIn("profileIntegrity", result["failedChecks"])

    def test_production_wrapper_retains_the_applied_profile_identity(self):
        output, diagnostics = run_pipeline(
            kiss_me_source(),
            "full",
            self.arranger_profile,
            self.recovery_profile,
            self.register_profile,
            [self.profile],
            allow_research_exact_song_profiles=True,
        )

        self.assertTrue(diagnostics["exactSongIntro"]["applied"])
        self.assertEqual(output["arrangementProfile"], "polymath-pianella-v004-motif-v1")


if __name__ == "__main__":
    unittest.main()
