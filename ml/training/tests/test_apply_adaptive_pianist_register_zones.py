import unittest

from ml.training.apply_adaptive_pianist_register_zones import (
    apply_register_zones,
    choose_shifts,
)


PROFILE = {
    "schema": "polymath-adaptive-pianist-register-zones-v1",
    "id": "test",
    "outputRange": {"minimumMidi": 33, "maximumMidi": 108},
    "melody": {
        "shiftSemitones": 12,
        "fluteDominanceThreshold": 0.5,
        "shiftHarmonyWithFluteMelody": True,
        "fluteHarmonyMinimumMidi": 60,
        "stableEventMinimum": 250,
        "lowTessituraMinimumMidi": 66,
        "lowTessituraMaximumExclusiveMidi": 72,
    },
    "bass": {
        "shiftSemitones": 12,
        "stableEventMinimum": 250,
        "melodyAbsentMinimumMedianMidi": 53,
        "highMelodyMinimumMedianMidi": 75,
        "highMelodyLowBassMaximumExclusiveMidi": 40,
        "veryLowBassMaximumMidi": 36,
    },
}


def notes(count, midi, role, source):
    return [
        {
            "midi": midi,
            "time": index * 0.2,
            "duration": 0.2,
            "arrangementRole": role,
            "sourceInstrument": source,
        }
        for index in range(count)
    ]


class AdaptiveRegisterZonesTest(unittest.TestCase):
    def test_stable_mid_register_melody_moves_up_one_octave(self):
        payload = {"title": "arbitrary", "notes": notes(300, 67, "melody", "voice")}
        shifts, _diagnostics = choose_shifts(payload, PROFILE)
        self.assertEqual(shifts["melody"], 12)

    def test_short_mid_register_melody_is_not_treated_as_stable(self):
        payload = {"notes": notes(100, 67, "melody", "voice")}
        shifts, _diagnostics = choose_shifts(payload, PROFILE)
        self.assertEqual(shifts["melody"], 0)

    def test_flute_derived_melody_uses_the_flute_rule(self):
        payload = {
            "notes": notes(20, 64, "melody", "flutes")
            + notes(1, 59, "harmony", "piano")
            + notes(1, 60, "harmony", "piano")
        }
        shifts, diagnostics = choose_shifts(payload, PROFILE)
        self.assertEqual(shifts["melody"], 12)
        self.assertEqual(shifts["harmony"], 12)
        self.assertEqual(diagnostics["reasons"]["melody"], "flute-derived melody")
        output, _ = apply_register_zones(payload, PROFILE)
        harmony_midis = sorted(
            note["midi"]
            for note in output["notes"]
            if note.get("arrangementRole") == "harmony"
        )
        self.assertEqual(harmony_midis, [59, 72])

    def test_melody_absent_high_bass_labelled_score_moves_up(self):
        payload = {"notes": notes(300, 54, "bass", "electric_bass")}
        output, diagnostics = apply_register_zones(payload, PROFILE)
        self.assertEqual(diagnostics["shifts"]["bass"], 12)
        self.assertTrue(all(note["midi"] == 66 for note in output["notes"]))

    def test_song_title_does_not_affect_the_decision(self):
        left = {"title": "one", "notes": notes(300, 67, "melody", "voice")}
        right = {"title": "two", "notes": notes(300, 67, "melody", "voice")}
        self.assertEqual(choose_shifts(left, PROFILE)[0], choose_shifts(right, PROFILE)[0])

    def test_narrow_melody_harmony_gap_can_activate_separation(self):
        profile = {
            **PROFILE,
            "melody": {
                **PROFILE["melody"],
                "stableEventMinimum": 10000,
                "separationMelodyEventMinimum": 200,
                "separationHarmonyEventMinimum": 200,
                "separationMelodyMinimumMidi": 60,
                "separationMelodyMaximumExclusiveMidi": 72,
                "minimumCurrentSeparationSemitones": 0,
                "maximumCurrentSeparationSemitones": 7,
            },
        }
        narrow = {
            "notes": notes(250, 65, "melody", "voice")
            + notes(300, 58, "harmony", "guitar")
        }
        already_separated = {
            "notes": notes(250, 65, "melody", "voice")
            + notes(300, 54, "harmony", "guitar")
        }
        shifts, diagnostics = choose_shifts(narrow, profile)
        self.assertEqual(shifts["melody"], 12)
        self.assertEqual(
            diagnostics["reasons"]["melody"],
            "melody too close to harmony register",
        )
        self.assertEqual(choose_shifts(already_separated, profile)[0]["melody"], 0)

    def test_dominant_piano_source_blocks_narrow_gap_octave_shift(self):
        profile = {
            **PROFILE,
            "melody": {
                **PROFILE["melody"],
                "stableEventMinimum": 10000,
                "separationMelodyEventMinimum": 200,
                "separationHarmonyEventMinimum": 200,
                "separationMelodyMinimumMidi": 60,
                "separationMelodyMaximumExclusiveMidi": 72,
                "minimumCurrentSeparationSemitones": 0,
                "maximumCurrentSeparationSemitones": 7,
                "blockedDominantSourceFamilies": ["piano"],
                "blockedDominantSourceFamilyMinimumShare": 0.8,
            },
        }
        payload = {
            "notes": notes(250, 65, "melody", "acoustic_piano")
            + notes(300, 58, "harmony", "acoustic_piano")
        }
        shifts, diagnostics = choose_shifts(payload, profile)
        self.assertEqual(shifts["melody"], 0)
        self.assertTrue(diagnostics["melodyShiftBlockedBySourceFamily"])
        self.assertIn("piano", diagnostics["blockedReasons"]["melody"])

    def test_non_piano_source_can_still_use_narrow_gap_rule(self):
        profile = {
            **PROFILE,
            "melody": {
                **PROFILE["melody"],
                "stableEventMinimum": 10000,
                "separationMelodyEventMinimum": 200,
                "separationHarmonyEventMinimum": 200,
                "separationMelodyMinimumMidi": 60,
                "separationMelodyMaximumExclusiveMidi": 72,
                "minimumCurrentSeparationSemitones": 0,
                "maximumCurrentSeparationSemitones": 7,
                "blockedDominantSourceFamilies": ["piano"],
                "blockedDominantSourceFamilyMinimumShare": 0.8,
            },
        }
        for source in ("lead_voice", "flutes"):
            with self.subTest(source=source):
                payload = {
                    "notes": notes(250, 65, "melody", source)
                    + notes(300, 58, "harmony", "acoustic_piano")
                }
                shifts, diagnostics = choose_shifts(payload, profile)
                self.assertEqual(shifts["melody"], 12)
                self.assertFalse(diagnostics["melodyShiftBlockedBySourceFamily"])

    def test_piano_source_below_dominance_threshold_does_not_block(self):
        profile = {
            **PROFILE,
            "melody": {
                **PROFILE["melody"],
                "stableEventMinimum": 10000,
                "separationMelodyEventMinimum": 200,
                "separationHarmonyEventMinimum": 200,
                "separationMelodyMinimumMidi": 60,
                "separationMelodyMaximumExclusiveMidi": 72,
                "minimumCurrentSeparationSemitones": 0,
                "maximumCurrentSeparationSemitones": 7,
                "blockedDominantSourceFamilies": ["piano"],
                "blockedDominantSourceFamilyMinimumShare": 0.8,
            },
        }
        payload = {
            "notes": notes(150, 65, "melody", "acoustic_piano")
            + [
                {**note, "time": note["time"] + 100.0}
                for note in notes(100, 65, "melody", "lead_voice")
            ]
            + notes(300, 58, "harmony", "acoustic_piano")
        }
        shifts, diagnostics = choose_shifts(payload, profile)
        self.assertEqual(shifts["melody"], 12)
        self.assertFalse(diagnostics["melodyShiftBlockedBySourceFamily"])


if __name__ == "__main__":
    unittest.main()
