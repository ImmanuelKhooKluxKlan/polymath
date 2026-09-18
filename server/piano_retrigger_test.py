import unittest

from omr.polymath_omr.performance import shape_piano_performance
from piano_arranger import annotate_musical_retriggers, merge_phrase_retriggers
from piano_arranger_adapter import _collapse_and_limit
from piano_retrigger import (
    classify_same_key_retrigger,
    summarize_same_key_retriggers,
)


def arranged_note(
    time,
    *,
    role="melody",
    source_instrument="voice",
    source_midi=60,
    source_index=None,
):
    result = {
        "midi": 60,
        "note": "C4",
        "time": time,
        "duration": 0.16,
        "velocity": 0.76,
        "instrument": "acoustic_piano",
        "arrangementRole": role,
        "sourceInstrument": source_instrument,
        "sourceMidiBeforeArrangement": source_midi,
        "selectionProbability": 0.8,
    }
    if source_index is not None:
        result["sourceIndex"] = source_index
    return result


class SameKeyRetriggerPolicyTests(unittest.TestCase):
    def test_merges_sub_65ms_transcription_fragment(self):
        previous = arranged_note(1.0)
        current = arranged_note(1.04)

        self.assertEqual(
            classify_same_key_retrigger(previous, current),
            "duplicate",
        )

    def test_preserves_a_fast_repeat_from_the_same_musical_part(self):
        previous = arranged_note(1.0)
        current = arranged_note(1.0975)

        self.assertEqual(
            classify_same_key_retrigger(previous, current),
            "musical-repeat",
        )

    def test_merges_nearby_roles_folded_onto_one_physical_key(self):
        previous = arranged_note(1.0, role="melody", source_instrument="voice")
        current = arranged_note(
            1.12,
            role="harmony",
            source_instrument="clean_electric_guitar",
        )

        self.assertEqual(
            classify_same_key_retrigger(previous, current),
            "collision",
        )

    def test_adapter_no_longer_erases_98ms_repeated_rhythm(self):
        notes = [
            arranged_note(0.0, source_index=1),
            arranged_note(0.0975, source_index=2),
            arranged_note(0.195, source_index=3),
        ]
        profile = {
            "style": {
                "duplicateOnsetSeconds": 0.065,
                "minimumRetriggerSeconds": 0.14,
                "maximumOnsetCluster": 6,
            }
        }

        result, diagnostics = _collapse_and_limit(notes, profile)

        self.assertEqual(len(result), 3)
        self.assertEqual(diagnostics["collapsedDuplicateNotes"], 0)
        self.assertEqual(diagnostics["mergedArtificialRetriggerCollisions"], 0)
        self.assertEqual(diagnostics["preservedFastMusicalRetriggers"], 2)

    def test_fast_repeats_release_the_key_but_keep_a_resonance_tail(self):
        notes = [
            arranged_note(0.0),
            arranged_note(0.0975),
            arranged_note(0.195),
        ]

        merged, removed = merge_phrase_retriggers(notes)
        diagnostics = annotate_musical_retriggers(merged)
        performed = shape_piano_performance(
            {"instrument": "piano", "bpm": 185, "notes": merged},
            infer_pedal=False,
        )

        self.assertEqual(removed, 0)
        self.assertEqual(len(performed["notes"]), 3)
        self.assertEqual(
            diagnostics["preservedFastMusicalRetriggers65To180ms"],
            2,
        )
        self.assertLessEqual(performed["notes"][0]["audioDuration"], 0.060)
        self.assertEqual(performed["notes"][0]["releaseSeconds"], 0.5)
        self.assertGreaterEqual(
            performed["notes"][1]["retriggerReleaseSeconds"],
            0.20,
        )
        self.assertLess(performed["notes"][1]["velocity"], 0.76)
        self.assertEqual(
            [item["time"] for item in performed["notes"]],
            [0.0, 0.0975, 0.195],
        )

    def test_diagnostics_do_not_call_fast_music_an_unresolved_duplicate(self):
        diagnostics = summarize_same_key_retriggers(
            [arranged_note(0.0), arranged_note(0.0975), arranged_note(0.195)]
        )

        self.assertEqual(diagnostics["unresolvedDuplicateRetriggersUnder65ms"], 0)
        self.assertEqual(
            diagnostics["preservedFastMusicalRetriggers65To180ms"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
