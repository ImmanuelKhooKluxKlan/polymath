import unittest

from piano_arranger import (
    MAX_ARRANGED_NOTES_PER_SECOND,
    PIANO_MAX_MIDI,
    PIANO_MIN_MIDI,
    arrange_payload,
)


def note(midi, time, instrument, duration=0.3, velocity=0.75):
    return {
        "midi": midi,
        "time": time,
        "duration": duration,
        "velocity": velocity,
        "instrument": instrument,
    }


class PianoLegatoTests(unittest.TestCase):
    def test_shapes_connected_harmony_with_a_long_release(self):
        notes = [note(72, 0, 'voice', duration=0.2), note(43, 0, 'electric_bass')]
        for index, onset in enumerate((0.0, 0.3, 0.6, 0.9)):
            for midi in (60, 64, 67, 71):
                notes.append(
                    note(midi + index, onset, 'clean_electric_guitar', duration=0.12)
                )

        result = arrange_payload({'title': 'Legato fixture', 'notes': notes}, 'full')
        harmony = [
            arranged
            for arranged in result['notes']
            if arranged['arrangementRole'] == 'harmony'
        ]

        self.assertEqual(result['pianoArrangement']['version'], 2)
        self.assertGreater(result['pianoArrangement']['legatoExtendedNotes'], 0)
        self.assertEqual(result['performance']['defaultAutoplayReleaseSeconds'], 0.62)
        self.assertTrue(any(arranged['duration'] >= 0.4 for arranged in harmony))
        self.assertGreaterEqual(len(harmony), 8)


class PianoArrangerTests(unittest.TestCase):
    def test_preserves_clean_acoustic_piano_inside_88_key_range(self):
        notes = [
            note(48 + index % 24, index * 0.125, "acoustic_piano")
            for index in range(120)
        ]
        payload = {"title": "Acoustic fixture", "notes": notes}

        result = arrange_payload(payload, "full")

        self.assertEqual(
            result["pianoArrangement"]["profile"],
            "acoustic-piano-preserve",
        )
        self.assertTrue(
            result["pianoArrangement"]["detectedAcousticPianoPerformance"]
        )
        self.assertEqual(len(result["notes"]), len(notes))
        self.assertTrue(
            all(
                PIANO_MIN_MIDI <= arranged["midi"] <= PIANO_MAX_MIDI
                for arranged in result["notes"]
            )
        )

    def test_full_mix_removes_drums_and_prioritizes_vocal_melody(self):
        notes = []
        for index in range(120):
            time = index * 0.08
            notes.extend(
                [
                    note(36, time, "drums", duration=0.05),
                    note(45 + index % 5, time, "electric_bass"),
                    note(55 + index % 12, time, "clean_electric_guitar"),
                    note(60 + index % 8, time + 0.01, "clean_electric_guitar"),
                ]
            )
            if index % 6 == 0:
                notes.append(note(67 + index % 5, time, "voice", velocity=0.82))
        notes.extend([note(10, 0, "voice"), note(120, 1, "voice")])

        result = arrange_payload({"title": "Full mix", "notes": notes}, "full")

        self.assertEqual(
            result["pianoArrangement"]["profile"],
            "full-mix-piano-reduction",
        )
        self.assertGreater(result["pianoArrangement"]["removedPercussionNotes"], 0)
        self.assertGreater(result["pianoArrangement"]["vocalMelodyNotes"], 0)
        self.assertTrue(result["vocalMelodyIncluded"])
        self.assertTrue(
            all(
                arranged.get("sourceInstrument") != "drums"
                for arranged in result["notes"]
            )
        )
        self.assertTrue(
            all(
                PIANO_MIN_MIDI <= arranged["midi"] <= PIANO_MAX_MIDI
                for arranged in result["notes"]
            )
        )
        self.assertLessEqual(
            result["pianoArrangement"]["outputNotesPerSecond"],
            MAX_ARRANGED_NOTES_PER_SECOND,
        )
        self.assertLessEqual(
            result["pianoArrangement"]["outputMaximumOnsetCluster"],
            6,
        )

    def test_instrumental_mode_excludes_voice(self):
        payload = {
            "title": "Instrumental fixture",
            "notes": [
                note(67, 0, "voice"),
                note(48, 0, "electric_bass"),
                note(60, 0, "acoustic_guitar"),
                note(64, 0.02, "acoustic_guitar"),
                note(67, 0.04, "acoustic_guitar"),
            ],
        }

        result = arrange_payload(payload, "instrumental")

        self.assertFalse(result["vocalMelodyIncluded"])
        self.assertEqual(result["pianoArrangement"]["vocalMelodyNotes"], 0)
        self.assertTrue(
            all(
                arranged.get("sourceInstrument") != "voice"
                for arranged in result["notes"]
            )
        )

    def test_suppresses_same_key_machine_gun_retriggers(self):
        notes = [
            note(60, 1.0, "clean_electric_guitar", duration=0.04),
            note(60, 1.04, "clean_electric_guitar", duration=0.04),
            note(60, 1.07, "clean_electric_guitar", duration=0.04),
            note(48, 1.0, "electric_bass"),
        ]

        result = arrange_payload({"title": "Retrigger fixture", "notes": notes}, "full")
        c_notes = [arranged for arranged in result["notes"] if arranged["midi"] == 60]

        self.assertLessEqual(len(c_notes), 1)
        self.assertGreater(
            result["pianoArrangement"]["removedRapidRetriggers"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
