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

        self.assertEqual(result['pianoArrangement']['version'], 4)
        self.assertGreater(result['pianoArrangement']['legatoExtendedNotes'], 0)
        self.assertEqual(result['performance']['defaultAutoplayReleaseSeconds'], 0.62)
        self.assertTrue(any(arranged['duration'] >= 0.4 for arranged in harmony))
        self.assertGreaterEqual(len(harmony), 8)

    def test_places_melody_in_front_and_softens_left_hand(self):
        notes = []
        for index in range(36):
            onset = index * 0.24
            notes.extend(
                [
                    note(43, onset, 'electric_bass', duration=0.5, velocity=0.78),
                    note(55, onset, 'clean_electric_guitar', duration=0.35, velocity=0.78),
                    note(64, onset, 'clean_electric_guitar', duration=0.35, velocity=0.78),
                    note(72 + index % 3, onset, 'voice', duration=0.42, velocity=0.78),
                ]
            )

        result = arrange_payload({'title': 'Melody balance fixture', 'notes': notes}, 'full')
        melody = [item for item in result['notes'] if item['arrangementRole'] == 'melody']
        bass = [item for item in result['notes'] if item['arrangementRole'] == 'bass']
        left = [item for item in result['notes'] if item['midi'] < 60]
        right = [item for item in result['notes'] if item['midi'] >= 60]
        expression = result['pianoArrangement']['expression']

        self.assertTrue(melody)
        self.assertTrue(bass)
        self.assertGreater(min(item['velocity'] for item in melody), max(item['velocity'] for item in bass))
        self.assertGreater(
            sum(item['velocity'] for item in right) / len(right),
            sum(item['velocity'] for item in left) / len(left),
        )
        self.assertGreater(expression['rightToLeftVelocityRatio'], 1.2)
        self.assertTrue(result['performance']['melodyForwardDynamics'])
        self.assertEqual(result['performance']['profile'], 'polymath-piano-arranger-v4')
        self.assertEqual(result['arrangementProfile'], 'piano-reduction-with-midi-phrasing-v4')

    def test_preserved_piano_gets_register_balance_without_clipping(self):
        notes = []
        for index in range(60):
            onset = index * 0.18
            notes.extend(
                [
                    note(48 + index % 4, onset, 'acoustic_piano', duration=0.45, velocity=0.99),
                    note(67 + index % 5, onset, 'acoustic_piano', duration=0.38, velocity=0.99),
                ]
            )

        result = arrange_payload({'title': 'Piano register fixture', 'notes': notes}, 'full')
        left = [item for item in result['notes'] if item['hand'] == 'left']
        right = [item for item in result['notes'] if item['hand'] == 'right']

        self.assertEqual(result['pianoArrangement']['profile'], 'acoustic-piano-preserve')
        self.assertGreater(min(item['velocity'] for item in right), max(item['velocity'] for item in left))
        self.assertLess(max(item['velocity'] for item in right), 1.0)
        self.assertGreater(
            result['pianoArrangement']['expression']['rightToLeftVelocityRatio'],
            1.2,
        )


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

    def test_mislabeled_full_mix_uses_cleanup_pressure_instead_of_trusting_piano_tag(self):
        notes = [
            note(38 + index % 34, index * 0.12, "acoustic_piano", duration=1.1)
            for index in range(180)
        ]
        payload = {
            "title": "Mislabeled full mix",
            "notes": notes,
            "transcriptionCleanup": {
                "inputNotes": 230,
                "removedDuplicateNotes": 28,
                "shortenedSameKeyOverlaps": 48,
            },
        }

        result = arrange_payload(payload, "instrumental")

        self.assertEqual(
            result["pianoArrangement"]["profile"],
            "full-mix-piano-reduction",
        )
        self.assertFalse(
            result["pianoArrangement"]["detectedAcousticPianoPerformance"]
        )
        self.assertGreater(
            result["pianoArrangement"]["cleanupArtifactPressure"],
            result["pianoArrangement"]["maximumDirectCleanupPressure"],
        )
        self.assertTrue(
            any(item["arrangementRole"] == "melody" for item in result["notes"])
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

    def test_full_mix_merges_cross_role_same_key_collisions(self):
        notes = []
        for index in range(36):
            onset = index * 0.24
            notes.extend(
                [
                    note(48, onset, "electric_bass", duration=0.5),
                    note(60, onset, "voice", duration=0.35),
                    note(60, onset + 0.12, "clean_electric_guitar", duration=0.5),
                    note(64, onset + 0.12, "clean_electric_guitar", duration=0.5),
                ]
            )

        result = arrange_payload({"title": "Role collision", "notes": notes}, "full")
        c4_onsets = sorted(
            item["time"] for item in result["notes"] if item["midi"] == 60
        )

        self.assertTrue(
            all(
                current - previous >= 0.18
                for previous, current in zip(c4_onsets, c4_onsets[1:])
            )
        )


if __name__ == "__main__":
    unittest.main()
