import tempfile
import unittest
from pathlib import Path

from ml.training.select_medleydb_vocal_holdout import (
    archive_index,
    eligible_vocal_rows,
    is_vocal_melody,
)


class MedleyDbVocalHoldoutSelectionTests(unittest.TestCase):
    def test_requires_noninstrumental_melody_stem_with_vocal_label(self):
        metadata = {
            "instrumental": "no",
            "stems": {
                "S04": {"component": "melody", "instrument": "female singer"}
            },
        }
        self.assertEqual(is_vocal_melody(metadata, 4), (True, ["female singer"]))
        metadata["instrumental"] = "yes"
        self.assertEqual(is_vocal_melody(metadata, 4), (False, []))

    def test_joins_exactly_one_annotation_to_official_metadata(self):
        members = [
            "MDB-melody-synth/audio_mix/Artist_Song_MIX_melsynth.wav",
            "MDB-melody-synth/annotation_melody/Artist_Song_STEM_02.RESYN.csv",
            "MDB-melody-synth/audio_mix/Artist_Instrumental_MIX_melsynth.wav",
            "MDB-melody-synth/annotation_melody/Artist_Instrumental_STEM_03.RESYN.csv",
            "MDB-melody-synth/audio_mix/._ResourceFork_MIX_melsynth.wav",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            metadata = Path(temporary)
            (metadata / "Artist_Song_METADATA.yaml").write_text(
                "instrumental: 'no'\nstems:\n  S02:\n    component: melody\n"
                "    instrument: male singer\n",
                encoding="utf-8",
            )
            (metadata / "Artist_Instrumental_METADATA.yaml").write_text(
                "instrumental: 'yes'\nstems:\n  S03:\n    component: melody\n"
                "    instrument: alto saxophone\n",
                encoding="utf-8",
            )
            rows = eligible_vocal_rows(archive_index(members), metadata)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["trackId"], "Artist_Song")
        self.assertEqual(rows[0]["annotationStemNumber"], 2)


if __name__ == "__main__":
    unittest.main()
