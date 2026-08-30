import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml.training.prepare_audio_clips import prepare_manifest


class PrepareAudioClipsTests(unittest.TestCase):
    def test_remote_manifest_root_does_not_change_local_render_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            source.write_bytes(b"placeholder")
            manifest = root / "train.jsonl"
            manifest.write_text(json.dumps({
                "clipId": "song-00001",
                "sourceMedia": str(source),
                "sourceStart": 0,
                "durationSeconds": 5,
            }) + "\n", encoding="utf-8")
            output = root / "upload"

            def fake_render(_ffmpeg, _record, destination):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"RIFF" + b"0" * 64)

            with patch("ml.training.prepare_audio_clips.render_clip", fake_render):
                prepare_manifest(
                    manifest,
                    output,
                    root / "ffmpeg.exe",
                    manifest_audio_root="/runpod-volume/training/phase-2-v001",
                )

            prepared = json.loads((output / "prepared-train.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(
                prepared["audioClip"],
                "/runpod-volume/training/phase-2-v001/audio/train/song-00001.wav",
            )
            self.assertTrue((output / "audio" / "train" / "song-00001.wav").is_file())


if __name__ == "__main__":
    unittest.main()
