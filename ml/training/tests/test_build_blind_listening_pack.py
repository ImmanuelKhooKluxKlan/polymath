import json
import tempfile
import unittest
import wave
from pathlib import Path

from ml.training.build_blind_listening_pack import (
    copy_wav_excerpt,
    mapped_reference_payload,
)


class BlindListeningPackTests(unittest.TestCase):
    def test_original_full_mix_is_physically_trimmed_to_review_duration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            destination = root / "excerpt.wav"
            with wave.open(str(source), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(1000)
                writer.writeframes(b"\x01\x00" * 5000)

            result = copy_wav_excerpt(source, destination, 1.25)

            self.assertEqual(result["frames"], 1250)
            self.assertEqual(result["seconds"], 1.25)
            self.assertEqual(result["sourceSeconds"], 5.0)
            self.assertTrue(result["truncated"])
            with wave.open(str(destination), "rb") as reader:
                self.assertEqual(reader.getnframes(), 1250)

    def test_excerpt_duration_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            copy_wav_excerpt(Path("unused.wav"), Path("unused-output.wav"), 0)

    def test_already_aligned_reference_keeps_its_source_timeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            reference = Path(temporary) / "aligned-reference.json"
            reference.write_text(
                json.dumps(
                    {
                        "notes": [
                            {
                                "time": 8.25,
                                "duration": 0.5,
                                "midi": 60,
                                "velocity": 0.7,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            payload = mapped_reference_payload(
                reference,
                None,
                0,
                reference_already_aligned=True,
            )

            self.assertTrue(payload["referenceAlreadyAligned"])
            self.assertEqual(payload["notes"][0]["time"], 8.25)
            self.assertNotIn("frozenAlignmentSha256", payload)

    def test_unaligned_reference_still_requires_an_alignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            reference = Path(temporary) / "reference.json"
            reference.write_text(json.dumps({"notes": []}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "alignment is required"):
                mapped_reference_payload(reference, None, 0)


if __name__ == "__main__":
    unittest.main()
