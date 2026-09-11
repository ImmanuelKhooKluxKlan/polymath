import math
import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from ml.training.audio_chroma_alignment import (
    AudioFeatures,
    align_files,
    constrained_dtw,
    coordinate_matches,
    estimate_pitch_class_shift,
    extract_chroma_features,
    map_time,
    path_to_anchors,
)


def synthetic_song(sample_rate: int = 8000, seconds: float = 14.0) -> np.ndarray:
    samples = np.zeros(int(sample_rate * seconds), dtype=np.float32)
    chords = ((60, 64, 67), (57, 60, 64), (53, 57, 60), (55, 59, 62))
    for beat in range(int(seconds * 2)):
        start = int(beat * 0.5 * sample_rate)
        end = min(len(samples), start + int(0.44 * sample_rate))
        time = np.arange(end - start, dtype=np.float32) / sample_rate
        envelope = np.minimum(1.0, time / 0.02) * np.exp(-time * 1.8)
        chord = chords[beat % len(chords)]
        signal = sum(
            np.sin(2 * math.pi * (440.0 * 2 ** ((midi - 69) / 12)) * time)
            for midi in chord
        ) / len(chord)
        samples[start:end] += 0.55 * signal * envelope
    return samples


class AudioChromaAlignmentTest(unittest.TestCase):
    def test_note_support_does_not_claim_pairs_beyond_250ms(self):
        anchors = [
            {"referenceTime": 0.0, "observedTime": 0.0},
            {"referenceTime": 4.0, "observedTime": 4.0},
        ]
        reference = [{"midi": 60, "time": 1.0, "duration": 0.4}]
        inside = [{"midi": 60, "time": 1.24, "duration": 0.4, "sourceIndex": 7}]
        outside = [{"midi": 60, "time": 1.251, "duration": 0.4, "sourceIndex": 8}]

        self.assertEqual(len(coordinate_matches(reference, inside, anchors)), 1)
        self.assertEqual(coordinate_matches(reference, outside, anchors), [])

    def test_ambiguous_key_evidence_does_not_transpose_every_label(self):
        chroma = np.zeros((8, 12), dtype=np.float32)
        chroma[:, 0] = 1.0
        observed_chroma = np.zeros((8, 12), dtype=np.float32)
        observed_chroma[:, 0] = 0.99
        observed_chroma[:, 5] = 1.0
        common = {
            "times": np.arange(8, dtype=np.float64) * 0.5,
            "onset": np.ones(8, dtype=np.float32),
            "energy": np.ones(8, dtype=np.float32),
            "duration_seconds": 4.0,
            "sample_rate": 8000,
            "hop_seconds": 0.5,
        }
        reference = AudioFeatures(chroma=chroma, **common)
        observed = AudioFeatures(chroma=observed_chroma, **common)
        shift, scores = estimate_pitch_class_shift(reference, observed, 1.0, 0.0)
        self.assertEqual(int(np.argmax(scores)), 7)
        self.assertEqual(shift, 0)

    def test_offset_and_local_speed_are_recovered_monotonically(self):
        sample_rate = 8000
        reference_samples = synthetic_song(sample_rate)
        scale = 1.035
        offset_seconds = 1.15
        observed_seconds = offset_seconds + len(reference_samples) / sample_rate * scale + 0.5
        observed_samples = np.zeros(int(observed_seconds * sample_rate), dtype=np.float32)
        output_start = int(offset_seconds * sample_rate)
        output_length = int(len(reference_samples) * scale)
        source_positions = np.linspace(0, len(reference_samples) - 1, output_length)
        stretched = np.interp(source_positions, np.arange(len(reference_samples)), reference_samples)
        observed_samples[output_start : output_start + output_length] = stretched

        reference = extract_chroma_features(
            reference_samples, sample_rate, hop_seconds=0.05, fft_seconds=0.128
        )
        observed = extract_chroma_features(
            observed_samples, sample_rate, hop_seconds=0.05, fft_seconds=0.128
        )
        shift, _scores = estimate_pitch_class_shift(reference, observed, scale, offset_seconds)
        self.assertEqual(shift, 0)
        path, _metrics = constrained_dtw(
            reference,
            observed,
            coarse_scale=scale,
            coarse_offset=offset_seconds,
            band_seconds=0.8,
            pitch_class_shift=shift,
        )
        anchors = path_to_anchors(path, reference, observed, interval_seconds=0.5)
        predicted = [map_time(value, anchors) for value in (2.0, 5.0, 9.0, 12.0)]
        expected = [value * scale + offset_seconds for value in (2.0, 5.0, 9.0, 12.0)]
        errors = [abs(first - second) for first, second in zip(predicted, expected)]
        self.assertLess(float(np.median(errors)), 0.10)
        self.assertTrue(all(
            right["observedTime"] >= left["observedTime"]
            for left, right in zip(anchors, anchors[1:])
        ))

    def test_full_alignment_writes_gradient_safe_supervision_package(self):
        sample_rate = 8000
        reference_samples = synthetic_song(sample_rate)
        scale = 1.02
        offset_seconds = 0.75
        output_length = int(len(reference_samples) * scale)
        observed_samples = np.zeros(output_length + int(1.25 * sample_rate), dtype=np.float32)
        source_positions = np.linspace(0, len(reference_samples) - 1, output_length)
        observed_samples[
            int(offset_seconds * sample_rate) : int(offset_seconds * sample_rate) + output_length
        ] = np.interp(source_positions, np.arange(len(reference_samples)), reference_samples)

        notes = [
            {
                "midi": 60 + (index % 4),
                "time": 0.6 + index * 0.7,
                "duration": 0.35,
                "velocity": 0.72,
                "instrument": "acoustic_piano",
            }
            for index in range(18)
        ]
        observed_notes = [
            {
                **note,
                "time": note["time"] * scale + offset_seconds,
                "instrument": "clean_electric_guitar",
            }
            for note in notes
        ]

        def write_wav(path: Path, values: np.ndarray) -> None:
            pcm = np.clip(values * 32767.0, -32768, 32767).astype("<i2")
            with wave.open(str(path), "wb") as destination:
                destination.setnchannels(1)
                destination.setsampwidth(2)
                destination.setframerate(sample_rate)
                destination.writeframes(pcm.tobytes())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_audio = root / "reference.wav"
            observed_audio = root / "observed.wav"
            reference_json = root / "reference.json"
            observed_json = root / "observed.json"
            output = root / "alignment"
            write_wav(reference_audio, reference_samples)
            write_wav(observed_audio, observed_samples)
            reference_json.write_text(json.dumps({"notes": notes}), encoding="utf-8")
            observed_json.write_text(json.dumps({"notes": observed_notes}), encoding="utf-8")

            report = align_files(
                reference_audio,
                observed_audio,
                reference_json,
                observed_json,
                output,
                hop_seconds=0.05,
                band_seconds=1.2,
            )
            package = json.loads((output / "aligned-training-labels.json").read_text("utf-8"))

        self.assertEqual(package["schema"], "polymath-supervision-package-v1")
        self.assertTrue(report["matches"])
        self.assertTrue(all("sourceIndex" in match["observed"] for match in report["matches"]))
        self.assertTrue(package["alignment"]["qualityWindows"])
        self.assertTrue(all("trainingEligible" in note for note in package["notes"]))
        self.assertTrue(any(note["trainingEligible"] for note in package["notes"]))


if __name__ == "__main__":
    unittest.main()
