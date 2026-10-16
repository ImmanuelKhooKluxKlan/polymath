import json
import tempfile
import unittest
import wave
from pathlib import Path

from ml.training.train_muscriptor_piano import (
    TrainingError,
    audit_records,
    build_rejection_receipt,
    configure_trainable_parameters,
    conditioning_group,
    load_cached_decoded_metrics,
    optimizer_step_limit_reached,
    should_run_periodic_validation,
    write_immutable_json_receipt,
)


class TrainingAuditTests(unittest.TestCase):
    def test_rejection_receipt_records_hashes_and_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "model.safetensors"
            train_manifest = root / "train.jsonl"
            validation_manifest = root / "validation.jsonl"
            receipt_path = root / "evidence" / "rejection.json"
            base.write_bytes(b"immutable-base")
            train_manifest.write_text('{"clipId":"train"}\n', encoding="utf-8")
            validation_manifest.write_text('{"clipId":"validation"}\n', encoding="utf-8")
            receipt = build_rejection_receipt(
                base_checkpoint=base,
                train_manifest=train_manifest,
                validation_manifest=validation_manifest,
                train_audit={"clips": 1},
                validation_audit={"clips": 1},
                instrument_focus="acoustic_piano",
                conditioning_mode="unconditioned",
                hyperparameters={"noteOnTokenWeight": 1.15},
                optimizer_steps=4,
                seed="held-constant-seed",
                reproducibility={"deterministicAlgorithms": True},
                baseline_validation_loss=0.9,
                lowest_validation_loss=0.8,
                decoded_checkpoint_selection={"checkpointDecisions": []},
                reason="frozen gate failed",
            )
            digest = write_immutable_json_receipt(receipt_path, receipt)
            stored = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(stored["schema"], "polymath-muscriptor-training-rejection-v1")
            self.assertFalse(stored["candidateWritten"])
            self.assertEqual(stored["hyperparameters"]["noteOnTokenWeight"], 1.15)
            self.assertEqual(len(digest), 64)
            self.assertIn(digest, Path(str(receipt_path) + ".sha256").read_text())
            with self.assertRaisesRegex(TrainingError, "already exists"):
                write_immutable_json_receipt(receipt_path, receipt)

    def test_cached_baseline_decode_is_identity_checked_and_rescored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "model.safetensors"
            manifest = root / "validation.jsonl"
            evaluation = root / "baseline.json"
            base.write_bytes(b"checkpoint")
            manifest.write_text("{}\n", encoding="utf-8")
            note = {
                "midi": 60,
                "time": 0.25,
                "duration": 0.5,
                "instrument": "acoustic_piano",
            }
            record = {
                "clipId": "clip-001",
                "songId": "song-001",
                "sourceStart": 0.0,
                "instrumentFocus": "acoustic_piano",
                "notes": [note],
            }
            payload = {
                "schema": "polymath-checkpoint-evaluation-v1",
                "checkpoint": str(base.resolve()),
                "validationManifest": str(manifest.resolve()),
                "instrumentConstraint": ["acoustic_piano"],
                "metrics": {
                    "decodedClips": [
                        {
                            "clipId": "clip-001",
                            "songId": "song-001",
                            "sourceStart": 0.0,
                            "notes": [note],
                        }
                    ]
                },
            }
            evaluation.write_text(json.dumps(payload), encoding="utf-8")
            metrics = load_cached_decoded_metrics(
                evaluation,
                [record],
                base_checkpoint=base,
                validation_manifest=manifest,
                instruments=("acoustic_piano",),
            )
            self.assertEqual(metrics["100ms"]["microF1"], 1.0)
            payload["checkpoint"] = str(root / "other.safetensors")
            evaluation.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(TrainingError, "different checkpoint"):
                load_cached_decoded_metrics(
                    evaluation,
                    [record],
                    base_checkpoint=base,
                    validation_manifest=manifest,
                    instruments=("acoustic_piano",),
                )

    def test_zero_final_layers_trains_only_norm_and_output_head(self):
        class Parameter:
            def __init__(self):
                self.requires_grad = True

            def requires_grad_(self, value):
                self.requires_grad = value
                return self

        class Module:
            def __init__(self, count=1):
                self.values = [Parameter() for _ in range(count)]

            def parameters(self):
                return iter(self.values)

        class Transformer:
            def __init__(self):
                self.layers = [Module(2), Module(2)]

        class Model:
            def __init__(self):
                self.transformer = Transformer()
                self.out_norm = Module(1)
                self.linear = Module(1)

            def parameters(self):
                for layer in self.transformer.layers:
                    yield from layer.parameters()
                yield from self.out_norm.parameters()
                yield from self.linear.parameters()

        model = Model()
        trainable = configure_trainable_parameters(model, 0)
        self.assertEqual(len(trainable), 2)
        self.assertTrue(all(parameter.requires_grad for parameter in trainable))
        self.assertTrue(
            all(
                not parameter.requires_grad
                for layer in model.transformer.layers
                for parameter in layer.parameters()
            )
        )

    def test_periodic_validation_counts_optimizer_updates(self):
        self.assertFalse(should_run_periodic_validation(1, 0))
        self.assertFalse(should_run_periodic_validation(3, 4))
        self.assertTrue(should_run_periodic_validation(4, 4))
        self.assertTrue(should_run_periodic_validation(8, 4))
        with self.assertRaises(TrainingError):
            should_run_periodic_validation(1, -1)

    def test_optimizer_step_limit_is_explicit_and_zero_means_unbounded(self):
        self.assertFalse(optimizer_step_limit_reached(20, 0))
        self.assertFalse(optimizer_step_limit_reached(19, 20))
        self.assertTrue(optimizer_step_limit_reached(20, 20))
        self.assertTrue(optimizer_step_limit_reached(21, 20))
        with self.assertRaises(TrainingError):
            optimizer_step_limit_reached(1, -1)

    def test_conditioning_mode_matches_route_contract(self):
        self.assertEqual(conditioning_group("acoustic_piano", "instrument"), "0")
        self.assertIsNone(conditioning_group("acoustic_piano", "unconditioned"))
        with self.assertRaises(TrainingError):
            conditioning_group("acoustic_piano", "mismatched")

    def test_composed_manifest_can_preflight_through_local_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "clip.wav"
            with wave.open(str(audio), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(b"\x00\x00" * 1600)
            audit = audit_records(
                [
                    {
                        "clipId": "local-audit-001",
                        "songId": "song-001",
                        "audioClip": "/runpod-volume/training/missing.wav",
                        "localAudioSource": str(audio),
                        "durationSeconds": 0.1,
                        "instrumentFocus": "acoustic_piano",
                        "notes": [
                            {
                                "midi": 60,
                                "time": 0.0,
                                "duration": 0.1,
                                "instrument": "acoustic_piano",
                            }
                        ],
                    }
                ]
            )
            self.assertEqual(audit["clips"], 1)
            self.assertEqual(audit["songs"], 1)


if __name__ == "__main__":
    unittest.main()
