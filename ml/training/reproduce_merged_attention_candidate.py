"""Require a saved plain checkpoint to reproduce its in-memory LoRA decoding."""

import argparse
import json
import os
from pathlib import Path

from ml.training.train_muscriptor_piano import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--expected-raw", type=Path, required=True)
    parser.add_argument("--expected-raw-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.out.with_suffix(".merge-reproduction.json")
    if args.out.exists() or report_path.exists():
        raise FileExistsError("Merged candidate reproduction targets already exist")
    audio = Path("/runpod-volume/jobs/phase45-foundation-eval-20260916/wane-gbt-original.wav")
    audio_hash = "c1cb84e5b11ae5314c297e96f82469e88922add794d3f9c111122ecb8278b142"
    for path, expected in ((args.checkpoint, args.checkpoint_sha256), (args.expected_raw, args.expected_raw_sha256), (audio, audio_hash)):
        if sha256_file(path) != expected:
            raise ValueError(f"Merged candidate reproduction input changed: {path}")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    from ml.training.evaluate_checkpoint import configure_piano_task_evaluation_runtime
    configure_piano_task_evaluation_runtime()
    from muscriptor import TranscriptionModel
    from ml.training.transcribe_batch import transcribe_one
    from ml.training.audit_decoder_baseline import compare_events
    model = TranscriptionModel.load_model(args.checkpoint, device="cuda")
    model._model.eval()
    payload = transcribe_one(model, audio, title="Merged attention-adapter piano research candidate", instruments=None, model_id=args.checkpoint.parent.name, provider="Polymath deterministic merged-checkpoint research replay")
    expected = json.loads(args.expected_raw.read_text())
    comparison = compare_events(expected["notes"], payload["notes"])
    report = {"schema": "polymath-merged-attention-candidate-reproduction-v1", "checkpointSha256": args.checkpoint_sha256, "expectedRawSha256": args.expected_raw_sha256, "sourceAudioSha256": audio_hash, "notes": comparison, "productionChanged": False, "verdict": "PASS" if comparison["identical"] else "REJECT_MERGED_REPRODUCTION"}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not comparison["identical"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
