"""Research-only separation of source conditioning from output instruments.

MuScriptor's public ``instruments`` argument controls both the input class
condition and the allowed output programs. This isolated-process probe leaves
the input class condition unset while retaining the public piano output mask.
It does not alter installed package files, weights, source audio, or production
routing. The temporary method override is restored when the generator closes.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import random
from typing import Any, Iterator

from ml.training.train_muscriptor_piano import deterministic_seed, sha256_file


class PianoOutputOnlyProbe:
    """An exclusive-use adapter, deliberately not a concurrent server API."""

    def __init__(self, transcription: Any):
        self.transcription = transcription
        if "instrument_group" not in inspect.signature(transcription._build_conditions).parameters:
            raise ValueError("Unsupported MuScriptor conditioning signature")

    def transcribe(self, audio: Any, *, instruments: list[str] | None = None) -> Iterator[Any]:
        if instruments != ["acoustic_piano"]:
            raise ValueError("This probe requires the acoustic-piano output mask")
        model = self.transcription
        original = model._build_conditions
        had_instance_override = "_build_conditions" in vars(model)
        previous_override = vars(model).get("_build_conditions")

        def source_unconditioned(wav: Any, instrument_group: Any = None):
            return original(wav, None)

        model._build_conditions = source_unconditioned
        try:
            yield from model.transcribe(audio, instruments=instruments)
        finally:
            if had_instance_override:
                model._build_conditions = previous_override
            else:
                delattr(model, "_build_conditions")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--audio-sha256", required=True)
    parser.add_argument("--runtime-audit", type=Path, required=True)
    parser.add_argument("--runtime-audit-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Probe output already exists: {args.output}")
    for path, expected in ((args.base, args.base_sha256), (args.audio, args.audio_sha256), (args.runtime_audit, args.runtime_audit_sha256)):
        if sha256_file(path) != expected:
            raise ValueError(f"Immutable probe input changed: {path}")

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import numpy as np
    import torch
    import muscriptor
    from muscriptor import TranscriptionModel
    from ml.training.audit_decoder_baseline import implementation_hashes
    from ml.training.transcribe_batch import transcribe_one

    audit = json.loads(args.runtime_audit.read_text())
    if audit["verdict"] != "CACHE_EVENT_REPRODUCTION_PASSED":
        raise ValueError("The unchanged-base runtime reproduction has not passed")
    if implementation_hashes(Path(muscriptor.__file__).parent) != audit["runtime"]["muscriptorPythonFiles"]:
        raise ValueError("Installed MuScriptor implementation differs from the verified runtime")
    if str(torch.__version__) != audit["runtime"]["torch"] or torch.cuda.get_device_name(0) != audit["runtime"]["gpu"]:
        raise ValueError("Torch or GPU differs from the verified baseline runtime")
    seed = deterministic_seed("polymath-phase100-v003-unconditioned")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    model = TranscriptionModel.load_model(args.base, device="cuda")
    model._model.eval()
    payload = transcribe_one(
        PianoOutputOnlyProbe(model), args.audio,
        title="WANE all-source conditioning with piano-only output probe",
        instruments=["acoustic_piano"], model_id="phase103-piano-output-mask-probe-v001",
        provider="Polymath isolated inference-only research probe",
    )
    if any(note["instrument"] != "acoustic_piano" for note in payload["notes"]):
        raise ValueError("The requested piano output mask was not enforced")
    payload["researchProvenance"].update({
        "sourceConditioning": None,
        "allowedOutputInstruments": ["acoustic_piano"],
        "baseCheckpointSha256": args.base_sha256,
        "sourceAudioSha256": args.audio_sha256,
        "verifiedRuntimeAuditSha256": args.runtime_audit_sha256,
        "weightsUpdated": False,
        "referenceAvailableAtInference": False,
        "productionRoutingChanged": False,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "notes": len(payload["notes"]), "sha256": sha256_file(args.output), "weightsUpdated": False}), flush=True)


if __name__ == "__main__":
    main()
