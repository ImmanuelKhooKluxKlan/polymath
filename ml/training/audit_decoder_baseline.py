"""Re-decode an immutable base before attributing cached-output drift to training.

This diagnostic never creates an optimizer or writes model weights. It records
the actual inference implementation/numerics and compares note events, not just
rounded aggregate scores. Historical caches without this evidence are not
assumed interchangeable across runtimes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
from typing import Any

from ml.training.train_muscriptor_piano import (
    deterministic_seed,
    load_cached_decoded_metrics,
    read_jsonl,
    sha256_file,
)


def note_events(notes: list[dict[str, Any]]) -> Counter:
    """Exclude presentation/source metadata while retaining physical events."""
    return Counter(
        (
            int(note["midi"]),
            round(float(note["time"]), 6),
            round(float(note["duration"]), 6),
            str(note.get("instrument") or "acoustic_piano"),
        )
        for note in notes
    )


def compare_events(cached: list[dict[str, Any]], fresh: list[dict[str, Any]]) -> dict[str, Any]:
    old, new = note_events(cached), note_events(fresh)
    return {
        "cachedEvents": sum(old.values()),
        "freshEvents": sum(new.values()),
        "identical": old == new,
        "addedEvents": sum((new - old).values()),
        "removedEvents": sum((old - new).values()),
    }


def implementation_hashes(package_root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(package_root)).replace("\\", "/"): sha256_file(path)
        for path in sorted(package_root.rglob("*.py"))
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import numpy as np
    import torch
    import muscriptor
    from muscriptor import TranscriptionModel
    from ml.training.decoded_checkpoint_gate import select_decoded_gate_records
    from ml.training.evaluate_checkpoint import evaluate_loaded_transcription
    from ml.training.full_song_target_development import load_config, score_payload
    from ml.training.transcribe_batch import transcribe_one

    if args.out.exists():
        raise FileExistsError(f"Audit output must be new: {args.out}")
    if sha256_file(args.base) != args.base_sha256:
        raise ValueError("Immutable base checkpoint hash does not match")
    config = load_config(args.full_song_config)
    records = select_decoded_gate_records(read_jsonl(args.validation_manifest), 6)
    cached_panel = load_cached_decoded_metrics(
        args.cached_panel, records, base_checkpoint=args.base,
        validation_manifest=args.validation_manifest, instruments=(),
    )
    cached_full = json.loads(Path(config["resolvedPaths"]["baselineRaw"]).read_text())
    seed = deterministic_seed(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    args.out.mkdir(parents=True)
    runtime = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "gpu": torch.cuda.get_device_name(0),
        "gpuComputeCapability": list(torch.cuda.get_device_capability(0)),
        "muscriptorVersion": importlib.metadata.version("muscriptor"),
        "muscriptorPythonFiles": implementation_hashes(Path(muscriptor.__file__).parent),
        "numericSeed": seed,
        "deterministicAlgorithms": torch.are_deterministic_algorithms_enabled(),
        "cublasWorkspaceConfig": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "matmulAllowTF32": torch.backends.cuda.matmul.allow_tf32,
        "cudnnAllowTF32": torch.backends.cudnn.allow_tf32,
        "matmulPrecision": torch.get_float32_matmul_precision(),
        "flashSdpEnabled": torch.backends.cuda.flash_sdp_enabled(),
        "memoryEfficientSdpEnabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "mathSdpEnabled": torch.backends.cuda.math_sdp_enabled(),
        "decode": {"useSampling": False, "beamSize": 1, "preludeForcing": True, "batchSize": 1, "instruments": []},
    }
    (args.out / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    model = TranscriptionModel.load_model(args.base, device="cuda")
    model._model.eval()
    runtime["parameterDtypes"] = sorted({str(parameter.dtype) for parameter in model._model.parameters()})
    runtime["inferenceAutocast"] = {
        key: str(value) for key, value in vars(model._model.autocast).items()
        if key in {"enabled", "dtype", "device_type"}
    }
    print(json.dumps({"stage": "fresh-baseline-full-song", "runtime": {k: v for k, v in runtime.items() if k != "muscriptorPythonFiles"}}), flush=True)
    fresh_full = transcribe_one(
        model, Path(config["resolvedPaths"]["audio"]),
        title="Immutable base under current deterministic training runtime",
        instruments=None, model_id="unchanged-base-runtime-audit",
        provider="Polymath no-optimizer inference audit",
    )
    (args.out / "fresh-baseline-full-raw.json").write_text(json.dumps(fresh_full, indent=2) + "\n")
    fresh_panel = evaluate_loaded_transcription(
        model, records, lambda message: print(message, flush=True),
        instruments=None, include_raw_predictions=True,
    )
    (args.out / "fresh-baseline-panel.json").write_text(json.dumps(fresh_panel, indent=2) + "\n")
    cached_by_id = {
        item["clipId"]: item for item in json.loads(args.cached_panel.read_text())["metrics"]["decodedClips"]
    }
    panel_comparisons = [
        {"clipId": item["clipId"], **compare_events(cached_by_id[item["clipId"]]["notes"], item["notes"])}
        for item in fresh_panel["decodedClips"]
    ]
    controls = []
    for path in args.control_raw:
        payload = json.loads(path.read_text())
        controls.append({"path": str(path), "sha256": sha256_file(path),
                         **compare_events(payload["notes"], fresh_full["notes"])})
    full_comparison = compare_events(cached_full["notes"], fresh_full["notes"])
    identical = full_comparison["identical"] and all(item["identical"] for item in panel_comparisons)
    final_base_hash = sha256_file(args.base)
    if final_base_hash != args.base_sha256:
        raise ValueError("Base checkpoint changed during the inference-only audit")
    report = {
        "schema": "polymath-unchanged-decoder-baseline-audit-v1",
        "optimizerCreated": False,
        "weightsWritten": False,
        "baseCheckpointSha256BeforeAndAfter": final_base_hash,
        "validationManifestSha256": sha256_file(args.validation_manifest),
        "fullSongConfigSha256": sha256_file(args.full_song_config),
        "cachedPanelSha256": sha256_file(args.cached_panel),
        "runtime": runtime,
        "cachedFullSongVersusFresh": full_comparison,
        "cachedTargetWindowScore": score_payload(cached_full, config),
        "freshTargetWindowScore": score_payload(fresh_full, config),
        "panelEventComparisons": panel_comparisons,
        "panelScoreComparison": {key: {"cached": cached_panel[key], "fresh": fresh_panel[key]} for key in ("100ms", "250ms")},
        "controlOutputVersusFreshBase": controls,
        "verdict": "CACHE_EVENT_REPRODUCTION_PASSED" if identical else "CACHE_RUNTIME_EQUIVALENCE_FAILED",
        "interpretation": "A mismatch is not a training gain or loss. Re-baseline the experiment under verified identical inference numerics before attributing note changes to weight updates. This opened development audit does not establish generalization accuracy.",
    }
    (args.out / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    report_path = args.out / "audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    report_path.with_suffix(".json.sha256").write_text(sha256_file(report_path) + "\n")
    print(json.dumps({"verdict": report["verdict"], "fullSong": full_comparison, "panelClipsChanged": sum(not item["identical"] for item in panel_comparisons), "controls": controls}, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--full-song-config", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--cached-panel", type=Path, required=True)
    parser.add_argument("--control-raw", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", default="polymath-phase100-v003-unconditioned")
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
