"""Measure the frozen compaction hypotheses without changing production."""

import argparse
import copy
import json
from pathlib import Path

from ml.training.evaluate_piano_arranger import evaluate, load_json, normalize_notes, notes_inside_ranges, prepare_reference_notes, trusted_source_ranges
from ml.training.train_muscriptor_piano import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Compaction probe folder already exists")
    from server.piano_arranger_pipeline import DEFAULT_ARRANGER_PROFILE, DEFAULT_RECOVERY_PROFILE, DEFAULT_REGISTER_PROFILE, run_pipeline
    config = load_json(args.config)
    if config["schema"] != "polymath-piano-compaction-probes-v1" or config["productionMutationAllowed"] is not False:
        raise ValueError("Unexpected compaction experiment scope")
    for name in ("source", "reference", "alignment"):
        item = config[name]
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Compaction probe input changed: {name}")
    source = load_json(Path(config["source"]["path"]))
    alignment = load_json(Path(config["alignment"]["path"]))
    reference = prepare_reference_notes({"referenceAlreadyAligned": True, "referenceTransposeSemitones": 0}, load_json(Path(config["reference"]["path"])), alignment)
    profiles = [Path(path) for path in (DEFAULT_ARRANGER_PROFILE, DEFAULT_RECOVERY_PROFILE, DEFAULT_REGISTER_PROFILE)]
    base_profile, recovery, register = [load_json(path) for path in profiles]
    args.out.mkdir(parents=True)
    results = []
    for variant in config["variants"]:
        profile = copy.deepcopy(base_profile)
        full_mix = profile["decoder"]["defaultFullMix"]
        full_mix["harmonyWindowSeconds"] = float(variant["harmonyWindowSeconds"])
        full_mix["bassGestureCompression"]["enabled"] = bool(variant["bassCompressionEnabled"])
        # Preserve the production profile ID: renaming it may change dispatch.
        payload, diagnostics = run_pipeline(copy.deepcopy(source), "full", profile, recovery, register)
        observed = notes_inside_ranges(normalize_notes(payload), trusted_source_ranges(alignment))
        metrics = evaluate(reference, observed)
        path = args.out / f"{variant['id']}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        result = {"id": variant["id"], "parameters": variant, "output": str(path), "sha256": sha256_file(path), "metrics": metrics, "cyclicTextureApplied": bool(payload["pianoArrangement"]["sourceSupportedCyclicHarmony"].get("applied")), "bassNotesRemoved": payload["pianoArrangement"]["bassGestureCompression"].get("removedNotes", 0)}
        results.append(result)
        print(json.dumps({"id": result["id"], "exact100ms": metrics["exactPitchOnset100ms"], "exact250ms": metrics["exactPitchOnset250ms"], "notes": len(observed), "bassNotesRemoved": result["bassNotesRemoved"]}), flush=True)
    report = {"schema": "polymath-piano-compaction-probe-results-v1", "configSha256": sha256_file(args.config), "profileInputs": [{"path": str(path), "sha256": sha256_file(path)} for path in profiles], "productionChanged": False, "weightsUpdated": False, "results": results}
    (args.out / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
