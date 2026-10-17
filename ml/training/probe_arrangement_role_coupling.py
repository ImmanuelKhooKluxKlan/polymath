"""Separate roles in an existing ablation without altering production rules.

Spliced role outputs are diagnostic counterfactuals, not deployable candidates.
No reference coordinates enter the splicing function.
"""

import argparse
import copy
import json
from pathlib import Path

from .evaluate_piano_arranger import evaluate, load_json, normalize_notes, notes_inside_ranges, prepare_reference_notes, trusted_source_ranges
from .train_muscriptor_piano import sha256_file


def splice_roles(baseline, ablation, roles):
    roles = set(roles)
    if not roles <= {"bass", "harmony", "melody"}:
        raise ValueError("Unknown diagnostic arrangement role")
    notes = [copy.deepcopy(note) for note in baseline["notes"] if note.get("arrangementRole") not in roles]
    notes.extend(copy.deepcopy(note) for note in ablation["notes"] if note.get("arrangementRole") in roles)
    notes.sort(key=lambda note: (float(note["time"]), int(note["midi"])))
    return {"notes": notes, "researchOnly": True, "deployableCandidate": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Append-only role-coupling report already exists")
    config = load_json(args.config)
    inputs = {}
    for name in ("baseline", "ablation", "reference", "alignment"):
        path = Path(config[name]["path"])
        if sha256_file(path) != config[name]["sha256"]:
            raise ValueError("Frozen role-coupling input changed: " + name)
        inputs[name] = load_json(path)
    reference = prepare_reference_notes({"referenceAlreadyAligned": True}, inputs["reference"], inputs["alignment"])
    ranges = trusted_source_ranges(inputs["alignment"])
    baseline_notes = notes_inside_ranges(normalize_notes(inputs["baseline"]), ranges)
    baseline_metrics = evaluate(reference, baseline_notes)
    if abs(baseline_metrics["exactPitchOnset100ms"]["f1"] - config["knownBaseline100msF1"]) > 1e-6:
        raise ValueError("The existing piano-route score did not reproduce")
    results = []
    for variant in config["variants"]:
        payload = splice_roles(inputs["baseline"], inputs["ablation"], variant["takeRolesFromAblation"])
        observed = notes_inside_ranges(normalize_notes(payload), ranges)
        metrics = evaluate(reference, observed)
        windows = config["openedDevelopmentWindows"]
        inside = lambda time: any(start <= float(time) < end for start, end in windows)
        window_metrics = evaluate([n for n in reference if inside(n["time"])], [n for n in observed if inside(n["time"])])
        keys = [(round(float(note["time"]), 4), int(note["midi"])) for note in payload["notes"]]
        result = {"id": variant["id"], "rolesTakenFromAblation": variant["takeRolesFromAblation"], "fullApprovedMetrics": metrics, "sixOpenedWindowsMetrics": window_metrics, "exactPitchTimeDuplicates": len(keys) - len(set(keys)), "deployableCandidate": False}
        results.append(result)
        print(json.dumps({"id": variant["id"], "100ms": metrics["exactPitchOnset100ms"], "250ms": metrics["exactPitchOnset250ms"], "duration": metrics["duration"], "sixWindow100ms": window_metrics["exactPitchOnset100ms"], "duplicates": result["exactPitchTimeDuplicates"]}), flush=True)
    report = {"schema": "polymath-arrangement-role-coupling-diagnostic-v1", "productionChanged": False, "weightsUpdated": False, "configSha256": sha256_file(args.config), "baselineMetrics": baseline_metrics, "results": results, "interpretation": "Opened WANE diagnostics. Baseline nonselected roles are preserved byte-for-byte per note; ablation roles are spliced without an answer key. This isolates downstream coupling, not a tested inference algorithm. Duration/retrigger checks, other-song transfer, independent references and listening remain mandatory before any production change."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
