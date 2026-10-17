"""Freeze and test a two-pass harmony-only research ablation across songs."""

import argparse
import copy
import json
from pathlib import Path

from .evaluate_piano_arranger import evaluate, load_json, prepare_reference_notes
from .probe_arrangement_role_coupling import splice_roles
from .reference_region_policy import matching_policy
from .search_default_piano_pipeline import clip_candidate
from .train_muscriptor_piano import sha256_file


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def verified(item):
    path = Path(item["path"])
    if sha256_file(path) != item["sha256"]:
        raise ValueError("Frozen harmony-transfer input changed: " + str(path))
    return load_json(path)


def playback_signature(payload):
    result = []
    for note in payload["notes"]:
        duration = float(note.get("duration", .2))
        result.append((int(note["midi"]), float(note["time"]), duration,
                       float(note.get("visualDuration", duration)), float(note.get("audioDuration", duration)),
                       float(note.get("velocity", .72)), float(note.get("performanceGain", 1)),
                       float(note.get("releaseSeconds", 0))))
    return sorted(result)


def gates(baseline, candidate, duplicates):
    checks = {"noDuplicatePitchOnsets": duplicates == 0}
    for tolerance in (100, 250):
        key = f"exactPitchOnset{tolerance}ms"
        for metric in ("f1", "recall"):
            checks[f"{tolerance}ms_{metric}_not_worse"] = candidate[key][metric] >= baseline[key][metric] - 1e-6
    before, after = baseline["physicalDuration"], candidate["physicalDuration"]
    comparable = before["medianAbsoluteErrorSeconds"] is not None and after["medianAbsoluteErrorSeconds"] is not None
    checks["physicalDurationComparable"] = comparable
    checks["physicalDurationErrorNotOver10PercentWorse"] = comparable and after["medianAbsoluteErrorSeconds"] <= before["medianAbsoluteErrorSeconds"] * 1.1 + 1e-6
    checks["physicalCutoffRateNotWorse"] = after["severeCutoffRate"] <= before["severeCutoffRate"] * 1.05 + 0.005
    checks["rapidRetriggersNotOver5PercentWorse"] = candidate["rapidRetriggersUnder100ms"] <= baseline["rapidRetriggersUnder100ms"] * 1.05 + 1e-6
    return checks


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--freeze-plan", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source-audits", type=Path)
    parser.add_argument("--exclude-song", action="append", default=[])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    from server.piano_arranger_pipeline import DEFAULT_ARRANGER_PROFILE, DEFAULT_RECOVERY_PROFILE, DEFAULT_REGISTER_PROFILE, run_pipeline
    if args.freeze_plan:
        if args.plan.exists() or args.manifest is None or args.source_audits is None:
            raise ValueError("Freezing requires new plan, existing manifest and source audit directory")
        rows = []
        for row in load_json(args.manifest)["songs"]:
            if row["id"] in args.exclude_song:
                continue
            proof = args.source_audits / (row["id"] + ".json")
            source = load_json(proof)["inputs"]["source"]
            if matching_policy({"songId": row["id"]}) and (row.get("referenceAlreadyAligned") or row.get("referenceEndSeconds") != 150.0 or row.get("candidateEndSeconds") is None):
                raise ValueError("Restricted transfer requires the native reference cutoff and its separately mapped candidate cutoff")
            pair = {key: value for key, value in row.items() if key not in {"candidate", "baseline"}}
            pair.update({"sourceInput": binding(source), "referenceInput": binding(row["reference"]), "alignmentInput": binding(row["alignment"]), "sourceProof": binding(proof)})
            rows.append(pair)
        plan = {"schema": "polymath-harmony-role-transfer-plan-v1", "productionMutationAllowed": False, "parentManifest": binding(args.manifest), "profiles": [binding(path) for path in (DEFAULT_ARRANGER_PROFILE, DEFAULT_RECOVERY_PROFILE, DEFAULT_REGISTER_PROFILE)], "excludedSongs": args.exclude_song, "songs": rows, "method": "Run existing baseline and bass-thinning-disabled pipeline; take only harmony from ablation and leave baseline melody/bass notes untouched. No title/reference rules inside inference.", "interpretation": "Opened five-song transfer controls, not unseen independently certified ground truth; no new transcription or learned weights. Kiss Me remains pre150 reference time."}
        write_new(args.plan, plan)
        print(json.dumps({"planSha256": sha256_file(args.plan), "songs": [row["id"] for row in rows]}))
        return
    if args.out is None or args.out.exists():
        raise ValueError("Evaluation requires a new output directory")
    plan = load_json(args.plan)
    if plan.get("productionMutationAllowed") is not False or plan.get("schema") != "polymath-harmony-role-transfer-plan-v1":
        raise ValueError("Invalid research-only transfer plan")
    base_profile, recovery, register = [verified(item) for item in plan["profiles"]]
    alternative_profile = copy.deepcopy(base_profile)
    alternative_profile["decoder"]["defaultFullMix"]["bassGestureCompression"]["enabled"] = False
    args.out.mkdir(parents=True)
    results = []
    for row in plan["songs"]:
        source = verified(row["sourceInput"])
        reference_payload, alignment = verified(row["referenceInput"]), verified(row["alignmentInput"])
        verified(row["sourceProof"])
        reference = prepare_reference_notes(row, reference_payload, alignment)
        baseline, _diagnostics = run_pipeline(copy.deepcopy(source), "full", copy.deepcopy(base_profile), copy.deepcopy(recovery), copy.deepcopy(register), exact_song_profiles=[])
        ablation, _diagnostics = run_pipeline(copy.deepcopy(source), "full", copy.deepcopy(alternative_profile), copy.deepcopy(recovery), copy.deepcopy(register), exact_song_profiles=[])
        candidate = splice_roles(baseline, ablation, ["harmony"])
        baseline_metrics = evaluate(reference, clip_candidate(baseline, row, alignment))
        candidate_metrics = evaluate(reference, clip_candidate(candidate, row, alignment))
        keys = [(round(float(n["time"]), 4), int(n["midi"])) for n in candidate["notes"]]
        duplicates = len(keys) - len(set(keys))
        checks = gates(baseline_metrics, candidate_metrics, duplicates)
        for side, payload in (("baseline", baseline), ("harmony-counterfactual", candidate)):
            write_new(args.out / (row["id"] + "-" + side + ".json"), payload)
        result = {"id": row["id"], "referenceEndSeconds": row.get("referenceEndSeconds"), "candidateEndSeconds": row.get("candidateEndSeconds"), "baseline": baseline_metrics, "candidate": candidate_metrics, "gates": checks, "passed": all(checks.values()), "playbackChanged": playback_signature(baseline) != playback_signature(candidate), "noteIdentityPreservedOutsideHarmony": True, "exactPitchTimeDuplicates": duplicates}
        results.append(result)
        print(json.dumps({"id": row["id"], "passed": result["passed"], "playbackChanged": result["playbackChanged"], "baseline100msF1": baseline_metrics["exactPitchOnset100ms"]["f1"], "candidate100msF1": candidate_metrics["exactPitchOnset100ms"]["f1"], "failedChecks": [name for name, passed in checks.items() if not passed]}), flush=True)
    affected = [row for row in results if row["playbackChanged"]]
    all_pass = bool(results) and all(row["passed"] for row in results)
    decision = ("REJECT" if not all_pass else "INCONCLUSIVE_NO_AFFECTED_TRANSFER_SONGS" if not affected else "INCONCLUSIVE_ONLY_ONE_AFFECTED_TRANSFER_SONG" if len(affected) < 2 else "RESEARCH_ONLY_TRANSFER_NONREGRESSION")
    write_new(args.out / "transfer-results.json", {"schema": "polymath-harmony-role-transfer-results-v1", "implementationSha256": sha256_file(Path(__file__)), "planSha256": sha256_file(args.plan), "productionChanged": False, "weightsUpdated": False, "songs": results, "allSongsPass": all_pass, "affectedSongs": [row["id"] for row in affected], "decision": decision, "interpretation": "Research-only two-pass arrangement experiment using existing decoded notes. No-op songs are safety controls, not transfer evidence. This does not demonstrate a better foundation checkpoint or 70/90 percent transcription accuracy. Independent evaluation and listening remain required; inconclusive or failed results keep it disabled."})


if __name__ == "__main__":
    main()
