"""Create an append-only dataset copy excluding disallowed reference passages."""

import argparse
import bisect
import copy
import json
from pathlib import Path
import shutil

from ml.training.reference_region_policy import audit_reference_regions, clock_for_package, make_proof, matching_policy, policies
from ml.training.train_muscriptor_piano import read_jsonl, sha256_file


def mapped_cutoff(package, reference_time):
    anchors = sorted(package["alignment"]["anchors"], key=lambda row: float(row["referenceTime"]))
    xs = [float(row["referenceTime"]) for row in anchors]
    ys = [float(row["observedTime"]) for row in anchors]
    if len(xs) < 2 or any(b <= a for a, b in zip(xs, xs[1:])) or any(b < a for a, b in zip(ys, ys[1:])):
        raise ValueError("Cutoff mapping requires a monotonic alignment")
    right = bisect.bisect_left(xs, reference_time)
    if right == 0 or right >= len(xs):
        raise ValueError("Reference cutoff must lie inside the verified anchors")
    left = right - 1
    fraction = (reference_time - xs[left]) / (xs[right] - xs[left])
    return ys[left] + fraction * (ys[right] - ys[left])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--weight-receipt", type=Path, required=True)
    parser.add_argument("--audio-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Filtered dataset destination already exists")
    for path in (args.train_manifest, args.validation_manifest, args.source_package, args.weight_receipt, args.audio_receipt):
        if not path.is_file():
            raise FileNotFoundError(f"Dataset input is missing before any output is written: {path}")
    source_hash = sha256_file(args.source_package)
    package = json.loads(args.source_package.read_text(encoding="utf-8-sig"))
    registry = policies()
    rows = read_jsonl(args.train_manifest)
    kept, removed = [], []
    for original in rows:
        row = copy.deepcopy(original)
        policy = matching_policy(row, registry)
        if policy is not None:
            clock = clock_for_package(policy, source_hash)
            cutoff = mapped_cutoff(package, float(policy["referenceEndSeconds"]))
            if abs(cutoff - float(clock["sourceEndSeconds"])) > 1e-7:
                raise ValueError("The policy's source-clock cutoff disagrees with the verified anchors")
            if float(row["sourceStart"]) + float(row["durationSeconds"]) > cutoff + 1e-8:
                removed.append({"clipId": row["clipId"], "songId": row["songId"], "sourceStart": row["sourceStart"], "durationSeconds": row["durationSeconds"], "policyId": policy["id"]})
                continue
            row["referenceRegionProof"] = make_proof(policy, source_hash)
        before_comparison = copy.deepcopy(row)
        before_comparison.pop("referenceRegionProof", None)
        original_comparison = copy.deepcopy(original)
        original_comparison.pop("referenceRegionProof", None)
        if before_comparison != original_comparison:
            raise ValueError("Reference filtering altered notes, audio, weights or another original field")
        kept.append(row)
    audit = audit_reference_regions(kept)
    args.out.mkdir(parents=True)
    train_out, val_out = args.out / "prepared-train.jsonl", args.out / "prepared-validation.jsonl"
    train_out.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in kept), encoding="utf-8")
    shutil.copyfile(args.validation_manifest, val_out)
    for source, name in ((args.weight_receipt, "variant-weight-ablation-receipt.json"), (args.audio_receipt, "audio-rebuild-receipt.json"), (args.source_package, "reference-clock-source-package.json")):
        shutil.copyfile(source, args.out / name)
    receipt = {"schema": "polymath-reference-region-filter-receipt-v1", "parentTrainManifestSha256": sha256_file(args.train_manifest), "trainManifestSha256": sha256_file(train_out), "validationManifestSha256": sha256_file(val_out), "sourcePackageSha256": source_hash, "trainClipsBefore": len(rows), "trainClipsAfter": len(kept), "removedClips": removed, "remainingRowsNotesAudioWeightsOrderUnchanged": True, "onlyAddedMetadata": "referenceRegionProof", "validationUnchanged": True, "regionAudit": audit, "waneRowsBefore": sum(str(row.get("songId", "")).startswith("wane") for row in rows), "waneRowsAfter": sum(str(row.get("songId", "")).startswith("wane") for row in kept)}
    path = args.out / "reference-region-filter-receipt.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), "trainClips": len(kept), "removedClips": len(removed), "trainManifestSha256": receipt["trainManifestSha256"], "filterReceiptSha256": sha256_file(path), "audit": audit, "waneRowsUnchanged": receipt["waneRowsBefore"] == receipt["waneRowsAfter"]}, indent=2))


if __name__ == "__main__":
    main()
