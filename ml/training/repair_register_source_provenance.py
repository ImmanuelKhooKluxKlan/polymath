"""Recover true source-pitch features through verified immutable event indices.

Only research metadata changes. Played pitches, starts, durations, dynamics,
hands and roles must stay identical. No answer-key notes enter this operation.
"""

import argparse
from collections import Counter
import copy
import json
from pathlib import Path

from ml.training.train_muscriptor_piano import sha256_file


PLAYBACK_FIELDS = ("midi", "time", "duration", "velocity", "audioDuration", "visualDuration", "performanceGain", "hand", "arrangementRole", "sourceInstrument")


def playback_values(note):
    return tuple(note.get(name) for name in PLAYBACK_FIELDS)


def attach_source_indices(candidate, indexed_candidate):
    old, indexed = candidate["notes"], indexed_candidate["notes"]
    if len(old) != len(indexed) or any(playback_values(a) != playback_values(b) for a, b in zip(old, indexed, strict=True)):
        raise ValueError("Index-provenance replay does not reproduce the same played notes")
    result = copy.deepcopy(candidate)
    for note, verified in zip(result["notes"], indexed, strict=True):
        note["sourceIndex"] = int(verified["sourceIndex"])
    return result


def repair(candidate, source):
    indexed = {}
    for position, note in enumerate(source["notes"]):
        index = int(note.get("sourceIndex", position))
        if index in indexed:
            raise ValueError("Ambiguous duplicate sourceIndex in source transcription")
        indexed[index] = note
    output = copy.deepcopy(candidate)
    changed, roles = 0, Counter()
    for note in output["notes"]:
        try:
            original = indexed[int(note["sourceIndex"])]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Arranged note has no verified source-index match") from exc
        if note.get("sourceInstrument") != original.get("instrument") or int(note["midi"]) % 12 != int(original["midi"]) % 12:
            raise ValueError("Source-index match contradicts source instrument or pitch class")
        inferred = int(note.get("sourceMidiBeforeArrangement", note.get("originalMidiBeforeRangeShift", note["midi"])))
        true_source = int(original["midi"])
        if inferred != true_source:
            changed += 1
            roles[str(note.get("arrangementRole") or "unknown")] += 1
        note["sourceMidiBeforeArrangement"] = true_source
    if [playback_values(note) for note in output["notes"]] != [playback_values(note) for note in candidate["notes"]]:
        raise ValueError("Research feature repair changed played note values")
    return output, {"notes": len(output["notes"]), "sourceEvents": len(indexed), "changedInferredSourcePitchFeatures": changed, "changedByRole": dict(roles), "missingSourceMatches": 0, "instrumentOrPitchClassContradictions": 0, "playedNoteValuesUnchanged": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Feature-repair artifact directory already exists")
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    manifest_path = Path(config["parentManifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    records, repaired_rows = [], []
    args.out.mkdir(parents=True)
    for original_row in manifest["songs"]:
        row = copy.deepcopy(original_row)
        song_id = row["id"]
        override = (config.get("songOverrides") or {}).get(song_id)
        if override:
            for field in ("candidate", "reference", "alignment"):
                row[field] = override[field]
            source_path = Path(override["source"])
            candidate_path = Path(row["candidate"])
            candidate = json.loads(candidate_path.read_text(encoding="utf-8-sig"))
            index_copy_path = candidate_path
        else:
            proof_path = Path(config["sourceProofDirectory"]) / f"{song_id}.json"
            proof = json.loads(proof_path.read_text(encoding="utf-8-sig"))
            source_path = Path(proof["inputs"]["source"])
            candidate_path = Path(row["candidate"])
            index_copy_path = Path(proof["inputs"]["candidate"])
            candidate = attach_source_indices(json.loads(candidate_path.read_text(encoding="utf-8-sig")), json.loads(index_copy_path.read_text(encoding="utf-8-sig")))
        source = json.loads(source_path.read_text(encoding="utf-8-sig"))
        output, audit = repair(candidate, source)
        inputs = (
            ("source", source_path),
            ("candidate", candidate_path),
            ("indexReplay", index_copy_path),
            ("reference", Path(row["reference"])),
            ("alignment", Path(row["alignment"])),
        )
        integrity = {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in inputs}
        output["registerSourceFeatureProvenance"] = {"method": "exact sourceIndex with instrument and pitch-class verification", "source": integrity["source"], "playedNotesChanged": False}
        destination = args.out / f"{song_id}.json"
        destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        row["candidate"] = str(destination)
        repaired_rows.append(row)
        records.append({"id": song_id, **audit, "integrity": integrity, "outputSha256": sha256_file(destination)})
    repaired_manifest = {**manifest, "songs": repaired_rows, "sourceFeatureRepair": "Verified original model-event MIDI; no played-note values changed."}
    output_manifest = args.out / "manifest.json"
    output_manifest.write_text(json.dumps(repaired_manifest, indent=2) + "\n", encoding="utf-8")
    receipt = {"schema": "polymath-register-source-feature-repair-receipt-v1", "parentManifestSha256": sha256_file(manifest_path), "configSha256": sha256_file(args.config), "manifestSha256": sha256_file(output_manifest), "referenceUsedForRepair": False, "playedNotesChanged": False, "productionChanged": False, "songs": records}
    (args.out / "repair-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(output_manifest), "sha256": receipt["manifestSha256"], "repairs": [{"song": row["id"], "changedSourceFeatures": row["changedInferredSourcePitchFeatures"], "notes": row["notes"]} for row in records]}, indent=2))


if __name__ == "__main__":
    main()
