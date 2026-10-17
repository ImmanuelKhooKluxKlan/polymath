"""Re-render selected audio variants while preserving labels and row order.

Use this after changing the note source from which synthetic curriculum audio
was rendered. Existing clips must not keep an obsolete synthetic answer while
their labels teach a different answer. Unselected examples remain identical.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Any

from ml.training.add_original_domain_supervision import read_jsonl, sha256, write_jsonl
from ml.training.prepare_audio_clips import prepare_manifest


def rebuild_rows(
    rows: list[dict[str, Any]],
    variants: dict[str, str],
    *,
    label_source_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not variants or any(not prefix or not path for prefix, path in variants.items()):
        raise ValueError("Each variant needs a clip prefix and source-media path")
    if len(label_source_sha256) != 64:
        raise ValueError("A SHA-256 digest for the rendered label source is required")
    counts = dict.fromkeys(variants, 0)
    selected: list[dict[str, Any]] = []
    for row in rows:
        matching = [prefix for prefix in variants if str(row.get("clipId", "")).startswith(prefix)]
        if len(matching) > 1:
            raise ValueError(f"Ambiguous audio variants for {row.get('clipId')}")
        if not matching:
            continue
        prefix = matching[0]
        actual = (row.get("relabeling") or {}).get("replacementSupervisionSha256")
        if actual != label_source_sha256:
            raise ValueError(f"{row.get('clipId')}: label source does not match rendered audio source")
        record = copy.deepcopy(row)
        record["sourceMedia"] = variants[prefix]
        record.pop("audioClip", None)
        record.pop("localAudioSource", None)
        record["audioLabelCoherence"] = {
            "schema": "polymath-curriculum-audio-label-coherence-v1",
            "labelSourceSha256": label_source_sha256,
            "variantPrefix": prefix,
            "replacedAudioClip": row.get("audioClip"),
            "labelsChanged": False,
        }
        selected.append(record)
        counts[prefix] += 1
    if any(count == 0 for count in counts.values()):
        raise ValueError("Every requested variant must match training examples")
    selected_ids = [str(row["clipId"]) for row in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Selected training examples contain duplicate clip IDs")
    return selected, counts


def merge_rendered_rows(
    original: list[dict[str, Any]], rendered: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    replacements = {str(row["clipId"]): row for row in rendered}
    if len(replacements) != len(rendered):
        raise ValueError("Rendered examples contain duplicate clip IDs")
    ids = [str(row.get("clipId", "")) for row in original]
    if len(ids) != len(set(ids)):
        raise ValueError("Source manifest contains duplicate clip IDs")
    if not set(replacements).issubset(ids):
        raise ValueError("Rendered examples contain unknown clip IDs")
    output: list[dict[str, Any]] = []
    for row in original:
        replacement = replacements.get(str(row["clipId"]))
        if replacement is not None:
            for key in ("notes", "exampleWeight", "sourceStart", "durationSeconds", "songId", "split"):
                if replacement.get(key) != row.get(key):
                    raise ValueError(f"{row['clipId']}: re-render changed frozen field {key}")
        output.append(copy.deepcopy(replacement if replacement is not None else row))
    return output


def parse_variant(value: str) -> tuple[str, str]:
    prefix, separator, source = value.partition("=")
    if not separator or not prefix.strip() or not source.strip():
        raise argparse.ArgumentTypeError("Variant must be CLIP_PREFIX=SOURCE_MEDIA")
    return prefix.strip(), source.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--label-source", type=Path, required=True)
    parser.add_argument("--variant", type=parse_variant, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runtime-root")
    parser.add_argument("--ffmpeg", type=Path, required=True)
    args = parser.parse_args()
    variants = dict(args.variant)
    if len(variants) != len(args.variant):
        raise ValueError("Variant prefixes must be unique")
    for source in variants.values():
        if not Path(source).is_file():
            raise FileNotFoundError(source)
    train_path = args.train_manifest.resolve()
    validation_path = args.validation_manifest.resolve()
    label_path = args.label_source.resolve()
    original = read_jsonl(train_path)
    selected, counts = rebuild_rows(original, variants, label_source_sha256=sha256(label_path))
    destination = args.out.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    selected_path = destination / "coherent-audio.jsonl"
    write_jsonl(selected_path, selected)
    prepared = prepare_manifest(selected_path, destination, args.ffmpeg.resolve(), manifest_audio_root=args.runtime_root)
    output = merge_rendered_rows(original, read_jsonl(Path(prepared["preparedManifest"])))
    train_output = destination / "prepared-train.jsonl"
    validation_output = destination / "prepared-validation.jsonl"
    write_jsonl(train_output, output)
    shutil.copyfile(validation_path, validation_output)
    receipt = {
        "schema": "polymath-curriculum-audio-rebuild-receipt-v1",
        "sourceTrainManifest": {"path": str(train_path), "sha256": sha256(train_path)},
        "labelSource": {"path": str(label_path), "sha256": sha256(label_path)},
        "variants": {prefix: {"sourceMedia": source, "sha256": sha256(Path(source)), "clips": counts[prefix]} for prefix, source in variants.items()},
        "trainClips": len(output),
        "rebuiltClips": len(selected),
        "recordOrderPreserved": True,
        "labelsChanged": False,
        "weightsChanged": False,
        "outputs": {
            "trainManifest": {"path": str(train_output), "sha256": sha256(train_output)},
            "validationManifest": {"path": str(validation_output), "sha256": sha256(validation_output)},
        },
    }
    (destination / "audio-rebuild-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
