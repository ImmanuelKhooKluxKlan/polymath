"""Validate, de-identify, deduplicate, and split reviewed ChatBoss examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROLES = {"support", "teacher"}
MESSAGE_ROLES = {"system", "user", "assistant"}
SENSITIVE_PATTERNS = {
    "email address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "likely API token": re.compile(r"\b(?:sk|rk|ghp|github_pat)_[A-Za-z0-9_-]{16,}\b"),
    "phone number": re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="Reviewed source JSONL; repeat for more files")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--validation-percent", type=int, default=20)
    return parser.parse_args()


def canonical_text(record: dict) -> str:
    return json.dumps(record.get("messages", []), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def validate_record(record: dict, source: str, line_number: int) -> dict:
    location = f"{source}:{line_number}"
    role = str(record.get("role", "")).strip().lower()
    if role not in ROLES:
        raise ValueError(f"{location}: role must be support or teacher")
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError(f"{location}: messages must contain system, user, and assistant turns")
    cleaned = []
    for message in messages:
        message_role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if message_role not in MESSAGE_ROLES or not content:
            raise ValueError(f"{location}: every message needs a valid role and non-empty content")
        for label, pattern in SENSITIVE_PATTERNS.items():
            if pattern.search(content):
                raise ValueError(f"{location}: remove detected {label} before training")
        cleaned.append({"role": message_role, "content": content})
    if cleaned[0]["role"] != "system" or cleaned[-1]["role"] != "assistant":
        raise ValueError(f"{location}: first turn must be system and last turn must be assistant")
    if not any(message["role"] == "user" for message in cleaned):
        raise ValueError(f"{location}: at least one user turn is required")
    return {
        "role": role,
        "prompt": cleaned[:-1],
        "completion": [cleaned[-1]],
        # Qwen3.5's default thinking template leaves an open <think> prefix
        # when prompt and completion are tokenized separately. Disabling
        # thinking makes the prompt tokens an exact prefix of the combined
        # conversation, so completion-only loss masks the correct boundary.
        "chat_template_kwargs": {"enable_thinking": False},
        "metadata": {
            "source": str(record.get("source", "human-reviewed-original"))[:80],
            "scenario": str(record.get("scenario", "general"))[:80],
            "splitGroup": str(record.get("splitGroup", "")).strip()[:80] or None,
        },
    }


def split_bucket(record: dict) -> int:
    """Keep prompt variants from one scenario on the same side of the split."""
    metadata = record.get("metadata", {})
    group = metadata.get("splitGroup")
    if group:
        identity = {
            "role": record.get("role"),
            "source": metadata.get("source"),
            "splitGroup": group,
        }
    else:
        identity = record
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 100


def main() -> None:
    args = parse_args()
    if not 5 <= args.validation_percent <= 40:
        raise SystemExit("--validation-percent must be between 5 and 40")
    records = []
    seen = set()
    rejected_duplicates = 0
    for input_name in args.input:
        source = Path(input_name)
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                digest = hashlib.sha256(canonical_text(raw).encode("utf-8")).hexdigest()
                if digest in seen:
                    rejected_duplicates += 1
                    continue
                seen.add(digest)
                records.append(validate_record(raw, str(source), line_number))
    if len(records) < 10:
        raise SystemExit("At least 10 reviewed examples are required to build a split.")

    train, validation = [], []
    by_role = Counter()
    for record in records:
        by_role[record["role"]] += 1
        bucket = split_bucket(record)
        (validation if bucket < args.validation_percent else train).append(record)

    # A role missing from validation makes role-specific regressions invisible.
    for role in ROLES:
        if not any(item["role"] == role for item in validation):
            candidate = next((item for item in train if item["role"] == role), None)
            if candidate:
                train.remove(candidate)
                validation.append(candidate)
    if not train or not validation:
        raise SystemExit("The deterministic split produced an empty train or validation set.")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("validation", validation)):
        with (output / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "schema": "polymath-chatboss-prompt-completion-v1",
        "sourceExamples": len(records),
        "duplicatesRemoved": rejected_duplicates,
        "trainExamples": len(train),
        "validationExamples": len(validation),
        "roles": dict(by_role),
        "groupedSplit": True,
        "inputs": [str(Path(name)) for name in args.input],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
