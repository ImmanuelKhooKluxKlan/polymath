"""Enforce owner-approved reference regions in their correctly mapped clock."""

import hashlib
import json
import math
from pathlib import Path


REGISTRY = Path(__file__).parent / "configs/reference-region-policies-v001.json"


class ReferenceRegionError(ValueError):
    pass


def policies():
    data = json.loads(REGISTRY.read_text(encoding="utf-8-sig"))
    if data.get("schema") != "polymath-reference-region-policies-v1":
        raise ReferenceRegionError("Unknown reference-region policy registry")
    return data["policies"]


def digest(policy):
    return hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def matching_policy(record, registry=None):
    song_id = str(record.get("compositionId") or record.get("songId") or "").lower().replace("_", "-")
    for policy in policies() if registry is None else registry:
        if any(song_id.startswith(prefix) for prefix in policy["songIdPrefixes"]):
            return policy
    return None


def clock_for_package(policy, package_hash):
    for clock in policy["approvedSourceClocks"]:
        if clock["sourcePackageSha256"] == package_hash:
            return clock
    raise ReferenceRegionError("The reference cutoff has not been mapped for this source package")


def make_proof(policy, package_hash):
    clock = clock_for_package(policy, package_hash)
    return {"schema": "polymath-reference-region-clip-proof-v1", "policyId": policy["id"], "policySha256": digest(policy), "sourcePackageSha256": package_hash, "referenceEndSeconds": policy["referenceEndSeconds"], "sourceEndSeconds": clock["sourceEndSeconds"]}


def audit_reference_regions(records):
    registry = policies()
    count = 0
    for record in records:
        policy = matching_policy(record, registry)
        if policy is None:
            continue
        clip_id = record.get("clipId", "unknown")
        proof = record.get("referenceRegionProof")
        if not isinstance(proof, dict):
            raise ReferenceRegionError(f"{clip_id}: missing mapped reference-region proof for {policy['id']}")
        expected = make_proof(policy, proof.get("sourcePackageSha256"))
        if proof != expected:
            raise ReferenceRegionError(f"{clip_id}: reference-region proof does not match the frozen policy/source clock")
        start = float(record["sourceStart"])
        end = start + float(record["durationSeconds"])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end > float(expected["sourceEndSeconds"]) + 1e-8:
            raise ReferenceRegionError(f"{clip_id}: training clip crosses the approved reference cutoff")
        count += 1
    return {"regionRestrictedClipsChecked": count, "violations": 0, "otherCompositionsUnchanged": True}
