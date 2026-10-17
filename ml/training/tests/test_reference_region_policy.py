import pytest

from ml.training.filter_reference_regions import mapped_cutoff
from ml.training.reference_region_policy import ReferenceRegionError, audit_reference_regions, make_proof, policies
from ml.training.train_muscriptor_piano import TrainingError, audit_records


def kiss(start=135):
    policy = policies()[0]
    return {"songId": "kiss-me-phase2-reviewed", "clipId": "kiss-test", "sourceStart": start, "durationSeconds": 5, "referenceRegionProof": make_proof(policy, policy["approvedSourceClocks"][0]["sourcePackageSha256"])}


def test_kiss_me_cutoff_is_mapped_to_the_source_clock():
    assert audit_reference_regions([kiss()])["regionRestrictedClipsChecked"] == 1
    with pytest.raises(ReferenceRegionError, match="crosses"):
        audit_reference_regions([kiss(140)])


def test_wane_is_not_accidentally_truncated_at_the_kiss_me_cutoff():
    assert audit_reference_regions([{"songId": "wane-gbt-pianella-supervision", "sourceStart": 185, "durationSeconds": 5}])["regionRestrictedClipsChecked"] == 0


def test_legacy_and_tampered_reference_proofs_fail_before_training():
    row = kiss()
    row.pop("referenceRegionProof")
    with pytest.raises(TrainingError, match="missing mapped reference-region"):
        audit_records([row])
    row = kiss()
    row["referenceRegionProof"]["sourceEndSeconds"] = 200
    with pytest.raises(ReferenceRegionError, match="does not match"):
        audit_reference_regions([row])


def test_reference_time_is_not_confused_with_source_time():
    package = {"alignment": {"anchors": [{"referenceTime": 100, "observedTime": 90}, {"referenceTime": 200, "observedTime": 180}]}}
    assert mapped_cutoff(package, 150) == 135
    with pytest.raises(ValueError, match="inside"):
        mapped_cutoff(package, 250)
