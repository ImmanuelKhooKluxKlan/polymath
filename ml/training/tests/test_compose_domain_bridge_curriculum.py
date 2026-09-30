from ml.training.compose_domain_bridge_curriculum import compose, parse_variant


def record(clip_id: str, song_id: str, weight: float = 1.0) -> dict:
    return {
        "clipId": clip_id,
        "songId": song_id,
        "audioClip": f"/runpod-volume/training/test/{clip_id}.wav",
        "exampleWeight": weight,
    }


def test_bridge_variants_share_composition_and_scale_existing_weight():
    train, validation, summary = compose(
        [record("base-a", "target"), record("base-b", "background")],
        [record("valid-a", "held-out")],
        [
            record("p050-a", "target-p050"),
            record("p050-silence", "target-p050", 0.25),
            record("p025-a", "target-p025"),
        ],
        [("target-p050", 0.75)],
        "target",
    )

    assert len(train) == 4
    assert len(validation) == 1
    assert train[-2]["songId"] == "target"
    assert train[-2]["exampleWeight"] == 0.75
    assert train[-1]["exampleWeight"] == 0.1875
    assert summary["trainCompositions"] == 2
    assert summary["variants"][0]["effectiveWeight"] == 0.9375


def test_variant_parser_rejects_out_of_range_weight():
    assert parse_variant("target-p050=0.75") == ("target-p050", 0.75)
    try:
        parse_variant("target-p050=0")
    except ValueError as error:
        assert "within" in str(error)
    else:
        raise AssertionError("Expected zero weight to be rejected")
