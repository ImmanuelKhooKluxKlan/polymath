import pytest

from ml.training.run_piano_task_vector_experiment import adaptation_flags, audit_separation, validate_preflight


def row(start=10, song="train-song", clip="wane-test"):
    return {"songId": song, "clipId": clip, "sourceStart": start, "durationSeconds": 5}


def test_wane_windows_are_audited_in_every_audio_variant():
    result = audit_separation([row()], [row(song="validation-song", clip="validation")], [{"start": 15, "end": 20}])
    assert result["overlappingTimeWindows"] == 0
    assert result["waneClipsChecked"] == 1
    with pytest.raises(ValueError, match="overlaps a held-out"):
        audit_separation([row(12.5)], [], [{"start": 15, "end": 20}])


def test_composition_overlap_is_rejected_even_with_different_clip_ids():
    with pytest.raises(ValueError, match="compositions overlap"):
        audit_separation([row()], [row(20, clip="different")], [])


def attention_fixture():
    config = {"trainingMode": "attention-lora", "baseCheckpoint": {"sha256": "base"}, "dataset": {"trainManifestSha256": "train"}, "frozenHyperparameters": {"attentionLoRALayers": 1, "attentionLoRARank": 8, "attentionLoRAAlpha": 8}, "gradientPreflight": {"maximumPeakGpuFraction": .9}}
    preflight = {"verdict": "PASS", "optimizerCreated": False, "weightsUpdated": False, "baseCheckpointSha256": "base", "trainManifestSha256": "train", "otherFoundationGradientsAbsent": True, "adapter": {"lastLayers": 1, "rank": 8, "alpha": 8}, "fullSongReproduction": {"identical": True}, "panelReproduction": {str(i): {"identical": True} for i in range(18)}, "zeroUpdateMatrixDeltas": [{"changedEffectiveParameters": 0}] * 2, "gradientEvidence": [{"BGradientNorm": .1, "AInitialGradientExactlyZero": True}] * 2, "peakGradientGpuFraction": .5}
    return config, preflight


def test_attention_training_never_inherits_task_vector_or_piano_mask_flags():
    config, _ = attention_fixture()
    training, evaluation = adaptation_flags(config)
    assert "--attention-lora-layers" in training
    assert "--train-piano-task-vector" not in training
    assert "--decoder-piano-output-only" not in training
    assert evaluation == ["--unconditioned", "--deterministic-runtime"]
    with pytest.raises(ValueError, match="Unknown"):
        adaptation_flags({"trainingMode": "typo"})


def test_attention_preflight_requires_complete_zero_update_reproduction():
    config, preflight = attention_fixture()
    validate_preflight(preflight, config)
    preflight["panelReproduction"]["0"]["identical"] = False
    with pytest.raises(ValueError, match="fidelity"):
        validate_preflight(preflight, config)
    preflight["panelReproduction"]["0"]["identical"] = True
    preflight["zeroUpdateMatrixDeltas"] = []
    with pytest.raises(ValueError, match="fidelity"):
        validate_preflight(preflight, config)


def test_attention_preflight_rejects_different_shapes_or_an_optimizer_update():
    config, preflight = attention_fixture()
    preflight["adapter"]["rank"] = 4
    with pytest.raises(ValueError, match="fidelity"):
        validate_preflight(preflight, config)
    preflight["adapter"]["rank"] = 8
    preflight["weightsUpdated"] = True
    with pytest.raises(ValueError, match="frozen-base"):
        validate_preflight(preflight, config)
