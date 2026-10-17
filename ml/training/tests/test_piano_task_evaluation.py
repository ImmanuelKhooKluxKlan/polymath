from types import SimpleNamespace

import pytest

from ml.training.evaluate_checkpoint import piano_task_decoder
from ml.training.probe_piano_output_constraint import PianoOutputOnlyProbe
import ml.training.evaluate_checkpoint as checkpoint_evaluation


def test_default_evaluation_does_not_change_decoder():
    model = object()
    assert piano_task_decoder(model, None, False) is model


def test_piano_task_evaluation_requires_output_mask():
    with pytest.raises(ValueError, match="output mask"):
        piano_task_decoder(object(), None, True)


def test_piano_task_evaluation_uses_source_unconditioned_adapter():
    model = SimpleNamespace(_model=object(), _build_conditions=lambda wav, instrument_group=None: None)
    wrapped = piano_task_decoder(model, ("acoustic_piano",), True)
    assert isinstance(wrapped, PianoOutputOnlyProbe)
    assert wrapped.transcription is model
    assert wrapped._model is model._model


@pytest.mark.parametrize("deterministic", [False, True])
def test_broad_route_cli_keeps_output_unmasked_and_optionally_pins_runtime(monkeypatch, tmp_path, deterministic):
    calls = []
    monkeypatch.setattr(checkpoint_evaluation, "configure_piano_task_evaluation_runtime", lambda: calls.append("runtime"))

    def compare(*args, **kwargs):
        calls.append("decode")
        assert kwargs["instruments"] is None
        assert "piano_output_only" not in kwargs
        return {"clips": 1, "candidateMinusBaselineMicroF1": {"100ms": 0}}

    monkeypatch.setattr(checkpoint_evaluation, "compare_checkpoints", compare)
    command = ["evaluate", "--base", "base.safetensors", "--candidate", "candidate.safetensors", "--validation-manifest", "val.jsonl", "--unconditioned", "--out", str(tmp_path / "report.json")]
    if deterministic:
        command.append("--deterministic-runtime")
    monkeypatch.setattr("sys.argv", command)
    checkpoint_evaluation.main()
    assert calls == (["runtime", "decode"] if deterministic else ["decode"])
