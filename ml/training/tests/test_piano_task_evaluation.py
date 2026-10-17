from types import SimpleNamespace

import pytest

from ml.training.evaluate_checkpoint import piano_task_decoder
from ml.training.probe_piano_output_constraint import PianoOutputOnlyProbe


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
