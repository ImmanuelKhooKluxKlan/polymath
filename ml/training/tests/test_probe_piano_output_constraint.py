import pytest

from ml.training.probe_piano_output_constraint import PianoOutputOnlyProbe


class FakeTranscription:
    def __init__(self, fail=False):
        self.conditions = []
        self.constraints = []
        self.fail = fail

    def _build_conditions(self, wav, instrument_group=None):
        self.conditions.append(instrument_group)
        return wav

    def transcribe(self, audio, *, instruments):
        self.constraints.append(instruments)
        self._build_conditions(audio, "0")
        yield "first"
        if self.fail:
            raise RuntimeError("decode failed")
        yield "last"


def test_probe_keeps_output_mask_but_unsets_only_source_class_condition():
    model = FakeTranscription()
    events = list(PianoOutputOnlyProbe(model).transcribe("audio", instruments=["acoustic_piano"]))
    assert events == ["first", "last"]
    assert model.conditions == [None]
    assert model.constraints == [["acoustic_piano"]]
    assert "_build_conditions" not in vars(model)
    model._build_conditions("audio", "voice")
    assert model.conditions[-1] == "voice"


def test_probe_restores_conditioning_on_failure_and_early_generator_close():
    model = FakeTranscription(fail=True)
    with pytest.raises(RuntimeError, match="decode failed"):
        list(PianoOutputOnlyProbe(model).transcribe("audio", instruments=["acoustic_piano"]))
    assert "_build_conditions" not in vars(model)
    generator = PianoOutputOnlyProbe(model).transcribe("audio", instruments=["acoustic_piano"])
    next(generator)
    generator.close()
    assert "_build_conditions" not in vars(model)


def test_probe_rejects_unsupported_output_constraints_before_modifying_model():
    model = FakeTranscription()
    with pytest.raises(ValueError, match="output mask"):
        list(PianoOutputOnlyProbe(model).transcribe("audio", instruments=["voice"]))
    assert "_build_conditions" not in vars(model)


def test_probe_restores_an_existing_instance_override_exactly():
    model = FakeTranscription()
    original = lambda wav, instrument_group=None: model.conditions.append(instrument_group)
    model._build_conditions = original
    list(PianoOutputOnlyProbe(model).transcribe("audio", instruments=["acoustic_piano"]))
    assert model._build_conditions is original
    assert model.conditions == [None]
