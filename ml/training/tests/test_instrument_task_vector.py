import importlib.util
from types import SimpleNamespace
import unittest

from ml.training.instrument_task_vector import (
    TaskVectorError,
    enable_instrument_task_vector,
    task_vector_delta,
)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch is tested in the GPU environment; no local installation required")
class InstrumentTaskVectorTests(unittest.TestCase):
    def make_model(self):
        import torch

        class Conditioner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(6, 4)

            def tokenize(self, values):
                return torch.tensor([[(-1 if value is None else int(value)) + 1] for value in values])

            def forward(self, tokens):
                embeds = self.embed(tokens + 1)
                return embeds, torch.ones_like(embeds[..., 0])

        model = torch.nn.Module()
        model.condition_provider = torch.nn.Module()
        model.condition_provider.conditioners = torch.nn.ModuleDict({"instrument_group": Conditioner()})
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return SimpleNamespace(_model=model)

    def test_active_row_is_observed_and_not_guessed_from_class_name(self):
        for condition, expected in ((None, 1), ("0", 2)):
            model = self.make_model()
            parameter, hook, metadata = enable_instrument_task_vector(model, condition)
            try:
                self.assertEqual(metadata["activeRow"], expected)
                self.assertEqual(metadata["activeParameters"], 4)
                self.assertTrue(parameter.requires_grad)
            finally:
                hook.remove()

    def test_gradient_mask_and_zero_decay_preserve_every_inactive_row(self):
        import torch
        model = self.make_model()
        parameter, hook, metadata = enable_instrument_task_vector(model, None)
        before = parameter.detach().clone()
        try:
            optimizer = torch.optim.AdamW([parameter], lr=0.01, weight_decay=0.0)
            parameter.sum().backward()
            self.assertEqual(int(torch.count_nonzero(parameter.grad)), 4)
            optimizer.step()
            report = task_vector_delta(parameter, before, metadata["activeRow"])
            self.assertTrue(report["inactiveRowsBitwiseUnchanged"])
            self.assertEqual(report["changedActiveParameters"], 4)
            self.assertGreater(report["rmsDelta"], 0)
        finally:
            hook.remove()

    def test_inactive_row_mutation_is_rejected_instead_of_saved(self):
        import torch
        model = self.make_model()
        parameter, hook, metadata = enable_instrument_task_vector(model, None)
        before = parameter.detach().clone()
        try:
            with torch.no_grad():
                parameter[0, 0] += 0.1
            with self.assertRaisesRegex(TaskVectorError, "inactive"):
                task_vector_delta(parameter, before, metadata["activeRow"])
        finally:
            hook.remove()

    def test_nonfinite_active_gradient_fails_before_an_optimizer_update(self):
        model = self.make_model()
        parameter, hook, _metadata = enable_instrument_task_vector(model, None)
        try:
            with self.assertRaisesRegex(TaskVectorError, "Non-finite"):
                (parameter.sum() * float("nan")).backward()
        finally:
            hook.remove()


if __name__ == "__main__":
    unittest.main()
