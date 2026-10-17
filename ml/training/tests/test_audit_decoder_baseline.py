import tempfile
import unittest
from pathlib import Path

from ml.training.audit_decoder_baseline import compare_events, implementation_hashes


class DecoderBaselineAuditTests(unittest.TestCase):
    def test_physical_event_comparison_ignores_order_and_display_metadata(self):
        first = {"midi": 60, "time": 1.25, "duration": 0.5, "instrument": "voice", "velocity": 0.7}
        second = {**first, "midi": 64}
        comparison = compare_events([first, second], [{**second, "source": "new"}, {**first, "velocity": 0.9}])
        self.assertTrue(comparison["identical"])
        self.assertEqual(comparison["addedEvents"], 0)

    def test_event_comparison_keeps_multiplicity_and_physical_changes(self):
        note = {"midi": 60, "time": 1.25, "duration": 0.5}
        comparison = compare_events([note, note], [note, {**note, "duration": 0.6}])
        self.assertFalse(comparison["identical"])
        self.assertEqual(comparison["addedEvents"], 1)
        self.assertEqual(comparison["removedEvents"], 1)

    def test_implementation_fingerprint_is_independent_of_install_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("one", "two"):
                package = root / name
                package.mkdir()
                (package / "model.py").write_text("same code")
                (package / "cache.bin").write_bytes(b"not-source")
            first = implementation_hashes(root / "one")
            self.assertEqual(first, implementation_hashes(root / "two"))
            self.assertEqual(set(first), {"model.py"})


if __name__ == "__main__":
    unittest.main()
