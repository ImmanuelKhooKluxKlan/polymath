import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]


class SourceAwareGateRunnerTests(unittest.TestCase):
    def test_inline_gate_rejects_failed_empty_or_nonboolean_checks(self):
        runner = (ROOT / "scripts/training/run_phase103_v007_visual_labels.sh").read_text()
        section = runner.split("echo PHASE103_STAGE_SOURCE_AWARE_PIANO_PROCESSING_GATE", 1)[1]
        code = section.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        cases = [
            ({"decision": "PROMOTE", "promotionGates": {"check": True}}, 0),
            ({"decision": "REJECT", "promotionGates": {"check": True}}, 1),
            ({"decision": "PROMOTE", "promotionGates": {"check": False}}, 1),
            ({"decision": "PROMOTE", "promotionGates": {}}, 1),
            ({"decision": "PROMOTE", "promotionGates": {"check": "true"}}, 1),
        ]
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "evaluation.json"
            for payload, expected in cases:
                report.write_text(json.dumps(payload))
                result = subprocess.run([sys.executable, "-", str(report)], input=code, text=True, capture_output=True)
                self.assertEqual(result.returncode, expected, payload)

    def test_both_cleanup_replays_receive_hash_checked_source_audio(self):
        script = (ROOT / "scripts/training/evaluate_phase103_wane_production_route.sh").read_text()
        self.assertEqual(script.count('--mode full --audio "$SOURCE_AUDIO"'), 2)
        self.assertIn('"$SOURCE_AUDIO_SHA256" "$SOURCE_AUDIO" | sha256sum --check --strict -', script)


if __name__ == "__main__":
    unittest.main()
