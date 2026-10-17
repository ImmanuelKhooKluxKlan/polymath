import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / "ml/training/configs"


class CoherentHeadOnlyFreezeTests(unittest.TestCase):
    def test_original_audio_weight_experiment_keeps_optimizer_and_gates_fixed(self):
        control = json.loads((CONFIGS / "phase103-v016-coherent-audio-experiment.json").read_text())
        candidate = json.loads((CONFIGS / "phase103-v018-p000-weight050-experiment.json").read_text())
        self.assertEqual(candidate["frozenHyperparameters"], control["frozenHyperparameters"])
        self.assertEqual(candidate["baseCheckpoint"], control["baseCheckpoint"])
        self.assertEqual(candidate["singleChangedVariable"]["changedClips"], 30)
        self.assertFalse(candidate["singleChangedVariable"]["labelsChanged"])
        self.assertFalse(candidate["singleChangedVariable"]["audioChanged"])
        gate = json.loads((CONFIGS / "phase103-v018-p000-weight050-validation-gate.json").read_text())
        old_gate = json.loads((CONFIGS / "phase103-v016-coherent-audio-validation-gate.json").read_text())
        self.assertEqual(gate["requiredChecks"], old_gate["requiredChecks"])
        self.assertEqual(gate["checkpointSelectionChecks"], old_gate["checkpointSelectionChecks"])
        self.assertEqual(gate["immutableInputs"]["trainManifestSha256"], candidate["dataset"]["trainManifestSha256"])
        runner = (ROOT / "scripts/training/run_phase103_v018_p000_weight050.sh").read_text()
        values = dict(re.findall(r'^export ([A-Z0-9_]+)="([^"]*)"$', runner, re.MULTILINE))
        self.assertEqual(values["POLYMATH_CANDIDATE_ID"], candidate["candidateId"])
        self.assertEqual(values["POLYMATH_TRAIN_SHA256"], candidate["dataset"]["trainManifestSha256"])
        self.assertEqual(values["POLYMATH_TRAIN_LAST_LAYERS"], "1")
        self.assertEqual(values["POLYMATH_MAX_OPTIMIZER_STEPS"], "40")
        self.assertIn(candidate["dataset"]["weightReceiptSha256"], runner)
        self.assertIn(candidate["baselineRuntimeAudit"]["sha256"], runner)

    def test_controlled_inputs_and_shared_updates_are_identical(self):
        control = json.loads((CONFIGS / "phase103-v016-coherent-audio-experiment.json").read_text())
        candidate = json.loads((CONFIGS / "phase103-v017-coherent-head-only-experiment.json").read_text())
        self.assertEqual(candidate["baseCheckpoint"], control["baseCheckpoint"])
        for name in ("id", "trainClips", "validationClips", "trainManifestSha256", "validationManifestSha256", "audioRebuildReceiptSha256"):
            self.assertEqual(candidate["dataset"][name], control["dataset"][name])
        for name, value in control["frozenHyperparameters"].items():
            if name not in ("trainLastLayers", "maxOptimizerSteps"):
                self.assertEqual(candidate["frozenHyperparameters"][name], value, name)
        self.assertEqual(candidate["frozenHyperparameters"]["trainLastLayers"], 0)
        self.assertEqual(candidate["budget"]["pairedComparisonSteps"], [5, 10, 15, 20])
        self.assertFalse(candidate["productionMutationAllowed"])

    def test_runner_and_gate_agree_with_frozen_experiment(self):
        experiment = json.loads((CONFIGS / "phase103-v017-coherent-head-only-experiment.json").read_text())
        gate = json.loads((CONFIGS / "phase103-v017-coherent-head-only-validation-gate.json").read_text())
        runner = (ROOT / "scripts/training/run_phase103_v017_coherent_head_only.sh").read_text()
        values = dict(re.findall(r'^export ([A-Z0-9_]+)="([^"]*)"$', runner, re.MULTILINE))
        self.assertEqual(values["POLYMATH_CANDIDATE_ID"], gate["candidateVersion"])
        self.assertEqual(values["POLYMATH_TRAIN_SHA256"], experiment["dataset"]["trainManifestSha256"])
        self.assertEqual(values["POLYMATH_TRAIN_LAST_LAYERS"], "0")
        self.assertEqual(values["POLYMATH_MAX_OPTIMIZER_STEPS"], "20")
        self.assertEqual(values["POLYMATH_FULL_SONG_TARGET_DEVELOPMENT_CONFIG"], experiment["fullSongTargetDevelopmentConfig"])
        for name in ("trainManifestSha256", "validationManifestSha256", "audioRebuildReceiptSha256"):
            self.assertEqual(gate["immutableInputs"][name], experiment["dataset"][name])
        for kind in ("BASE", "CANDIDATE"):
            path = ROOT / values[f"POLYMATH_{kind}_CANARY_MANIFEST"]
            items = json.loads(path.read_text())
            self.assertEqual(items[0]["instruments"], [])
            self.assertIn(experiment["candidateId"], items[0]["output"])


if __name__ == "__main__":
    unittest.main()
