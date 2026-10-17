"""Run a frozen piano task-vector or attention-adapter experiment; never deploy.

All optimizer work is gated by an independently saved gradient/memory preflight.
The final comparison also checks the existing broad-route piano output, not
only the substantially weaker piano-output-mask baseline.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from ml.training.full_song_target_development import load_config
from ml.training.train_muscriptor_piano import audit_records, read_jsonl, sha256_file


BASE = Path("/runpod-volume/models/muscriptor-tester/phase46-v017/model.safetensors")
RUNTIME_AUDIT = Path("/runpod-volume/training/phase-103-decoder-baseline-audit-v001/audit.json")
RUNTIME_AUDIT_HASH = "d558cab2304a4f2c8122945f44cba5c5969e2b1dca4846582005c1c268beee81"


def verify(path: Path, expected: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"Immutable experiment input changed: {path}")


def audit_separation(train: list[dict], validation: list[dict], windows: list[dict]) -> dict:
    train_songs = {str(row["songId"]) for row in train}
    validation_songs = {str(row["songId"]) for row in validation}
    if train_songs & validation_songs:
        raise ValueError("Training and validation compositions overlap")
    wane = [row for row in train if str(row["clipId"]).startswith("wane-")]
    for row in wane:
        start = float(row["sourceStart"])
        end = start + float(row["durationSeconds"])
        if any(start < float(window["end"]) - 1e-8 and end > float(window["start"]) + 1e-8 for window in windows):
            raise ValueError(f"WANE training audio overlaps a held-out time window: {row['clipId']}")
    return {"trainingCompositions": len(train_songs), "validationCompositions": len(validation_songs), "waneClipsChecked": len(wane), "overlappingTimeWindows": 0, "note": "Opened development windows, not an untouched song."}


def run_stage(stage: str, command: list[str]) -> None:
    print(json.dumps({"stage": stage, "command": command}), flush=True)
    subprocess.run(command, check=True)


def adaptation_flags(config: dict) -> tuple[list[str], list[str]]:
    mode = config.get("trainingMode", "task-vector")
    if mode == "attention-lora":
        hyper = config["frozenHyperparameters"]
        return (["--attention-lora-layers", str(hyper["attentionLoRALayers"]), "--attention-lora-rank", str(hyper["attentionLoRARank"]), "--attention-lora-alpha", str(hyper["attentionLoRAAlpha"])], ["--unconditioned", "--deterministic-runtime"])
    if mode != "task-vector":
        raise ValueError("Unknown frozen adaptation mode")
    return (["--decoder-piano-output-only", "--train-piano-task-vector", "--piano-task-vector-learning-rate", str(config["frozenHyperparameters"]["taskVectorLearningRate"])], ["--piano-output-only"])


def validate_preflight(preflight: dict, config: dict) -> None:
    if not (preflight["verdict"] == "PASS" and preflight["optimizerCreated"] is False and preflight["weightsUpdated"] is False and preflight["baseCheckpointSha256"] == config["baseCheckpoint"]["sha256"] and preflight["trainManifestSha256"] == config["dataset"]["trainManifestSha256"] and preflight["otherFoundationGradientsAbsent"] is True):
        raise ValueError("Preflight did not prove frozen-base training feasibility")
    if config.get("trainingMode", "task-vector") == "attention-lora":
        hyper = config["frozenHyperparameters"]
        adapter = preflight["adapter"]
        expected_matrices = 2 * int(hyper["attentionLoRALayers"])
        if not (adapter["lastLayers"] == hyper["attentionLoRALayers"] and adapter["rank"] == hyper["attentionLoRARank"] and adapter["alpha"] == hyper["attentionLoRAAlpha"] and preflight["fullSongReproduction"]["identical"] is True and len(preflight["panelReproduction"]) == 18 and all(row["identical"] is True for row in preflight["panelReproduction"].values()) and len(preflight["zeroUpdateMatrixDeltas"]) == expected_matrices and all(row["changedEffectiveParameters"] == 0 for row in preflight["zeroUpdateMatrixDeltas"]) and len(preflight["gradientEvidence"]) == expected_matrices and all(row["BGradientNorm"] > 0 and row["AInitialGradientExactlyZero"] is True for row in preflight["gradientEvidence"]) and preflight["peakGradientGpuFraction"] < config["gradientPreflight"]["maximumPeakGpuFraction"]):
            raise ValueError("Attention-adapter fidelity/memory preflight did not pass")
    elif not (preflight["inactiveGradientRowsZero"] is True and preflight["taskVectorGradientNorm"] > 0 and preflight["peakGpuFraction"] < config["gradientPreflight"]["maximumPeakGpuFraction"]):
        raise ValueError("Task-vector gradient/memory preflight did not pass")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-commit", required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--experiment", type=Path, default=Path("ml/training/configs/phase105-v001-piano-task-vector-experiment.json"))
    parser.add_argument("--gate", type=Path, default=Path("ml/training/configs/phase105-v001-piano-task-vector-validation-gate.json"))
    args = parser.parse_args()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)
    actual_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != args.repo_commit:
        raise ValueError("The checked-out implementation is not the frozen experiment commit")
    config_path = (repo / args.experiment).resolve()
    gate_path = (repo / args.gate).resolve()
    config = json.loads(config_path.read_text())
    training_adaptation_flags, evaluation_route_flags = adaptation_flags(config)
    gate_config = json.loads(gate_path.read_text())
    if gate_config["candidateVersion"] != config["candidateId"]:
        raise ValueError("Gate and experiment candidate identities differ")
    candidate_id, dataset_id = config["candidateId"], config["dataset"]["id"]
    if not all(re.fullmatch(r"[a-zA-Z0-9-]+", name) for name in (candidate_id, dataset_id)):
        raise ValueError("Invalid experiment identifier")
    data = Path("/runpod-volume/training") / dataset_id
    output = Path("/runpod-volume/models/muscriptor-tester") / candidate_id
    status = data / f"{candidate_id}-status.txt"
    receipt = data / f"{candidate_id}-rejection.json"
    result_path = data / f"{candidate_id}-runner-result.json"
    for path in (output, status, receipt, result_path):
        if path.exists():
            raise FileExistsError(f"Append-only experiment target already exists: {path}")
    import fcntl
    with (data.parent / "polymath-research-gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status.write_text("RUNNING\n")
        result = {"schema": "polymath-piano-task-experiment-runner-result-v1", "candidateId": candidate_id, "repoCommit": actual_commit, "experimentConfigSha256": sha256_file(config_path), "gateConfigSha256": sha256_file(gate_path), "productionPromotionPerformed": False}
        code = 1
        try:
            train_path, val_path = data / "prepared-train.jsonl", data / "prepared-validation.jsonl"
            preflight_path = Path(config.get("preflightPath", data / "task-vector-gradient-preflight.json"))
            for path, expected in ((BASE, config["baseCheckpoint"]["sha256"]), (train_path, config["dataset"]["trainManifestSha256"]), (val_path, config["dataset"]["validationManifestSha256"]), (data / "variant-weight-ablation-receipt.json", config["dataset"]["weightReceiptSha256"]), (data / "audio-rebuild-receipt.json", config["dataset"]["coherentAudioReceiptSha256"]), (RUNTIME_AUDIT, RUNTIME_AUDIT_HASH), (preflight_path, args.preflight_sha256)):
                verify(path, expected)
            preflight = json.loads(preflight_path.read_text())
            validate_preflight(preflight, config)
            import muscriptor
            import torch
            from ml.training.audit_decoder_baseline import implementation_hashes
            runtime_audit = json.loads(RUNTIME_AUDIT.read_text())
            if runtime_audit["verdict"] != "CACHE_EVENT_REPRODUCTION_PASSED" or implementation_hashes(Path(muscriptor.__file__).parent) != runtime_audit["runtime"]["muscriptorPythonFiles"] or str(torch.__version__) != runtime_audit["runtime"]["torch"] or torch.cuda.get_device_name(0) != runtime_audit["runtime"]["gpu"]:
                raise ValueError("Verified decoding runtime changed")
            full_config_path = repo / config["fullSongTargetConfig"]
            full_config = load_config(full_config_path)
            from ml.training.full_song_target_development import baseline_score
            observed_baseline = baseline_score(full_config)
            for tolerance in ("100ms", "250ms"):
                for metric in ("matchedNotes", "precision", "recall", "microF1"):
                    if abs(observed_baseline[tolerance][metric] - full_config["knownBaseline"][tolerance][metric]) > 1e-6:
                        raise ValueError("The frozen full-song development baseline score changed")
            broad_raw = Path(config["existingBroadRaw"]["path"])
            verify(broad_raw, config["existingBroadRaw"]["sha256"])
            train, validation = read_jsonl(train_path), read_jsonl(val_path)
            audits = {"train": audit_records(train), "validation": audit_records(validation), "separation": audit_separation(train, validation, full_config["windows"])}
            if len(train) != config["dataset"]["trainClips"] or len(validation) != config["dataset"]["validationClips"] or audits["separation"]["waneClipsChecked"] != 235:
                raise ValueError("The frozen dataset counts changed")
            result["preflightSha256"] = args.preflight_sha256
            result["inputAudit"] = audits
            print(json.dumps({"stage": "pre-optimizer-audits-passed", **audits}), flush=True)
            selection, hyper = config["selection"], config["frozenHyperparameters"]
            target_artifacts = data / f"{candidate_id}-full-song-target-development"
            command = [sys.executable, "-m", "ml.training.train_muscriptor_piano", "--train-manifest", str(train_path), "--validation-manifest", str(val_path), "--base", str(BASE), "--out", str(output), "--candidate-id", candidate_id, "--rejection-receipt", str(receipt), "--execute", "--rights-acknowledgement", "I_HAVE_TRAINING_RIGHTS", "--minimum-train-songs", "20", "--train-last-layers", "0", "--conditioning-mode", "unconditioned", "--precision", hyper["precision"], "--seed", hyper["seed"], "--full-song-target-development-config", str(full_config_path), "--full-song-target-development-artifact-dir", str(target_artifacts), *training_adaptation_flags]
            if config.get("diagnoseTargetWhenBroadSafetyFails") is True:
                command.append("--target-development-diagnostics-on-rejection")
            learning_rate_key = "learningRate" if config.get("trainingMode") == "attention-lora" else "headLearningRate"
            decay_key = "weightDecay" if config.get("trainingMode") == "attention-lora" else "headWeightDecay"
            for flag, key in (("epochs", "epochs"), ("learning-rate", learning_rate_key), ("weight-decay", decay_key), ("gradient-accumulation", "gradientAccumulation"), ("gradient-clip-norm", "gradientClipNorm"), ("validation-interval-steps", "validationIntervalSteps"), ("max-optimizer-steps", "maxOptimizerSteps"), ("timing-token-weight", "timingTokenWeight"), ("note-on-token-weight", "noteOnTokenWeight"), ("note-off-token-weight", "noteOffTokenWeight"), ("eos-token-weight", "eosTokenWeight")):
                command += [f"--{flag}", str(hyper[key])]
            for flag, key in (("decoded-gate-max-aggregate-f1-regression", "maximumBroadPanelAggregateF1Regression"), ("decoded-gate-max-aggregate-recall-regression", "maximumBroadPanelAggregateRecallRegression"), ("decoded-gate-max-per-song-f1-regression", "maximumBroadPanelPerSongF1Regression"), ("target-development-minimum-100ms-f1-improvement", "minimumHeldOut100msF1Improvement"), ("target-development-minimum-100ms-recall-improvement", "minimumHeldOut100msRecallImprovement"), ("target-development-maximum-250ms-f1-regression", "maximumHeldOut250msF1Regression"), ("target-development-maximum-250ms-recall-regression", "maximumHeldOut250msRecallRegression")):
                command += [f"--{flag}", str(selection[key])]
            result["stage"] = "optimizer-run"
            run_stage(result["stage"], command)
            comparison = output / "evaluation-piano-task-full-validation.json"
            result["stage"] = "full-258-clip-validation"
            run_stage(result["stage"], [sys.executable, "-m", "ml.training.evaluate_checkpoint", "--base", str(BASE), "--candidate", str(output / "model.safetensors"), "--validation-manifest", str(val_path), *evaluation_route_flags, "--out", str(comparison)])
            timelines = {}
            for side in ("baseline", "candidate"):
                timelines[side] = output / f"evaluation-{side}-continuous.json"
                run_stage(f"rescore-{side}-continuous", [sys.executable, "-m", "ml.training.rescore_song_timelines", "--evaluation", str(comparison), "--validation-manifest", str(val_path), "--comparison-side", side, "--out", str(timelines[side])])
            result["stage"] = "full-validation-gate"
            run_stage(result["stage"], [sys.executable, "-m", "ml.training.apply_route_matched_validation_gate", "--comparison", str(comparison), "--baseline-timeline", str(timelines["baseline"]), "--candidate-timeline", str(timelines["candidate"]), "--training-metadata", str(output / "training-metadata.json"), "--gate", str(gate_path), "--out", str(output / "route-matched-validation-gate.json"), "--fail-on-rejection"])
            metadata = json.loads((output / "training-metadata.json").read_text())
            selection_metadata = metadata["decodedCheckpointSelection"]
            selected = next(row for row in selection_metadata["checkpointDecisions"] if row["checkpointIndex"] == selection_metadata["selectedCheckpointIndex"])
            raw = selected["targetDevelopmentArtifact"]["raw"]
            verify(Path(raw["path"]), raw["sha256"])
            if metadata["optimizerSteps"] != hyper["expectedOptimizerSteps"]:
                raise ValueError("The run did not complete its frozen one-epoch schedule")
            shutil.copyfile(broad_raw, output / "wane-full-base-raw.json")
            if config.get("trainingMode") == "attention-lora":
                result["stage"] = "merged-checkpoint-whole-song-reproduction"
                candidate_hash = json.loads(comparison.read_text())["candidateCheckpointSha256"]
                run_stage(result["stage"], [sys.executable, "-m", "ml.training.reproduce_merged_attention_candidate", "--checkpoint", str(output / "model.safetensors"), "--checkpoint-sha256", candidate_hash, "--expected-raw", raw["path"], "--expected-raw-sha256", raw["sha256"], "--out", str(output / "wane-full-candidate-raw.json")])
            else:
                shutil.copyfile(raw["path"], output / "wane-full-candidate-raw.json")
            result["stage"] = "existing-route-source-aware-piano-comparison"
            run_stage(result["stage"], ["bash", "scripts/training/evaluate_phase103_wane_production_route.sh", candidate_id])
            production_replay = json.loads((output / "wane-production-route-evaluation/evaluation.json").read_text())
            gates = production_replay.get("promotionGates")
            if production_replay.get("decision") != "PROMOTE" or not isinstance(gates, dict) or not gates or not all(value is True for value in gates.values()):
                raise ValueError("Candidate did not beat the existing source-aware piano output")
            result["stage"] = "research-gates-passed-other-song-transfer-and-listening-still-required"
            code = 0
        except Exception as exc:
            result["error"] = str(exc)
            print(json.dumps({"stage": result.get("stage", "pre-optimizer-audit"), "error": str(exc)}), flush=True)
        finally:
            result["exitCode"] = code
            result["candidateFileWritten"] = (output / "model.safetensors").is_file()
            result_path.write_text(json.dumps(result, indent=2) + "\n")
            status.write_text(f"{code}\n")
            os.sync()
        raise SystemExit(code)


if __name__ == "__main__":
    main()
