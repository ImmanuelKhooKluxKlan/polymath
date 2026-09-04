"""Orchestrate base evaluation, QLoRA training, and candidate evaluation on one Pod."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume-root", default="/runpod-volume/chatboss")
    parser.add_argument("--version", default="v002")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--cases", default=None, help="Independent behavior holdout JSONL")
    parser.add_argument(
        "--evaluate-existing-candidate",
        action="store_true",
        help="Resume evaluation only when a complete candidate and training manifest already exist.",
    )
    return parser.parse_args()


def run(command: list[str], log_handle) -> None:
    printable = " ".join(command)
    print(f"\n$ {printable}", flush=True)
    log_handle.write(f"\n$ {printable}\n")
    log_handle.flush()
    subprocess.run(command, check=True, stdout=log_handle, stderr=subprocess.STDOUT)


def safe_command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=30).strip()
    except Exception:
        return None


def write_status(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def reusable_base_report(path: Path, cases_path: Path, model: str, revision: str) -> bool:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        expected_ids = [json.loads(line)["id"] for line in cases_path.read_text(encoding="utf-8").splitlines() if line]
        actual_ids = [item["id"] for item in report["results"]]
        return all((
            report.get("schema") == "polymath-chatboss-behavior-eval-v1",
            report.get("model") == model,
            report.get("revision") == revision,
            report.get("adapter") is None,
            report.get("loadingMode") == "image-text-conditional",
            report.get("casesSha256") == hashlib.sha256(cases_path.read_bytes()).hexdigest(),
            actual_ids == expected_ids,
        ))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def main() -> None:
    args = parse_args()
    script_root = Path(__file__).resolve().parent
    volume_root = Path(args.volume_root).resolve()
    data_root = volume_root / "datasets" / args.version
    candidate_root = volume_root / "candidates" / args.version
    eval_root = volume_root / "evals" / args.version
    run_root = volume_root / "runs" / args.version
    cache_root = volume_root / "huggingface-cache"
    status_path = run_root / "status.json"
    log_path = run_root / "run.log"
    lock_path = run_root / "active.lock"

    for directory in (candidate_root, eval_root, run_root, cache_root):
        directory.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        raise SystemExit(f"Another training run may be active: {lock_path}")
    candidate_exists = (candidate_root / "adapter" / "adapter_model.safetensors").exists()
    candidate_manifest_exists = (candidate_root / "training_manifest.json").exists()
    if candidate_exists and not args.evaluate_existing_candidate:
        raise SystemExit(f"Candidate {args.version} already exists; use a new version instead of overwriting it.")
    if args.evaluate_existing_candidate and not (candidate_exists and candidate_manifest_exists):
        raise SystemExit("Evaluation resume requires both a saved adapter and its training manifest.")
    lock_path.write_text(f"pid={os.getpid()} started={now()}\n", encoding="utf-8")

    status = {
        "schema": "polymath-chatboss-run-status-v1",
        "version": args.version,
        "state": "starting",
        "startedAt": now(),
        "finishedAt": None,
        "baseModel": args.base_model,
        "revision": args.revision,
        "candidatePromoted": False,
        "host": {
            "python": sys.version,
            "platform": platform.platform(),
            "gpu": safe_command_output([
                "nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"
            ]),
        },
    }
    write_status(status_path, status)
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    base_report = eval_root / "base.json"
    candidate_report = eval_root / "candidate.json"
    decision_report = eval_root / "decision.json"
    cases = Path(args.cases).resolve() if args.cases else script_root / "behavior_holdout_v001.jsonl"
    if not cases.is_file():
        raise SystemExit(f"Behavior holdout not found: {cases}")

    try:
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            if not args.evaluate_existing_candidate:
                status["state"] = "verifying-token-boundaries"
                write_status(status_path, status)
                run([
                    sys.executable, str(script_root / "verify_chat_boundaries.py"),
                    "--base-model", args.base_model,
                    "--revision", args.revision,
                    "--input", str(data_root / "train.jsonl"),
                    "--input", str(data_root / "validation.jsonl"),
                    "--cache-dir", str(cache_root),
                ], log)

            if reusable_base_report(base_report, cases, args.base_model, args.revision):
                status["state"] = "reusing-verified-base-evaluation"
                write_status(status_path, status)
                log.write("Reusing the matching immutable base evaluation.\n")
            else:
                status["state"] = "evaluating-base"
                write_status(status_path, status)
                run([
                    sys.executable, str(script_root / "evaluate_adapter.py"),
                    "--base-model", args.base_model,
                    "--revision", args.revision,
                    "--cases", str(cases),
                    "--output", str(base_report),
                    "--cache-dir", str(cache_root),
                ], log)

            if args.evaluate_existing_candidate:
                status["state"] = "reusing-complete-candidate"
                write_status(status_path, status)
                log.write("Reusing a saved candidate with its completed training manifest.\n")
            else:
                status["state"] = "training-candidate"
                write_status(status_path, status)
                run([
                    sys.executable, str(script_root / "train_lora.py"),
                    "--base-model", args.base_model,
                    "--revision", args.revision,
                    "--train", str(data_root / "train.jsonl"),
                    "--validation", str(data_root / "validation.jsonl"),
                    "--output", str(candidate_root),
                    "--cache-dir", str(cache_root),
                    "--epochs", str(args.epochs),
                    "--learning-rate", str(args.learning_rate),
                ], log)

            status["state"] = "evaluating-candidate"
            write_status(status_path, status)
            run([
                sys.executable, str(script_root / "evaluate_adapter.py"),
                "--base-model", args.base_model,
                "--revision", args.revision,
                "--adapter", str(candidate_root / "adapter"),
                "--cases", str(cases),
                "--output", str(candidate_report),
                "--cache-dir", str(cache_root),
            ], log)
            run([
                sys.executable, str(script_root / "compare_evaluations.py"),
                "--base", str(base_report),
                "--candidate", str(candidate_report),
                "--output", str(decision_report),
            ], log)

        decision = json.loads(decision_report.read_text(encoding="utf-8"))
        status["state"] = "complete-awaiting-human-review"
        status["automatedGatesPassed"] = decision["allAutomatedGatesPassed"]
    except Exception as error:
        status["state"] = "failed"
        status["errorType"] = type(error).__name__
        status["error"] = str(error)[:1000]
        (run_root / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        status["finishedAt"] = now()
        write_status(status_path, status)
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
