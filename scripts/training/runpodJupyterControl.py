#!/usr/bin/env python3
"""Execute commands and transfer files through an authenticated RunPod Jupyter Pod.

The Jupyter password must be supplied through RUNPOD_JUPYTER_PASSWORD.  Keeping it
out of command-line arguments prevents it from being stored in shell history.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import requests
import websocket


class JupyterControlError(RuntimeError):
    pass


def authenticated_session(base_url: str) -> requests.Session:
    password = os.environ.get("RUNPOD_JUPYTER_PASSWORD", "")
    if not password:
        raise JupyterControlError("RUNPOD_JUPYTER_PASSWORD is not set")

    session = requests.Session()
    login_url = f"{base_url.rstrip('/')}/login"
    response = session.get(login_url, timeout=30)
    response.raise_for_status()
    xsrf = session.cookies.get("_xsrf")
    if not xsrf:
        raise JupyterControlError("Jupyter did not issue an XSRF cookie")

    response = session.post(
        login_url,
        data={"_xsrf": xsrf, "password": password, "next": "/lab"},
        headers={"X-XSRFToken": xsrf},
        allow_redirects=False,
        timeout=30,
    )
    if response.status_code not in {302, 303}:
        raise JupyterControlError(
            f"Jupyter login failed with HTTP {response.status_code}"
        )
    return session


def api_headers(session: requests.Session) -> dict[str, str]:
    xsrf = session.cookies.get("_xsrf")
    return {"X-XSRFToken": xsrf} if xsrf else {}


def execute(base_url: str, session: requests.Session, code: str, timeout: int) -> int:
    api_root = f"{base_url.rstrip('/')}/api"
    response = session.post(
        f"{api_root}/kernels",
        json={"name": "python3"},
        headers=api_headers(session),
        timeout=30,
    )
    response.raise_for_status()
    kernel_id = response.json()["id"]

    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    session_id = uuid.uuid4().hex
    ws_url = (
        f"{scheme}://{parsed.netloc}/api/kernels/{kernel_id}/channels"
        f"?session_id={session_id}"
    )
    cookie_header = "; ".join(
        f"{key}={value}" for key, value in session.cookies.items()
    )
    ws = websocket.create_connection(
        ws_url,
        cookie=cookie_header,
        origin=f"{parsed.scheme}://{parsed.netloc}",
        timeout=min(timeout, 60),
    )
    message_id = uuid.uuid4().hex
    request = {
        "header": {
            "msg_id": message_id,
            "username": "polymath-training",
            "session": session_id,
            "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "msg_type": "execute_request",
            "version": "5.3",
        },
        "parent_header": {},
        "metadata": {},
        "content": {
            "code": code,
            "silent": False,
            "store_history": False,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": True,
        },
        "buffers": [],
        "channel": "shell",
    }
    ws.send(json.dumps(request))

    deadline = time.monotonic() + timeout
    exit_code = 0
    try:
        while time.monotonic() < deadline:
            ws.settimeout(min(15, max(1, deadline - time.monotonic())))
            try:
                message = json.loads(ws.recv())
            except websocket.WebSocketTimeoutException:
                continue
            if message.get("parent_header", {}).get("msg_id") != message_id:
                continue
            msg_type = message.get("msg_type")
            content = message.get("content", {})
            if msg_type == "stream":
                target = sys.stderr if content.get("name") == "stderr" else sys.stdout
                print(content.get("text", ""), end="", file=target, flush=True)
            elif msg_type in {"execute_result", "display_data"}:
                text = content.get("data", {}).get("text/plain")
                if text:
                    print(text)
            elif msg_type == "error":
                exit_code = 1
                traceback = content.get("traceback") or [content.get("evalue", "error")]
                print("\n".join(traceback), file=sys.stderr)
            elif msg_type == "status" and content.get("execution_state") == "idle":
                return exit_code
        raise JupyterControlError(f"execution timed out after {timeout} seconds")
    finally:
        ws.close()
        try:
            session.delete(
                f"{api_root}/kernels/{kernel_id}",
                headers=api_headers(session),
                timeout=15,
            )
        except requests.RequestException:
            pass


def upload_chunked(
    base_url: str,
    session: requests.Session,
    local_path: Path,
    remote_path: str,
    *,
    chunk_bytes: int = 512 * 1024,
) -> None:
    """Upload through a kernel WebSocket to avoid proxy PUT timeouts.

    RunPod's Cloudflare-backed Jupyter proxy can time out while buffering a
    multi-megabyte Contents API PUT.  Small ordered WebSocket messages avoid
    that limit and the final size/SHA-256 check makes the transfer auditable.
    """

    remote = PurePosixPath(remote_path)
    if not remote.is_absolute() or not str(remote).startswith("/workspace/"):
        raise JupyterControlError("chunked uploads are restricted to /workspace")
    expected_size = local_path.stat().st_size
    expected_sha256 = hashlib.sha256(local_path.read_bytes()).hexdigest()

    api_root = f"{base_url.rstrip('/')}/api"
    response = session.post(
        f"{api_root}/kernels",
        json={"name": "python3"},
        headers=api_headers(session),
        timeout=30,
    )
    response.raise_for_status()
    kernel_id = response.json()["id"]
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    session_id = uuid.uuid4().hex
    ws_url = (
        f"{scheme}://{parsed.netloc}/api/kernels/{kernel_id}/channels"
        f"?session_id={session_id}"
    )
    cookie_header = "; ".join(
        f"{key}={value}" for key, value in session.cookies.items()
    )
    ws = websocket.create_connection(
        ws_url,
        cookie=cookie_header,
        origin=f"{parsed.scheme}://{parsed.netloc}",
        timeout=60,
    )

    def run(code: str, timeout: int = 90) -> None:
        message_id = uuid.uuid4().hex
        request = {
            "header": {
                "msg_id": message_id,
                "username": "polymath-transfer",
                "session": session_id,
                "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "msg_type": "execute_request",
                "version": "5.3",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": False,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            },
            "buffers": [],
            "channel": "shell",
        }
        ws.send(json.dumps(request))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ws.settimeout(min(15, max(1, deadline - time.monotonic())))
            try:
                message = json.loads(ws.recv())
            except websocket.WebSocketTimeoutException:
                continue
            if message.get("parent_header", {}).get("msg_id") != message_id:
                continue
            msg_type = message.get("msg_type")
            content = message.get("content", {})
            if msg_type == "error":
                traceback = content.get("traceback") or [content.get("evalue", "error")]
                raise JupyterControlError("\n".join(traceback))
            if msg_type == "status" and content.get("execution_state") == "idle":
                return
        raise JupyterControlError("chunk execution timed out")

    temporary = remote.with_name(f".{remote.name}.{uuid.uuid4().hex}.uploading")
    try:
        run(
            "from pathlib import Path; "
            f"p=Path({str(temporary)!r}); p.parent.mkdir(parents=True, exist_ok=True); "
            "p.write_bytes(b'')"
        )
        transferred = 0
        with local_path.open("rb") as source:
            while chunk := source.read(chunk_bytes):
                encoded = base64.b64encode(chunk).decode("ascii")
                run(
                    "import base64; "
                    f"open({str(temporary)!r}, 'ab').write(base64.b64decode({encoded!r}))"
                )
                transferred += len(chunk)
                print(
                    f"{local_path.name}: {transferred}/{expected_size} bytes",
                    flush=True,
                )
        run(
            "from pathlib import Path; import hashlib, os; "
            f"p=Path({str(temporary)!r}); "
            f"assert p.stat().st_size == {expected_size}; "
            f"assert hashlib.sha256(p.read_bytes()).hexdigest() == {expected_sha256!r}; "
            f"os.replace(p, Path({str(remote)!r}))"
        )
    finally:
        ws.close()
        try:
            session.delete(
                f"{api_root}/kernels/{kernel_id}",
                headers=api_headers(session),
                timeout=15,
            )
        except requests.RequestException:
            pass
    print(f"uploaded and verified {local_path} -> {remote}")


def upload(
    base_url: str, session: requests.Session, local_path: Path, remote_path: str
) -> None:
    payload = {
        "type": "file",
        "format": "base64",
        "content": base64.b64encode(local_path.read_bytes()).decode("ascii"),
    }
    response = session.put(
        f"{base_url.rstrip('/')}/api/contents/{remote_path.lstrip('/')}",
        json=payload,
        headers=api_headers(session),
        timeout=600,
    )
    response.raise_for_status()
    print(f"uploaded {local_path} -> /{remote_path.lstrip('/')}")


def download(
    base_url: str, session: requests.Session, remote_path: str, local_path: Path
) -> None:
    response = session.get(
        f"{base_url.rstrip('/')}/api/contents/{remote_path.lstrip('/')}",
        params={"content": 1},
        timeout=600,
    )
    response.raise_for_status()
    body = response.json()
    if body.get("format") == "base64":
        data = base64.b64decode(body["content"])
    else:
        data = str(body.get("content", "")).encode("utf-8")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(data)
    print(f"downloaded /{remote_path.lstrip('/')} -> {local_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    execute_parser = subparsers.add_parser("exec")
    code_group = execute_parser.add_mutually_exclusive_group(required=True)
    code_group.add_argument("--code")
    code_group.add_argument("--code-file", type=Path)
    execute_parser.add_argument("--timeout", type=int, default=600)

    upload_parser = subparsers.add_parser("upload")
    upload_parser.add_argument("--local", type=Path, required=True)
    upload_parser.add_argument("--remote", required=True)

    chunked_upload_parser = subparsers.add_parser("upload-chunked")
    chunked_upload_parser.add_argument("--local", type=Path, required=True)
    chunked_upload_parser.add_argument("--remote", required=True)
    chunked_upload_parser.add_argument(
        "--chunk-bytes", type=int, default=512 * 1024
    )

    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--remote", required=True)
    download_parser.add_argument("--local", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    # PowerShell may expose a legacy Windows code page.  Training logs can
    # contain Unicode arrows/progress symbols, so force lossless UTF-8 output.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = parse_args()
    session = authenticated_session(args.base_url)
    if args.operation == "exec":
        code = args.code if args.code is not None else args.code_file.read_text("utf-8")
        return execute(args.base_url, session, code, args.timeout)
    if args.operation == "upload":
        upload(args.base_url, session, args.local, args.remote)
        return 0
    if args.operation == "upload-chunked":
        upload_chunked(
            args.base_url,
            session,
            args.local,
            args.remote,
            chunk_bytes=args.chunk_bytes,
        )
        return 0
    if args.operation == "download":
        download(args.base_url, session, args.remote, args.local)
        return 0
    raise AssertionError(args.operation)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (JupyterControlError, requests.RequestException) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
