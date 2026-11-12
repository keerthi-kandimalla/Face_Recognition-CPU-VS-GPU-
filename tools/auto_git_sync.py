from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "logs" / "auto_git_sync.log"
PENDING_PUSH_PATH = ROOT / "logs" / ".auto_git_pending_push"
POLL_SECONDS = 5
QUIET_SECONDS = 45
PUSH_RETRY_SECONDS = 300
MAX_FILE_BYTES = 90 * 1024 * 1024
INCOMPLETE_SUFFIXES = {".tmp", ".temp", ".part", ".swp", ".bak"}
SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(
        rb"(?i)\b(?:[A-Za-z][A-Za-z0-9]*[_-])?"
        rb"(?:key|token|secret|password|passwd|credential)[\"']?\s*[:=]\s*[\"']?"
        rb"([A-Za-z0-9_./+=-]{12,})"
    ),
)
PLACEHOLDER_PREFIXES = (
    b"os.getenv",
    b"getenv",
    b"your_",
    b"replace_",
    b"changeme",
    b"placeholder",
    b"example",
    b"redacted",
    b"none",
    b"null",
)


def configure_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def ensure_git_on_path() -> None:
    if shutil.which("git"):
        return
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    candidates = (
        Path(local_app_data) / "Programs/Git/cmd/git.exe",
        Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/cmd/git.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            os.environ["PATH"] = str(candidate.parent) + os.pathsep + os.environ.get("PATH", "")
            return
    raise RuntimeError("Git was not found. Install Git before enabling automatic sync.")


def git(args: list[str], *, check: bool = True, binary: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
        encoding=None if binary else "utf-8",
        errors=None if binary else "replace",
        check=False,
    )
    if check and result.returncode:
        stderr = result.stderr.decode("utf-8", "replace") if binary else result.stderr
        raise RuntimeError(stderr.strip() or f"git {' '.join(args)} failed")
    return result


def changed_entries() -> list[tuple[str, str]]:
    result = git(["status", "--porcelain=v1", "-z", "--untracked-files=all"], binary=True)
    records = result.stdout.decode("utf-8", "surrogateescape").split("\0")
    entries: list[tuple[str, str]] = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        status = record[:2]
        path = record[3:]
        entries.append((status, path))
        if "R" in status or "C" in status:
            if index < len(records) and records[index]:
                entries.append((status, records[index]))
                index += 1
    return entries


def pending_paths(entries: list[tuple[str, str]]) -> list[str]:
    return sorted({path.replace("\\", "/") for _, path in entries})


def staged_entries(entries: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [entry for entry in entries if entry[0][0] not in (" ", "?")]


def unsafe_path(path: str) -> str | None:
    normalized = path.replace("\\", "/")
    parts = [part.lower() for part in normalized.split("/")]
    name = parts[-1]
    if name == ".env" or name.startswith(".env.") and name != ".env.example":
        return "environment file"
    if name in {"credentials.json", "secrets.json", "secrets.yaml", "secrets.yml"}:
        return "credential file"
    if name.startswith(("id_rsa", "id_ed25519")) or name.endswith((".pem", ".p12", ".pfx", ".ppk")):
        return "private key or certificate file"
    if any(part in {".aws", ".ssh"} for part in parts):
        return "private configuration directory"
    if Path(name).suffix in INCOMPLETE_SUFFIXES or name.endswith("~"):
        return "temporary or incomplete file"
    return None


def content_has_secret(data: bytes) -> bool:
    if b"\0" in data:
        return False
    for pattern in SECRET_PATTERNS:
        for match in pattern.finditer(data):
            candidate = match.group(1) if match.lastindex else match.group(0)
            if candidate.lower().startswith(PLACEHOLDER_PREFIXES):
                continue
            if pattern is SECRET_PATTERNS[-1] and candidate.lower() in {b"true", b"false"}:
                continue
            return True
    return False


def scan_paths(paths: list[str]) -> None:
    for path in paths:
        reason = unsafe_path(path)
        if reason:
            raise RuntimeError(f"Refusing to sync {path}: {reason} is not allowed.")
        file_path = ROOT / Path(path)
        if not file_path.is_file():
            continue
        size = file_path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise RuntimeError(f"Refusing to sync {path}: file exceeds the 90 MiB safety limit.")
        if content_has_secret(file_path.read_bytes()):
            raise RuntimeError(f"Refusing to sync {path}: a possible secret was detected.")


def fingerprint(paths: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        file_path = ROOT / Path(path)
        if file_path.is_file():
            result[path] = hashlib.sha256(file_path.read_bytes()).hexdigest()
        else:
            result[path] = "<deleted>"
    return result


def change_signature(paths: list[str]) -> tuple[tuple[str, int, int], ...]:
    signature = []
    for path in paths:
        file_path = ROOT / Path(path)
        try:
            stat = file_path.stat()
            signature.append((path, stat.st_mtime_ns, stat.st_size))
        except FileNotFoundError:
            signature.append((path, -1, -1))
    return tuple(signature)


def validate(paths: list[str]) -> None:
    diff_check = git(["diff", "--check"], check=False)
    if diff_check.returncode:
        raise RuntimeError("git diff --check failed; fix whitespace errors before syncing.")

    python_paths = [str(ROOT / Path(path)) for path in paths if path.endswith(".py") and (ROOT / Path(path)).is_file()]
    if python_paths:
        subprocess.run([sys.executable, "-m", "py_compile", *python_paths], cwd=ROOT, check=True)
        if any(path.startswith(("backend/", "tests/")) for path in paths):
            subprocess.run([sys.executable, "-m", "tests.test_pipeline"], cwd=ROOT, check=True)

    javascript_paths = [path for path in paths if path.endswith(".js") and (ROOT / Path(path)).is_file()]
    if javascript_paths:
        node = shutil.which("node")
        if not node:
            raise RuntimeError("Node.js is required to validate changed JavaScript files.")
        for path in javascript_paths:
            subprocess.run([node, "--check", str(ROOT / Path(path))], cwd=ROOT, check=True)


def staged_secret_scan(paths: list[str]) -> None:
    for path in paths:
        reason = unsafe_path(path)
        if reason:
            raise RuntimeError(f"Refusing to sync {path}: {reason} is not allowed.")
        result = git(["show", f":{path}"], check=False, binary=True)
        if result.returncode == 0 and content_has_secret(result.stdout):
            raise RuntimeError(f"Refusing to sync {path}: a possible secret was detected.")


def outgoing_secret_scan() -> None:
    upstream = git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], check=False)
    if upstream.returncode:
        commits = git(["rev-list", "--reverse", "HEAD"]).stdout.splitlines()
    else:
        commits = git(["rev-list", "--reverse", f"{upstream.stdout.strip()}..HEAD"]).stdout.splitlines()
    for commit in commits:
        changed = git(["diff-tree", "--root", "--no-commit-id", "--name-only", "-r", "-z", commit], binary=True)
        paths = [path for path in changed.stdout.decode("utf-8", "surrogateescape").split("\0") if path]
        for path in paths:
            reason = unsafe_path(path)
            if reason:
                raise RuntimeError(f"Refusing to push commit {commit[:8]}: {path} is a {reason}.")
            blob = git(["show", f"{commit}:{path}"], check=False, binary=True)
            if blob.returncode:
                continue
            if len(blob.stdout) > MAX_FILE_BYTES:
                raise RuntimeError(f"Refusing to push commit {commit[:8]}: {path} exceeds the 90 MiB safety limit.")
            if content_has_secret(blob.stdout):
                raise RuntimeError(f"Refusing to push commit {commit[:8]}: a possible secret was detected in {path}.")


def commit_message(paths: list[str]) -> str:
    paths = sorted(paths)
    if len(paths) <= 3:
        details = ", ".join(paths)
    else:
        details = ", ".join(paths[:2]) + f", and {len(paths) - 2} more"
    message = f"Auto-sync: update {details}"
    return message if len(message) <= 72 else message[:69].rstrip() + "..."


def push_head() -> None:
    remotes = git(["remote"], check=True).stdout.splitlines()
    if "origin" not in remotes:
        raise RuntimeError("No 'origin' remote is configured; automatic push is disabled.")
    outgoing_secret_scan()
    upstream = git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], check=False)
    if upstream.returncode:
        result = git(["push", "--set-upstream", "origin", "HEAD"], check=False)
    else:
        result = git(["push", "origin", "HEAD"], check=False)
    if result.returncode:
        raise RuntimeError(f"Push failed; local commit was kept. Resolve the issue manually. {result.stderr.strip()}")
    commit = git(["log", "-1", "--format=%h %s"]).stdout.strip()
    logging.info("Push succeeded: %s", commit)
    print(f"Automatic GitHub push succeeded: {commit}", flush=True)
    PENDING_PUSH_PATH.unlink(missing_ok=True)


def commit_and_push(paths: list[str]) -> None:
    entries = changed_entries()
    if staged_entries(entries):
        raise RuntimeError("Existing staged changes detected; automatic sync will not alter the index.")
    scan_paths(paths)
    before = fingerprint(paths)
    validate(paths)
    if fingerprint(paths) != before:
        raise RuntimeError("Files changed during validation; automatic sync was cancelled.")
    git(["add", "-A", "--", "."])
    staged = git(["diff", "--cached", "--name-only", "-z"], binary=True)
    staged_paths = [path for path in staged.stdout.decode("utf-8", "surrogateescape").split("\0") if path]
    if not staged_paths:
        return
    staged_secret_scan(staged_paths)
    if any(entry[1] and entry[0][1] not in (" ", "?") for entry in changed_entries()):
        raise RuntimeError("Files changed while staging; staged changes were left for review.")
    git(["commit", "-m", commit_message(staged_paths)])
    PENDING_PUSH_PATH.parent.mkdir(parents=True, exist_ok=True)
    PENDING_PUSH_PATH.write_text("pending\n", encoding="utf-8")
    push_head()


def run_watch() -> None:
    configure_logging()
    ensure_git_on_path()
    logging.info("Automatic Git sync watcher started for %s", ROOT)
    previous = pending_paths(changed_entries())
    previous_signature = change_signature(previous)
    last_change = time.monotonic()
    last_push_attempt = 0.0
    while True:
        try:
            entries = changed_entries()
            paths = pending_paths(entries)
            signature = change_signature(paths)
            now = time.monotonic()
            if paths != previous or signature != previous_signature:
                previous = paths
                previous_signature = signature
                last_change = now
            if paths and now - last_change >= QUIET_SECONDS:
                try:
                    commit_and_push(paths)
                    previous = pending_paths(changed_entries())
                    previous_signature = change_signature(previous)
                    last_change = time.monotonic()
                    last_push_attempt = last_change
                except Exception as error:
                    logging.error("Automatic sync stopped: %s", error)
                    last_change = now
            elif PENDING_PUSH_PATH.exists() and now - last_push_attempt >= PUSH_RETRY_SECONDS:
                try:
                    push_head()
                except Exception as error:
                    logging.error("Pending push needs attention: %s", error)
                last_push_attempt = now
        except Exception as error:
            logging.error("Watcher check failed: %s", error)
        time.sleep(POLL_SECONDS)


def self_test() -> None:
    password_sample = b"pass" + b"word = 'a-real-password-value-123'"
    github_sample = b"ghp_" + b"abcdefghijklmnopqrstuvwxyz123456"
    token_sample = b"api_" + b"token" + b" = 'a-real-token-value-123'"
    assert content_has_secret(password_sample)
    assert content_has_secret(github_sample)
    assert content_has_secret(token_sample)
    assert not content_has_secret(b"password = os.getenv('APP_PASSWORD')")
    assert not content_has_secret(b"password = 'replace_me_with_a_secret'")
    assert unsafe_path(".env")
    assert unsafe_path("dataset/person/image.jpg") is None
    assert commit_message(["backend/main.py", "frontend/app.js"]) == "Auto-sync: update backend/main.py, frontend/app.js"
    print("Auto-sync safety self-test passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Debounced, validated GitHub auto-sync watcher.")
    parser.add_argument("command", choices=("watch", "self-test"))
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    else:
        run_watch()


if __name__ == "__main__":
    main()