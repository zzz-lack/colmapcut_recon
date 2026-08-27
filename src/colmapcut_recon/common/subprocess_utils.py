"""Run external tools without shell interpolation and record structured metadata."""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path


def format_command(command: Sequence[str]) -> str:
    """Return a copy/pasteable representation without executing a shell."""

    return shlex.join(str(part) for part in command)


def run_command(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    record_path: Path | None = None,
) -> dict[str, object]:
    """Execute an argument vector and optionally write an invocation JSON record."""

    argv = [str(part) for part in command]
    started = time.time()
    record: dict[str, object] = {
        "command": argv,
        "command_text": format_command(argv),
        "cwd": str(cwd.resolve()) if cwd else None,
        "started_unix": started,
    }
    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            env=dict(env) if env is not None else None,
            check=True,
        )
        record["returncode"] = completed.returncode
        return record
    except subprocess.CalledProcessError as exc:
        record["returncode"] = exc.returncode
        raise
    finally:
        record["duration_seconds"] = time.time() - started
        if record_path is not None:
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(
                json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
