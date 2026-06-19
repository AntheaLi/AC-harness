# This module reads AC-Core outputs / writes observed evidence.
# It must not implement compiler logic.
"""
Provenance helpers.

Every record inserted into the evidence store must be stamped with enough
context to be reconstructed later: schema version, when, what code,
which file it came from, and which command produced it. The spec (§6)
treats this as non-negotiable.

Provenance is a plain dict (not a pydantic model) so it remains flexible
for ad-hoc extensions and survives roundtrips through JSON without
shape gymnastics.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


def _git_commit() -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        out = subprocess.run(
            [git, "rev-parse", "HEAD"],
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        return None
    return None


def make_provenance(
    *,
    source_path: str | None = None,
    command: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a provenance dict for a newly created record.

    Args:
        source_path: file the record was imported from (if any).
        command: CLI invocation or function call that produced it.
        extra: caller-specific extras (e.g. tool version, plan id).
    """
    prov: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": _safe_host(),
        "python_version": sys.version.split()[0],
    }
    commit = _git_commit()
    if commit:
        prov["git_commit"] = commit
    if source_path:
        prov["source_path"] = os.path.abspath(source_path)
    if command:
        prov["command"] = command
    if extra:
        prov.update(extra)
    return prov


def _safe_host() -> str:
    try:
        import socket

        return socket.gethostname()
    except Exception:
        return "unknown"
