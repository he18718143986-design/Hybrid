#!/usr/bin/env python3
"""Write BASELINE.lock from submission_v2.py (production baseline fingerprint)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "submission_v2.py"
LOCK = ROOT / "BASELINE.lock"


def git_short() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return "N/A"


def main() -> int:
    if not BASELINE.is_file():
        print(f"missing {BASELINE}", file=sys.stderr)
        return 1
    data = BASELINE.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    version_tag = ""
    for line in BASELINE.read_text(encoding="utf-8").splitlines()[:5]:
        if line.startswith("#") and "writefile" not in line.lower():
            version_tag = line.lstrip("# ").strip()
            break
    payload = {
        "production_file": "submission_v2.py",
        "version_tag": version_tag,
        "sha256": sha,
        "sha256_short": sha[:12],
        "lines": len(data.splitlines()),
        "locked_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit_short": git_short(),
        "policy": "All experiments branch from submission_v2.py. Do not edit in place.",
    }
    LOCK.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {LOCK} prod={payload['sha256_short']} tag={version_tag!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
