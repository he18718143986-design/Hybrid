#!/usr/bin/env python3
"""Bundle submission/main.py + src/ for Kaggle tar.gz."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="hybrid_submission.tar.gz")
    args = parser.parse_args()

    out = ROOT / args.output
    with tarfile.open(out, "w:gz") as tar:
        tar.add(ROOT / "submission" / "main.py", arcname="main.py")
        tar.add(ROOT / "src", arcname="src")
        if (ROOT / "submission_v2.py").is_file():
            tar.add(ROOT / "submission_v2.py", arcname="submission_v2.py")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
