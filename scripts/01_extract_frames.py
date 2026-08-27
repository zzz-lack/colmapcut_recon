#!/usr/bin/env python3
"""CLI entry point for deterministic video frame sampling."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from colmapcut_recon.preprocessing.extract_frames import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
