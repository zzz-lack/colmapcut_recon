#!/usr/bin/env python3
"""CLI entry point for external 3DGRUT training."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from colmapcut_recon.reconstruction.train_3dgrut import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
