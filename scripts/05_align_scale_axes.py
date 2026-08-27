#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=2.0",
#   "opencv-contrib-python-headless>=4.10",
#   "pycolmap>=3.11",
#   "PyYAML>=6.0",
# ]
# ///
"""Align a COLMAP model to metric plant coordinates using coplanar AprilTags."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from colmapcut_recon.geometry.apriltag_alignment import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
