#!/usr/bin/env python3
"""Audit the supplied and repaired URDFs and write a machine-readable report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from colmapcut_recon.simulation.urdf_audit import audit_urdf


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "assets/robots/mobile_manipulator/urdf/combined_source.urdf",
    )
    parser.add_argument(
        "--repaired",
        type=Path,
        default=ROOT / "assets/robots/mobile_manipulator/urdf/combined_mobile.urdf",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/simulation/robot_urdf_audit.json",
    )
    args = parser.parse_args()
    report = {
        "source": {
            "path": str(args.source.resolve()),
            "issues": [issue.to_dict() for issue in audit_urdf(args.source)],
        },
        "repaired": {
            "path": str(args.repaired.resolve()),
            "issues": [issue.to_dict() for issue in audit_urdf(args.repaired)],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
