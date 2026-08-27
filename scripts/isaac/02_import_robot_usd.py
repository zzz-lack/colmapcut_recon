#!/usr/bin/env python3
"""Import the repaired URDF as an unfixed Isaac Sim 5.1 articulation asset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaacsim import SimulationApp


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = ROOT / "assets/robots/mobile_manipulator/urdf/combined_mobile.urdf"
DEFAULT_USD = (
    ROOT
    / "assets/robots/mobile_manipulator/usd/combined_mobile_merged/combined_mobile_merged.usd"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_USD)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


args = parse_args()
simulation_app = SimulationApp({"headless": args.headless})

import omni.kit.commands  # noqa: E402
from pxr import Usd, UsdPhysics  # noqa: E402


def main() -> int:
    urdf_path = args.urdf.resolve(strict=True)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        raise RuntimeError("Could not create the Isaac Sim URDF import configuration")
    # Fixed camera frames, locked legs, and the arm mount do not need separate
    # PhysX bodies. Merging them produces a smaller, more stable articulation.
    import_config.merge_fixed_joints = True
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    import_config.fix_base = False
    import_config.self_collision = False
    import_config.make_default_prim = True
    import_config.create_physics_scene = False
    import_config.distance_scale = 1.0

    status, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(urdf_path),
        import_config=import_config,
        dest_path=str(output_path),
        get_articulation_root=True,
    )
    if not status:
        raise RuntimeError(f"Isaac Sim failed to import {urdf_path}")

    stage = Usd.Stage.Open(str(output_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"OpenUSD could not open imported asset: {output_path}")
    fixed_world_joints = []
    articulation_roots = []
    cameras = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.FixedJoint):
            joint = UsdPhysics.FixedJoint(prim)
            if not joint.GetBody0Rel().GetTargets():
                fixed_world_joints.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(str(prim.GetPath()))
        if prim.GetTypeName() == "Camera":
            cameras.append(str(prim.GetPath()))
    if fixed_world_joints:
        raise RuntimeError(f"Mobile import unexpectedly contains world-fixed joints: {fixed_world_joints}")
    if not articulation_roots:
        raise RuntimeError("Imported asset has no PhysicsArticulationRootAPI")

    report = {
        "urdf": str(urdf_path),
        "usd": str(output_path),
        "imported_prim_path": str(prim_path),
        "default_prim": str(stage.GetDefaultPrim().GetPath()),
        "articulation_roots": articulation_roots,
        "world_fixed_joints": fixed_world_joints,
        "camera_prims": cameras,
        "note": "The end-effector Camera prim is authored at runtime, so camera_prims is expected to be empty.",
    }
    report_path = output_path.with_name(output_path.stem + "_import_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


try:
    raise SystemExit(main())
finally:
    simulation_app.close()
