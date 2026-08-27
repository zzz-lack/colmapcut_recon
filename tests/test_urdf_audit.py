from pathlib import Path

from colmapcut_recon.simulation.urdf_audit import audit_urdf


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets/robots/mobile_manipulator/urdf/combined_source.urdf"
REPAIRED = ROOT / "assets/robots/mobile_manipulator/urdf/combined_mobile.urdf"


def test_source_urdf_reports_known_physics_problems() -> None:
    issues = audit_urdf(SOURCE)
    codes_and_names = {(issue.code, issue.element_name) for issue in issues}
    assert ("invalid_inertia", "link_base") in codes_and_names
    assert ("locked_revolute", "front_left_leg_joint") in codes_and_names
    assert ("invalid_name", "$center_zed2_camera_center_joint") in codes_and_names


def test_repaired_urdf_has_no_error_or_locked_revolute() -> None:
    issues = audit_urdf(REPAIRED)
    assert not [issue for issue in issues if issue.severity == "error"]
    assert not [issue for issue in issues if issue.code == "locked_revolute"]
