"""Static checks for robot URDF properties needed by Isaac Sim/PhysX."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path


_VALID_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class AuditIssue:
    severity: str
    component: str
    element_type: str
    element_name: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _component(name: str) -> str:
    if name.startswith("link") or name.startswith("joint") or "eef" in name:
        return "机械臂"
    if any(token in name for token in ("wheel", "leg", "base", "imu", "gps", "zed2")):
        return "底盘/底盘传感器"
    return "未分类"


def _float_attr(element: ET.Element, name: str) -> float:
    return float(element.attrib[name])


def _inertia_is_positive_definite(inertia: ET.Element) -> bool:
    ixx = _float_attr(inertia, "ixx")
    ixy = _float_attr(inertia, "ixy")
    ixz = _float_attr(inertia, "ixz")
    iyy = _float_attr(inertia, "iyy")
    iyz = _float_attr(inertia, "iyz")
    izz = _float_attr(inertia, "izz")
    minor_1 = ixx
    minor_2 = ixx * iyy - ixy * ixy
    determinant = (
        ixx * (iyy * izz - iyz * iyz)
        - ixy * (ixy * izz - iyz * ixz)
        + ixz * (ixy * iyz - iyy * ixz)
    )
    return minor_1 > 0.0 and minor_2 > 0.0 and determinant > 0.0


def _is_frame_only(link: ET.Element, child_joint: ET.Element | None) -> bool:
    return (
        child_joint is not None
        and child_joint.attrib.get("type") == "fixed"
        and link.find("collision") is None
        and link.find("visual") is None
    )


def audit_urdf(path: Path) -> list[AuditIssue]:
    """Return deterministic issues without modifying *path*."""

    path = path.resolve(strict=True)
    robot = ET.parse(path).getroot()
    issues: list[AuditIssue] = []
    child_joints = {
        joint.find("child").attrib["link"]: joint
        for joint in robot.findall("joint")
        if joint.find("child") is not None
    }

    for element_type in ("link", "joint"):
        for element in robot.findall(element_type):
            name = element.attrib["name"]
            if not _VALID_NAME.fullmatch(name):
                issues.append(
                    AuditIssue(
                        "error",
                        _component(name),
                        element_type,
                        name,
                        "invalid_name",
                        "名称包含 USD prim 不支持的字符；Isaac 导入时会静默改名。",
                    )
                )

    for link in robot.findall("link"):
        name = link.attrib["name"]
        inertial = link.find("inertial")
        if inertial is None:
            if not _is_frame_only(link, child_joints.get(name)):
                issues.append(
                    AuditIssue(
                        "warning",
                        _component(name),
                        "link",
                        name,
                        "missing_inertial",
                        "非纯坐标 frame 的 link 缺少 inertial。",
                    )
                )
            continue
        mass = inertial.find("mass")
        inertia = inertial.find("inertia")
        if mass is None or not math.isfinite(_float_attr(mass, "value")) or _float_attr(mass, "value") <= 0:
            issues.append(
                AuditIssue(
                    "error",
                    _component(name),
                    "link",
                    name,
                    "invalid_mass",
                    "质量必须是有限正数。",
                )
            )
        if inertia is None or not _inertia_is_positive_definite(inertia):
            issues.append(
                AuditIssue(
                    "error",
                    _component(name),
                    "link",
                    name,
                    "invalid_inertia",
                    "惯量矩阵不是正定矩阵，PhysX 中会形成奇异刚体。",
                )
            )

    for joint in robot.findall("joint"):
        if joint.attrib.get("type") != "revolute":
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        if math.isclose(_float_attr(limit, "lower"), _float_attr(limit, "upper"), abs_tol=1e-12):
            name = joint.attrib["name"]
            issues.append(
                AuditIssue(
                    "warning",
                    _component(name),
                    "joint",
                    name,
                    "locked_revolute",
                    "revolute 上下限相等；应烘焙锁定角度后改为 fixed。",
                )
            )

    for mesh in robot.findall(".//mesh"):
        filename = mesh.attrib.get("filename", "")
        if filename.startswith("package://"):
            continue
        mesh_path = path.parent / filename
        if not mesh_path.is_file():
            issues.append(
                AuditIssue(
                    "error",
                    "资产路径",
                    "mesh",
                    filename,
                    "missing_mesh",
                    f"相对 URDF 的 mesh 文件不存在：{mesh_path}",
                )
            )

    return issues
