# 番茄巡检仿真的机器人 URDF 审计 / Robot URDF Audit

[中文概述](#中文概述) · [English details](#confirmed-problems-and-repairs)

> 外部依赖与路径修改 / External dependencies and path configuration: [EXTERNAL_DEPENDENCIES.md](EXTERNAL_DEPENDENCIES.md)

## 中文概述

本文记录移动机械臂 URDF 在 Isaac Sim/PhysX 中的结构审计与修复：补全奇异惯量、将锁死的腿部关节转换为 fixed joint、修正非法 USD 名称、校准轮径/轮距元数据，并增加末端相机坐标系。原始展开 URDF 保存在 `combined_source.urdf`，修复候选为 `combined_mobile.urdf`。

机器人 USD/URDF 与场景资产路径在 `configs/simulation/tomato_orbit.toml` 修改。Isaac Sim、OpenUSD/`pxr` 和 PhysX 由独立 Isaac Sim 安装提供；启动器路径通过运行命令显式指定。完整外部库与路径入口见上方双语依赖文档。

## English details

The supplied expanded URDF is preserved as
`assets/robots/mobile_manipulator/urdf/combined_source.urdf`. The repair candidate
is `combined_mobile.urdf` in the same directory.

## Confirmed problems and repairs

| Element | Subsystem | Finding | Repair |
| --- | --- | --- | --- |
| `link_base` | Lite6 arm base | Mass was 2.11 kg but all six inertia values were zero. | Recomputed COM and full positive-definite tensor from the watertight collision STL at 2.11 kg. |
| `base_footprint` | TerraSentia chassis root | Virtual root had no inertia; mobile import made it a singular root rigid body. | Added a small 1 g, `1e-6 kg m²` virtual-frame inertia. This is not a replacement for measured chassis inertia; `base_link` retains the physical chassis mass. |
| four `*_leg_joint` joints | TerraSentia suspension/legs | Revolute joints had equal lower and upper limits. | Converted them to fixed joints and baked the original `+/-0.52 rad` lock angle into each origin transform. |
| three `$*_zed2_*_joint` joints | Chassis camera mounts | `$` is not a valid USD prim-name character; the importer renamed these joints. | Replaced `$...` names with stable `..._mount_joint` names. |
| wheel controller constants | TerraSentia drive | Gazebo metadata used diameter 0.194 m and separation 0.4318 m, while the actual expanded collision geometry is diameter 0.18 m with lateral centers at approximately `+/-0.129573 m`. | Updated the repair candidate and simulation config to 0.18 m and 0.259146 m. The real robot should still be measured before controller calibration. |
| `eef_camera_frame` | Lite6 end effector | No end-effector camera frame existed. | Added a fixed frame 0.04 m along local `-Z`; the actual USD `Camera` prim is created at runtime. |

## Not treated as errors

- `center/left/right_zed2_camera_frame`, their optical frames, `link_eef`, and
  `eef_camera_frame` have no inertial blocks because they are fixed, geometry-free
  coordinate frames.
- The unusual `rear_left_wheel_joint` orientation is retained. When composed with
  the locked leg transform it makes all four wheel axes parallel in `base_link`.
- `arm_to_chassis_joint` remains at `xyz="0 0 0.25"`; its own comment says the mount
  must be physically measured, so changing it without hardware dimensions would be
  speculative.

## Remaining validation work

- Tune Lite6 joint drives with Isaac Sim Gain Tuner.
- Refine the coarse Lula collision spheres in Robot Description Editor.
- Measure the real wheel center spacing and arm mounting transform.
- Add a real gripper asset; the current robot still ends at `link_eef`.
