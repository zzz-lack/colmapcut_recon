# 文档索引 / Documentation Index

[中文](#中文) · [English](#english)

## 中文

所有项目自有说明文件均包含中文和英文。外部仓库、可执行程序和 Python 环境地址统一优先在 `configs/tools.local.yaml` 修改；可提交模板为 `configs/tools.example.yaml`。

| 文档 | 内容 |
| --- | --- |
| [外部依赖与路径配置](EXTERNAL_DEPENDENCIES.md) | 完整外部库清单、Python 依赖组和所有路径修改入口 |
| [视频采样](VIDEO_FRAME_SAMPLING.md) | FFmpeg/ffprobe 抽帧、时间戳和 manifest |
| [COLMAP 与 3DGRUT](COLMAP_3DGRUT_INTEGRATION.md) | 稀疏重建、数据适配和训练调用 |
| [AprilTag 米制对齐](APRILTAG_METRIC_ALIGNMENT.md) | 比例尺、原点和右手坐标轴 |
| [SAGA 番茄实体](SAGA_TOMATO_ENTITIES.md) | 掩膜、实例拆分和刚体 USD |
| [SuGaR 地面网格](SUGAR_GROUND_MESH.md) | 地面网格试验和 RTX 5090 环境 |
| [USD 仿真资产](USD_SIMULATION_ASSET.md) | NuRec/3DGS 和碰撞资产打包 |
| [机器人 URDF 审计](ROBOT_URDF_AUDIT.md) | 移动机械臂 URDF/PhysX 修复 |
| [番茄巡检仿真](TOMATO_ORBIT_SIMULATION.md) | Isaac Sim 场景和巡检流程 |

## English

Every project-owned documentation file contains both Chinese and English. Edit external repository, executable, and Python-environment paths primarily in `configs/tools.local.yaml`; the committed template is `configs/tools.example.yaml`.

| Document | Contents |
| --- | --- |
| [External dependencies and paths](EXTERNAL_DEPENDENCIES.md) | Complete library inventory, Python dependency groups, and every path-editing entry point |
| [Video sampling](VIDEO_FRAME_SAMPLING.md) | FFmpeg/ffprobe extraction, timestamps, and manifests |
| [COLMAP and 3DGRUT](COLMAP_3DGRUT_INTEGRATION.md) | Sparse reconstruction, dataset adaptation, and training invocation |
| [AprilTag metric alignment](APRILTAG_METRIC_ALIGNMENT.md) | Scale, origin, and right-handed axes |
| [SAGA tomato entities](SAGA_TOMATO_ENTITIES.md) | Masks, instance splitting, and rigid-body USD |
| [SuGaR ground mesh](SUGAR_GROUND_MESH.md) | Ground-mesh experiment and RTX 5090 environment |
| [USD simulation asset](USD_SIMULATION_ASSET.md) | NuRec/3DGS and collision packaging |
| [Robot URDF audit](ROBOT_URDF_AUDIT.md) | Mobile-manipulator URDF/PhysX repairs |
| [Tomato inspection simulation](TOMATO_ORBIT_SIMULATION.md) | Isaac Sim scene and inspection workflow |
