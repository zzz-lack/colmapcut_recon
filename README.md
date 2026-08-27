# colmapcut_recon

[中文](#中文说明) · [English](#english-readme)

## 中文说明

一套面向植物的、可复现的三维重建流水线骨架。项目计划覆盖相机采集、视频抽帧与筛选、COLMAP 相机标定和稀疏重建、SAM/SegmentAnything3D 植物分割、跨视角点云过滤、真实尺度与坐标系对齐、高斯重建、背景清理，以及 PLY / ParticleField USD / 可选代理 mesh 导出。

> 当前状态：大部分流水线仍是占位骨架；现有高斯的自适应地面提取与植物/地面组合已经实现。此仓库不会安装、复制或修改 COLMAP、SegmentAnything3D、3DGRUT 或 gsplat。

## 第一版处理顺序

1. 将照片或视频写入 `00_capture`，并记录采集元数据。
2. 从视频抽取并筛选多视角帧，写入 `01_frames`。
3. 使用原始筛选图像运行 COLMAP，结果写入 `02_colmap_full`。
4. 使用 SAM/SegmentAnything3D 生成并细化二维植物掩膜，写入 `03_masks`。
5. 通过多视角掩膜投票删除稀疏点云背景点，写入 `04_colmap_foreground`。
6. 估计尺度、轴向和原点，并将统一相似变换应用到点云和相机位姿，写入 `05_alignment`。
7. 将植物前景与固定纯色背景合成，写入 `06_composite`。
8. 由同一份合成图像和对齐后的 COLMAP 模型分别准备 3DGRUT 与 gsplat/3DGUT 数据集，写入 `07_datasets`。
9. 训练输出只写入 `runs/<scene>/<backend>`；第一版 3DGRUT 不加载 loss mask，以避免额外显存占用。
10. 清理残余背景高斯，并将最终 PLY、USD、预览和指标写入 `outputs/<scene>`。

统一相似变换约定为：

```text
world_point = scale * rotation * colmap_point + translation
```

## 外部工具

外部项目保持独立安装。通用路径模板位于 `configs/tools.example.yaml`；本机路径位于被 Git 忽略的 `configs/tools.local.yaml`。本项目的 `pyproject.toml` 仅描述管线自身，不会把外部项目安装为依赖。gsplat 当前未配置，本机配置中的 `repository` 与 `python` 均为 `null`。

完整的外部库清单、Python 包分组以及每类路径应该在哪个文件修改，见中英双语文档 [docs/EXTERNAL_DEPENDENCIES.md](docs/EXTERNAL_DEPENDENCIES.md)。

全部说明文件的中英双语索引见 [docs/README.md](docs/README.md)。

COLMAP 与 3DGRUT 已通过参数安全的适配脚本接入：COLMAP 分阶段生成数据库和稀疏模型，3DGRUT 数据适配器以符号链接组装标准 `images/ + sparse/0/` 输入，训练结果严格写入 `runs/`。目录契约、配置项、预检与命令示例见 [docs/COLMAP_3DGRUT_INTEGRATION.md](docs/COLMAP_3DGRUT_INTEGRATION.md)。

`05_alignment` 已接入基于共面 AprilTag 的米制定标与坐标系建立：多视角三角化 Tag 角点，以黑框实测边长确定尺度，以 Tag 平面和指定中心连线确定右手坐标轴，并统一变换稀疏点、相机、rig 和 frame。详见 [docs/APRILTAG_METRIC_ALIGNMENT.md](docs/APRILTAG_METRIC_ALIGNMENT.md)。

`01_frames` 已接入基于 ffprobe/ffmpeg 的确定性视频采样，支持按时间、每 N 帧或固定目标数量采样，并记录输出图片到原视频帧号和时间戳的完整映射。详见 [docs/VIDEO_FRAME_SAMPLING.md](docs/VIDEO_FRAME_SAMPLING.md)。

## 数据阶段

| 阶段 | 输入 | 输出 |
| --- | --- | --- |
| `00_capture` | 相机设备 | 原始视频/照片、采集元数据 |
| `01_frames` | `00_capture/videos` 或原始照片 | 经过抽取与筛选的多视角图像 |
| `02_colmap_full` | `01_frames/images` | 完整 COLMAP 稀疏模型和日志 |
| `03_masks` | `01_frames/images` | 原始及细化植物掩膜 |
| `04_colmap_foreground` | 完整稀疏模型、细化掩膜 | 过滤背景点后的 COLMAP 模型 |
| `05_alignment` | 前景 COLMAP 模型、尺度/轴向参考 | 相似变换和已对齐模型 |
| `06_composite` | 原图、细化掩膜 | 固定纯色背景合成图像 |
| `07_datasets` | 合成图像、已对齐 COLMAP 模型 | 两个后端共享来源的数据集副本 |
| `runs` | 准备好的训练数据集 | 3DGRUT 与 gsplat 训练产物 |
| `outputs` | 原始训练产物 | 清理后高斯、PLY、USD、预览、指标 |

`00_capture` 中的原始数据视为只读。任何派生结果必须进入后续编号目录；训练产物不得写回 `data`。当前为避免失效链接，两个数据集目录都是普通空目录，未来可改成指向同一来源的符号链接。

## 使用约定

- `configs/pipeline.yaml` 定义阶段顺序，场景入口位于 `configs/scenes/plant_001.yaml`。
- `scripts/` 仅作为 CLI 入口，未来调用 `src/colmapcut_recon/` 中的实现。
- `data/` 保存输入及确定性的中间数据，`runs/` 保存训练结果，`outputs/` 保存交付资产。
- 在实现算法前，不要把这些占位脚本当作可运行流水线。

## 已实现：现有高斯的地面资产组装

`scripts/12_assemble_ground_asset.py` 可以处理已经完成米制对齐的高斯场景。默认的 `adaptive_heightfield` 方法只限制 XY 范围，不使用全局 Z 厚度：它按网格估计连续局部地面高度，通过最大坡度传播和填洞适应起伏，并结合局部重建带宽与每个高斯的旋转/尺度判断高斯是否接触地面。植物和地面在组合时执行精确 XYZ 并集去重，不再预先删除整个分割集合。旧的固定厚度行为仍可用 `--ground-method slab` 复现。所有外部输入均只读，参数记录在 `configs/scenes/roman_tomato_02.yaml`。

## SuGaR 地面网格试验

`scripts/13_prepare_sugar_ground.py` 将 metric COLMAP 相机转换成 SuGaR 所需的 `cameras.json`，并建立 vanilla 3DGS 风格 checkpoint。`scripts/14_try_sugar_ground_mesh.sh` 提供依赖检查、高斯中心快速试验和完整相机表面采样入口。RTX 5090 环境、SuGaR 补丁、运行参数及当前 Mesh 质量见 [docs/SUGAR_GROUND_MESH.md](docs/SUGAR_GROUND_MESH.md)。

## SAGA 番茄实例与物理实体

SAGA 的 CUDA 13 兼容运行时、稀疏多视角 SAM masks、自动番茄特征掩码、实例拆分及 Isaac Sim 刚体 USD 流程见 [docs/SAGA_TOMATO_ENTITIES.md](docs/SAGA_TOMATO_ENTITIES.md)。

## English README

`colmapcut_recon` is a reproducible plant reconstruction pipeline covering video frame sampling, COLMAP sparse reconstruction, SAM/SegmentAnything3D segmentation, cross-view point filtering, metric AprilTag alignment, Gaussian reconstruction, background cleanup, and PLY/USD/simulation-asset export.

The coordinate transform convention is:

```text
world_point = scale * rotation * colmap_point + translation
```

### Pipeline stages

1. Store immutable photos or videos and capture metadata in `00_capture`.
2. sample multi-view frames into `01_frames` with a source-frame/timestamp manifest.
3. Run COLMAP and write its database, sparse model, logs, and manifest to `02_colmap_full`.
4. Generate and refine 2D plant masks in `03_masks`.
5. Filter sparse background points into `04_colmap_foreground`.
6. Establish metric scale, origin, and right-handed axes in `05_alignment`.
7. Composite the plant over a fixed background in `06_composite`.
8. Stage aligned COLMAP inputs for 3DGRUT/gsplat in `07_datasets`.
9. Keep all training outputs under `runs/<scene>/<backend>`.
10. Write cleaned Gaussians, PLY/USD assets, previews, and metrics under `outputs/<scene>`.

Raw capture data is read-only. Deterministic intermediate data belongs under `data`, training artifacts under `runs`, and deliverable assets under `outputs`.

### External tools and paths

Third-party repositories remain independently installed. Edit machine-local repository, executable, and Python paths in `configs/tools.local.yaml`; use `configs/tools.example.yaml` as the committed template. Python dependency constraints live in `pyproject.toml`, with resolved versions in `uv.lock`.

See the bilingual [external dependency and path guide](docs/EXTERNAL_DEPENDENCIES.md) for the complete library inventory, configuration keys, and specialist workflow exceptions.

See [docs/README.md](docs/README.md) for the bilingual documentation index.

### Implemented integrations

- Deterministic ffprobe/FFmpeg video sampling: [docs/VIDEO_FRAME_SAMPLING.md](docs/VIDEO_FRAME_SAMPLING.md)
- Staged COLMAP and external 3DGRUT adapters: [docs/COLMAP_3DGRUT_INTEGRATION.md](docs/COLMAP_3DGRUT_INTEGRATION.md)
- Coplanar-AprilTag metric alignment: [docs/APRILTAG_METRIC_ALIGNMENT.md](docs/APRILTAG_METRIC_ALIGNMENT.md)
- SAGA tomato instances and Isaac Sim entities: [docs/SAGA_TOMATO_ENTITIES.md](docs/SAGA_TOMATO_ENTITIES.md)
- SuGaR ground-mesh experiment: [docs/SUGAR_GROUND_MESH.md](docs/SUGAR_GROUND_MESH.md)
