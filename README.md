# colmapcut_recon

[中文](#中文说明) · [English](#english-readme)

## 中文说明

一套面向植物的、可复现的三维重建流水线骨架。项目计划覆盖相机采集、视频抽帧与筛选、COLMAP 相机标定和稀疏重建、SAM/SegmentAnything3D 植物分割、跨视角点云过滤、真实尺度与坐标系对齐、高斯重建、背景清理，以及 PLY / ParticleField USD / 可选代理 mesh 导出。

> 当前状态：无遮罩路径“视频抽帧 → COLMAP → AprilTag 米制对齐 → 3DGRUT → 后训练背景/地面/植物分离 → 地面碰撞 mesh → 果实 mesh/刚体 → NuRec USDZ”已经实跑并接入可恢复的一键入口。前景掩膜辅助训练、完整语义分割和 gsplat 训练仍是占位或独立试验。此仓库不会安装、复制或修改外部项目。

## 已跑通的无遮罩流水线

`fruit_tomato.mp4` 的场景配置位于 `configs/scenes/fruit_tomato.yaml`。以下命令从视频运行到仿真 USDZ；默认开启恢复模式，已有且完整的阶段会被跳过：

```bash
uv run --extra alignment python scripts/run_pipeline.py \
  --video /home/linzz/Desktop/simple_photo_capture/video/fruit_tomato.mp4
```

使用 `--no-resume` 可要求各阶段重新运行，但各适配器会保护已有输出并拒绝覆盖。最终资产为 `data/scenes/fruit_tomato/11_simulation_asset/fruit_tomato_simulation.usdz`，总清单写入场景数据目录的 `pipeline_manifest.json`。外部项目和解释器路径统一在 `configs/tools.local.yaml` 修改；空间裁剪、地面网格与导出参数在 `configs/simulation/fruit_tomato_asset.yaml` 修改。

Isaac Sim 5.x 使用包内 NuRec volume 渲染静态植物/地面，并使用标准 USD Physics 碰撞。Genesis 可使用闭合地面 mesh、果实三角 mesh、质量和碰撞代理；NVIDIA NuRec 是专用视觉 schema，Genesis 通常会忽略该高斯视觉层。当前 fruit 示例只有 100 次训练，并用成熟果实颜色引导生成 mesh，只用于验证资产链路，不能代表最终分割精度。

## 仓库内可运行示例

`examples/roman_tomato2/input/roman_tomato2_example.mp4` 是从完整 4K/60 fps 手机视频压缩并清除 GPS/拍摄设备元数据后的公开示例：保留完整 94.995 秒相机轨迹，编码为 1920×1080、29.97 fps H.264，无音频，大小 20,304,540 字节（约 19.36 MiB）。它低于 [GitHub 网页单文件 25 MiB 的上传限制](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository)，不需要 Git LFS。

```bash
uv run --extra alignment python scripts/run_pipeline.py \
  --scene-config configs/scenes/roman_tomato2_example.yaml \
  --video examples/roman_tomato2/input/roman_tomato2_example.mp4
```

本机已经使用该压缩文件完成十阶段试运行：95 张采样图像、52 张图像的 COLMAP 主模型、AprilTag 米制对齐、100 次 3DGRUT 冒烟训练、闭合地面碰撞 mesh、1 个颜色引导的果实刚体，以及通过 OpenUSD 验证的组合 USDZ。派生数据和训练结果仍由 `.gitignore` 排除，克隆仓库后可从示例视频重新生成。压缩参数、SHA-256、逐阶段命令和实跑指标见 [examples/roman_tomato2/README.md](examples/roman_tomato2/README.md)。

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

当前无遮罩示例的主要外部依赖及官方仓库如下：

| 依赖 | 用途 | GitHub |
| --- | --- | --- |
| FFmpeg/ffprobe | 视频压缩、探测与抽帧 | [FFmpeg/FFmpeg](https://github.com/FFmpeg/FFmpeg) |
| COLMAP/pycolmap | 特征、匹配、稀疏重建与模型变换 | [colmap/colmap](https://github.com/colmap/colmap) |
| OpenCV | AprilTag 36h11 检测 | [opencv/opencv](https://github.com/opencv/opencv) |
| NVIDIA 3DGRUT | 3DGUT 训练、PLY 与 NuRec 导出 | [nv-tlabs/3dgrut](https://github.com/nv-tlabs/3dgrut) |
| Open3D | 地面碰撞 mesh 验证 | [isl-org/Open3D](https://github.com/isl-org/Open3D) |
| OpenUSD | USD/USDZ 编写与验证 | [PixarAnimationStudios/OpenUSD](https://github.com/PixarAnimationStudios/OpenUSD) |

SAM、SegmentAnything3D/SegAnyGAussians、SuGaR、gsplat、Isaac Sim 与 Genesis 属于可选分割、网格实验、替代训练后端或下游仿真工具；完整链接和“必需/可选/占位”状态也列在外部依赖文档中。各外部项目保持独立安装，本机地址只在 `configs/tools.local.yaml` 修改。

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
| `08_asset_separation` | 训练后的高斯 PLY | 植物、地面、丢弃背景和组合高斯 |
| `09_ground_collision` | 地面高斯 | 闭合、流形、米制地面 PLY/OBJ mesh |
| `10_fruit_entities` | 植物高斯、SAGA 掩膜或颜色种子 | 果实 mesh、刚体层、无果实静态高斯 |
| `11_simulation_asset` | 静态高斯、地面 mesh、果实实体 | Isaac/Genesis 仿真 USDZ 与验证清单 |

`00_capture` 中的原始数据视为只读。任何派生结果必须进入后续编号目录；训练产物不得写回 `data`。当前为避免失效链接，两个数据集目录都是普通空目录，未来可改成指向同一来源的符号链接。

## 使用约定

- `configs/pipeline_no_masks.yaml` 定义当前已实现的无遮罩阶段顺序，示例场景入口位于 `configs/scenes/fruit_tomato.yaml`。
- `scripts/` 是 CLI 入口，实现位于 `src/colmapcut_recon/`。
- `data/` 保存输入及确定性的中间数据，`runs/` 保存训练结果，`outputs/` 保存交付资产。
- 未列入 `configs/pipeline_no_masks.yaml` 的占位脚本不能视为已实现阶段。

## 机器人 URDF

用于 Isaac Sim/机器人仿真的组合移动机械臂已经包含在仓库中：

- [`assets/robots/mobile_manipulator/urdf/combined_mobile.urdf`](assets/robots/mobile_manipulator/urdf/combined_mobile.urdf)：当前修复后的组合机器人入口。
- [`assets/robots/mobile_manipulator/urdf/combined_source.urdf`](assets/robots/mobile_manipulator/urdf/combined_source.urdf)：原始展开 URDF 的只读副本。
- [`assets/robots/mobile_manipulator/urdf/combined_source.urdf.xacro`](assets/robots/mobile_manipulator/urdf/combined_source.urdf.xacro)：组合 Xacro 来源。
- `assets/robots/mobile_manipulator/urdf/uie_description/`：Lite6 机械臂的 visual/collision mesh。
- `assets/robots/mobile_manipulator/urdf/terrasentia_gazebo/`：TerraSentia 底盘及传感器 mesh，作为 Git 子模块引用。

克隆后需要初始化子模块，才能取得底盘的完整 mesh 依赖：

```bash
git submodule update --init --recursive
```

机器人结构、Isaac Sim 导入约定及路径配置见 [assets/robots/mobile_manipulator/README.md](assets/robots/mobile_manipulator/README.md)。

## 已实现：现有高斯的地面资产组装

`scripts/12_assemble_ground_asset.py` 可以处理已经完成米制对齐的高斯场景。默认的 `adaptive_heightfield` 方法只限制 XY 范围，不使用全局 Z 厚度：它按网格估计连续局部地面高度，通过最大坡度传播和填洞适应起伏，并结合局部重建带宽与每个高斯的旋转/尺度判断高斯是否接触地面。植物和地面在组合时执行精确 XYZ 并集去重，不再预先删除整个分割集合。旧的固定厚度行为仍可用 `--ground-method slab` 复现。所有外部输入均只读，参数记录在 `configs/scenes/roman_tomato_02.yaml`。

## SuGaR 地面网格试验

`scripts/13_prepare_sugar_ground.py` 将 metric COLMAP 相机转换成 SuGaR 所需的 `cameras.json`，并建立 vanilla 3DGS 风格 checkpoint。`scripts/14_try_sugar_ground_mesh.sh` 提供依赖检查、高斯中心快速试验和完整相机表面采样入口。RTX 5090 环境、SuGaR 补丁、运行参数及当前 Mesh 质量见 [docs/SUGAR_GROUND_MESH.md](docs/SUGAR_GROUND_MESH.md)。

## SAGA 番茄实例与物理实体

SAGA 的 CUDA 13 兼容运行时、稀疏多视角 SAM masks、自动番茄特征掩码、实例拆分及 Isaac Sim 刚体 USD 流程见 [docs/SAGA_TOMATO_ENTITIES.md](docs/SAGA_TOMATO_ENTITIES.md)。

## English README

`colmapcut_recon` is a reproducible plant reconstruction pipeline covering video frame sampling, COLMAP sparse reconstruction, SAM/SegmentAnything3D segmentation, cross-view point filtering, metric AprilTag alignment, Gaussian reconstruction, background cleanup, and PLY/USD/simulation-asset export.

The implemented mask-free path—video sampling, COLMAP, metric AprilTag alignment, 3DGRUT, post-training scene separation, ground collision meshing, fruit mesh/rigid-body generation, and NuRec USDZ composition—has been exercised end to end. Mask-assisted training, complete semantic segmentation, and gsplat training remain placeholders or separate experiments.

Run the resumable mask-free pipeline with:

```bash
uv run --extra alignment python scripts/run_pipeline.py \
  --video /home/linzz/Desktop/simple_photo_capture/video/fruit_tomato.mp4
```

Completed stages are skipped by default. Pass `--no-resume` to request a fresh execution; existing-output guards still prevent accidental overwrites. The final package is `data/scenes/fruit_tomato/11_simulation_asset/fruit_tomato_simulation.usdz`. Edit machine-local external paths in `configs/tools.local.yaml`, and edit crop, mesh, and export parameters in `configs/simulation/fruit_tomato_asset.yaml`.

Isaac Sim 5.x renders the packaged NuRec volume and uses the standard USD Physics colliders. Genesis can consume the standard closed ground mesh, fruit triangle meshes, masses, and collision proxies, but normally ignores the NVIDIA-specific NuRec Gaussian visual layer. The checked-in fruit smoke configuration uses only 100 training iterations and colour-guided ripe-fruit bootstrapping; it validates the pipeline rather than final segmentation accuracy.

### Repository example

`examples/roman_tomato2/input/roman_tomato2_example.mp4` is a public example made from the full 4K/60 fps phone capture with GPS and device metadata removed. It retains the complete 94.995-second camera path at 1920×1080, 29.97 fps H.264, has no audio, and is 20,304,540 bytes (about 19.36 MiB), below [GitHub's 25 MiB browser-upload limit](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository), so ordinary Git can track it without Git LFS.

```bash
uv run --extra alignment python scripts/run_pipeline.py \
  --scene-config configs/scenes/roman_tomato2_example.yaml \
  --video examples/roman_tomato2/input/roman_tomato2_example.mp4
```

The exercised ten-stage run produced 95 sampled images, a 52-image primary COLMAP model, metric AprilTag alignment, a 100-iteration 3DGRUT smoke reconstruction, a closed ground collider, one colour-bootstrapped fruit rigid body, and a combined USDZ that passed OpenUSD validation. Generated data and runs remain ignored and can be reproduced from the committed video. See the [bilingual example guide](examples/roman_tomato2/README.md) for compression settings, SHA-256, stage-by-stage commands, and measured results.

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

### Robot URDF

The combined mobile manipulator used for Isaac Sim and robot simulation is included in the repository:

- [`assets/robots/mobile_manipulator/urdf/combined_mobile.urdf`](assets/robots/mobile_manipulator/urdf/combined_mobile.urdf): current repaired combined-robot entry point.
- [`assets/robots/mobile_manipulator/urdf/combined_source.urdf`](assets/robots/mobile_manipulator/urdf/combined_source.urdf): read-only copy of the original expanded URDF.
- [`assets/robots/mobile_manipulator/urdf/combined_source.urdf.xacro`](assets/robots/mobile_manipulator/urdf/combined_source.urdf.xacro): combined Xacro source.
- `assets/robots/mobile_manipulator/urdf/uie_description/`: Lite6 visual and collision meshes.
- `assets/robots/mobile_manipulator/urdf/terrasentia_gazebo/`: TerraSentia chassis and sensor meshes, referenced as a Git submodule.

Initialize the submodule after cloning so all chassis mesh dependencies are present:

```bash
git submodule update --init --recursive
```

See [the robot asset guide](assets/robots/mobile_manipulator/README.md) for structure, Isaac Sim import conventions, and path configuration.

### External tools and paths

Third-party repositories remain independently installed. Edit machine-local repository, executable, and Python paths in `configs/tools.local.yaml`; use `configs/tools.example.yaml` as the committed template. Python dependency constraints live in `pyproject.toml`, with resolved versions in `uv.lock`.

See the bilingual [external dependency and path guide](docs/EXTERNAL_DEPENDENCIES.md) for the complete library inventory, configuration keys, and specialist workflow exceptions.

The main dependencies for the current mask-free example are [FFmpeg](https://github.com/FFmpeg/FFmpeg), [COLMAP/pycolmap](https://github.com/colmap/colmap), [OpenCV](https://github.com/opencv/opencv), [NVIDIA 3DGRUT](https://github.com/nv-tlabs/3dgrut), [Open3D](https://github.com/isl-org/Open3D), and [OpenUSD](https://github.com/PixarAnimationStudios/OpenUSD). SAM/SegmentAnything3D/SegAnyGAussians, SuGaR, gsplat, Isaac Sim, and Genesis are optional segmentation, meshing, alternative-backend, or downstream simulation projects. Their links and implemented/optional/placeholder status are listed in the dependency guide. Edit all machine-local installation paths only in `configs/tools.local.yaml`.

See [docs/README.md](docs/README.md) for the bilingual documentation index.

### Implemented integrations

- Deterministic ffprobe/FFmpeg video sampling: [docs/VIDEO_FRAME_SAMPLING.md](docs/VIDEO_FRAME_SAMPLING.md)
- Staged COLMAP and external 3DGRUT adapters: [docs/COLMAP_3DGRUT_INTEGRATION.md](docs/COLMAP_3DGRUT_INTEGRATION.md)
- Coplanar-AprilTag metric alignment: [docs/APRILTAG_METRIC_ALIGNMENT.md](docs/APRILTAG_METRIC_ALIGNMENT.md)
- SAGA tomato instances and Isaac Sim entities: [docs/SAGA_TOMATO_ENTITIES.md](docs/SAGA_TOMATO_ENTITIES.md)
- SuGaR ground-mesh experiment: [docs/SUGAR_GROUND_MESH.md](docs/SUGAR_GROUND_MESH.md)
