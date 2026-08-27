# 外部依赖与路径配置 / External Dependencies and Path Configuration

[中文](#中文) · [English](#english)

## 中文

本仓库只管理重建流水线、输入输出契约和适配代码。大型外部项目与专用运行环境保持独立安装，主项目不会复制或修改它们。

### 修改外部库地址

标准外部工具的本机路径统一在以下文件中修改：

```text
configs/tools.local.yaml
```

该文件被 Git 忽略，适合保存本机绝对路径。可提交的字段模板位于：

```text
configs/tools.example.yaml
```

首次配置时复制模板，然后按本机安装位置修改：

```bash
cp configs/tools.example.yaml configs/tools.local.yaml
```

当前支持的标准路径键：

| 外部工具/库 | 用途 | `tools.local.yaml` 字段 |
| --- | --- | --- |
| COLMAP | 特征提取、匹配、稀疏重建 | `colmap.repository`、`colmap.executable`、`colmap.python` |
| FFmpeg/ffprobe | 视频探测与抽帧 | `ffmpeg.executable`、`ffmpeg.ffprobe` |
| SegmentAnything3D | SAM/SA3D 掩膜 | `segment_anything_3d.repository`、`segment_anything_3d.python` |
| NVIDIA 3DGRUT | 3DGRT/3DGUT 训练和导出 | `threedgrut.repository`、`threedgrut.python` |
| gsplat | 可选 Gaussian Splatting 后端 | `gsplat.repository`、`gsplat.python` |
| SuGaR | 地面高斯到网格的试验流程 | `sugar.repository`、`sugar.python` |

CLI 参数（例如 `--executable`、`--repository`、`--python`）优先于配置文件，可用于一次性覆盖。

### Python 包

Python 依赖版本在 `pyproject.toml` 修改，锁定结果在 `uv.lock`：

| 依赖组 | 外部包 | 用途 |
| --- | --- | --- |
| 核心 | PyYAML | YAML 配置读取 |
| `alignment` | NumPy、opencv-contrib-python-headless、pycolmap | AprilTag 检测、三角化和 COLMAP Sim3 变换 |
| `gaussians` | NumPy、plyfile、tomli（Python 3.10） | PLY、高斯后处理与旧版 TOML 支持 |

`pytest` 和 `scipy` 当前只在完整验证命令中临时加入，不是核心运行依赖。PyTorch、CUDA、PyTorch3D、Open3D 等由 3DGRUT、SAGA 或 SuGaR 的独立环境管理。

### 专项流程的其他路径

以下路径不是“外部库安装地址”，因此不放进 `tools.local.yaml`：

| 路径类型 | 修改文件/入口 |
| --- | --- |
| SAGA 输入 PLY、mask 和实体输出 | `configs/segmentation/saga_tomato.toml` |
| Isaac Sim 场景、机器人 USD/URDF | `configs/simulation/tomato_orbit.toml` |
| 场景数据根目录与阶段配置 | `configs/scenes/*.yaml` |
| AprilTag 尺寸和方向 | `configs/alignment/*.yaml` |
| 视频采样参数 | `configs/preprocessing/video_sampling.yaml` |

仍使用 shell 的专项脚本允许环境变量覆盖：

- `scripts/14_try_sugar_ground_mesh.sh`：`SUGAR_ROOT`、`SUGAR_PYTHON`。
- `scripts/15_install_sugar_rasterizer.sh`：`SUGAR_ROOT`、`SUGAR_PYTHON`、`SUGAR_PIP`。
- `scripts/18_run_saga_tomato.sh`：当前在脚本顶部设置 `PROJECT_ROOT`、`SAGA_ROOT`、`SAGA_PYTHON`、`SAM_CHECKPOINT`；运行另一台机器时应修改这些变量。
- Isaac Sim 启动器路径在运行命令中显式给出，例如 `/path/to/isaacsim/python.sh`。

### 外部库清单

流水线涉及：COLMAP、FFmpeg/ffprobe、OpenCV ArUco/AprilTag、pycolmap、NVIDIA 3DGRUT、gsplat、Segment Anything、SegmentAnything3D/SegAnyGAussians、SuGaR、PyTorch、CUDA、PyTorch3D、Open3D、Isaac Sim、OpenUSD/`pxr`、NumPy、SciPy、plyfile 和 PyYAML。并非每个阶段都需要全部依赖；请按对应文档安装。

## English

This repository owns the reconstruction pipeline, I/O contracts, and adapters. Large third-party repositories and specialized runtime environments remain independently installed; the project does not copy or modify them.

### Changing external library paths

Edit machine-local paths for standard external tools in:

```text
configs/tools.local.yaml
```

This file is Git-ignored so it can safely contain absolute paths. The committed template is:

```text
configs/tools.example.yaml
```

Create a local configuration with:

```bash
cp configs/tools.example.yaml configs/tools.local.yaml
```

Supported standard path keys are:

| External tool/library | Purpose | `tools.local.yaml` keys |
| --- | --- | --- |
| COLMAP | Feature extraction, matching, sparse reconstruction | `colmap.repository`, `colmap.executable`, `colmap.python` |
| FFmpeg/ffprobe | Video probing and frame extraction | `ffmpeg.executable`, `ffmpeg.ffprobe` |
| SegmentAnything3D | SAM/SA3D masks | `segment_anything_3d.repository`, `segment_anything_3d.python` |
| NVIDIA 3DGRUT | 3DGRT/3DGUT training and export | `threedgrut.repository`, `threedgrut.python` |
| gsplat | Optional Gaussian Splatting backend | `gsplat.repository`, `gsplat.python` |
| SuGaR | Experimental ground-Gaussian mesh extraction | `sugar.repository`, `sugar.python` |

CLI options such as `--executable`, `--repository`, and `--python` override the file for one-off runs.

### Python packages

Change Python dependency constraints in `pyproject.toml`; resolved versions are stored in `uv.lock`.

| Dependency group | Packages | Purpose |
| --- | --- | --- |
| Core | PyYAML | YAML configuration loading |
| `alignment` | NumPy, opencv-contrib-python-headless, pycolmap | AprilTag detection, triangulation, and COLMAP Sim3 transforms |
| `gaussians` | NumPy, plyfile, tomli on Python 3.10 | PLY/Gaussian processing and legacy TOML support |

`pytest` and `scipy` are currently added only by the full verification command. PyTorch, CUDA, PyTorch3D, and Open3D are managed by the independent 3DGRUT, SAGA, or SuGaR environments.

### Other workflow-specific paths

These are data or asset paths rather than library installations, so they use separate configuration files:

| Path type | File/entry to edit |
| --- | --- |
| SAGA input PLY, masks, and entity output | `configs/segmentation/saga_tomato.toml` |
| Isaac Sim scene and robot USD/URDF assets | `configs/simulation/tomato_orbit.toml` |
| Scene data root and stage configurations | `configs/scenes/*.yaml` |
| AprilTag sizes and orientation | `configs/alignment/*.yaml` |
| Video sampling parameters | `configs/preprocessing/video_sampling.yaml` |

Shell-based specialist workflows still expose path variables:

- `scripts/14_try_sugar_ground_mesh.sh`: `SUGAR_ROOT`, `SUGAR_PYTHON`.
- `scripts/15_install_sugar_rasterizer.sh`: `SUGAR_ROOT`, `SUGAR_PYTHON`, `SUGAR_PIP`.
- `scripts/18_run_saga_tomato.sh`: edit `PROJECT_ROOT`, `SAGA_ROOT`, `SAGA_PYTHON`, and `SAM_CHECKPOINT` near the top when moving to another machine.
- Pass the Isaac Sim launcher explicitly in commands, for example `/path/to/isaacsim/python.sh`.

### External library inventory

The overall pipeline uses COLMAP, FFmpeg/ffprobe, OpenCV ArUco/AprilTag, pycolmap, NVIDIA 3DGRUT, gsplat, Segment Anything, SegmentAnything3D/SegAnyGAussians, SuGaR, PyTorch, CUDA, PyTorch3D, Open3D, Isaac Sim, OpenUSD/`pxr`, NumPy, SciPy, plyfile, and PyYAML. Individual stages require only their documented subset.
