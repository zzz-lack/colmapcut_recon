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

`pytest` 只用于验证。地面碰撞阶段通过 `sugar.python` 对应环境使用 SciPy 和 Open3D；高斯分离、果实 mesh 与 USDZ 阶段通过 `threedgrut.python` 对应环境使用 NumPy、plyfile、OpenUSD/`pxr` 和 3DGRUT NuRec 导出器。PyTorch、CUDA、PyTorch3D 等仍由各外部环境管理。

### 专项流程的其他路径

以下路径不是“外部库安装地址”，因此不放进 `tools.local.yaml`：

| 路径类型 | 修改文件/入口 |
| --- | --- |
| SAGA 输入 PLY、mask 和实体输出 | `configs/segmentation/saga_tomato.toml` |
| Isaac Sim 场景、机器人 USD/URDF | `configs/simulation/tomato_orbit.toml` |
| 背景裁剪、地面 mesh、果实与 USDZ 参数 | `configs/simulation/*_asset.yaml`、`configs/segmentation/*_entities.toml` |
| 场景数据根目录与阶段配置 | `configs/scenes/*.yaml` |
| AprilTag 尺寸和方向 | `configs/alignment/*.yaml` |
| 视频采样参数 | `configs/preprocessing/video_sampling.yaml` |

仍使用 shell 的专项脚本允许环境变量覆盖：

- `scripts/14_try_sugar_ground_mesh.sh`：`SUGAR_ROOT`、`SUGAR_PYTHON`。
- `scripts/15_install_sugar_rasterizer.sh`：`SUGAR_ROOT`、`SUGAR_PYTHON`、`SUGAR_PIP`。
- `scripts/18_run_saga_tomato.sh`：当前在脚本顶部设置 `PROJECT_ROOT`、`SAGA_ROOT`、`SAGA_PYTHON`、`SAM_CHECKPOINT`；运行另一台机器时应修改这些变量。
- Isaac Sim 启动器路径在运行命令中显式给出，例如 `/path/to/isaacsim/python.sh`。

### 外部库清单

| 状态 | 外部项目 | 用途与调用位置 | 官方 GitHub |
| --- | --- | --- | --- |
| 当前示例必需 | FFmpeg/ffprobe | 视频压缩、探测、抽帧；`scripts/01_extract_frames.py` | [FFmpeg/FFmpeg](https://github.com/FFmpeg/FFmpeg) |
| 当前示例必需 | COLMAP/pycolmap | 稀疏重建与 Sim3 模型变换；脚本 02、05 | [colmap/colmap](https://github.com/colmap/colmap) |
| 当前示例必需 | OpenCV + contrib | AprilTag/ArUco 检测；脚本 05 | [opencv/opencv](https://github.com/opencv/opencv)、[opencv/opencv_contrib](https://github.com/opencv/opencv_contrib) |
| 当前示例必需 | NVIDIA 3DGRUT | 3DGUT 训练、PLY、NuRec USDZ；脚本 08、11 | [nv-tlabs/3dgrut](https://github.com/nv-tlabs/3dgrut) |
| 当前示例必需 | Open3D | 地面碰撞 mesh 拓扑验证；脚本 16 | [isl-org/Open3D](https://github.com/isl-org/Open3D) |
| 当前示例必需 | OpenUSD/`pxr` | USD/USDZ 编写、组合与验证；脚本 11、21 | [PixarAnimationStudios/OpenUSD](https://github.com/PixarAnimationStudios/OpenUSD) |
| 可选分割 | Segment Anything | 2D 掩膜基础模型 | [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) |
| 可选分割 | SegmentAnything3D | NeRF/多视角分割试验 | [Jumpat/SegmentAnythingin3D](https://github.com/Jumpat/SegmentAnythingin3D) |
| 可选分割 | SegAnyGAussians/SAGA | Gaussian 特征与实例掩膜 | [Jumpat/SegAnyGAussians](https://github.com/Jumpat/SegAnyGAussians) |
| 可选网格 | SuGaR | Gaussian 到表面 mesh 试验 | [Anttwo/SuGaR](https://github.com/Anttwo/SuGaR) |
| 尚未接入训练 | gsplat | 可选 Gaussian 后端，占位 | [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) |
| 外部环境依赖 | PyTorch/PyTorch3D/CUDA | 3DGRUT、SAGA、SuGaR 的 GPU 环境 | [pytorch/pytorch](https://github.com/pytorch/pytorch)、[facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d) |
| 下游仿真 | Isaac Sim | NuRec 视觉层与 USD Physics | [isaac-sim/IsaacSim](https://github.com/isaac-sim/IsaacSim) |
| 下游仿真 | Genesis | 标准 mesh、质量、刚体和碰撞代理 | [Genesis-Embodied-AI/Genesis](https://github.com/Genesis-Embodied-AI/Genesis) |

NumPy、SciPy、plyfile 和 PyYAML 是 Python 包依赖；主环境约束在 `pyproject.toml`，3DGRUT/SuGaR/SAGA 专用环境内的版本由各外部项目管理。逐阶段独立命令见 `examples/roman_tomato2/README.md`。

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

`pytest` is verification-only. The ground-collision stage uses SciPy and Open3D from the environment configured as `sugar.python`. Gaussian separation, fruit meshing, and USDZ export use NumPy, plyfile, OpenUSD/`pxr`, and the 3DGRUT NuRec exporter from `threedgrut.python`. PyTorch, CUDA, and PyTorch3D remain managed by their external environments.

### Other workflow-specific paths

These are data or asset paths rather than library installations, so they use separate configuration files:

| Path type | File/entry to edit |
| --- | --- |
| SAGA input PLY, masks, and entity output | `configs/segmentation/saga_tomato.toml` |
| Isaac Sim scene and robot USD/URDF assets | `configs/simulation/tomato_orbit.toml` |
| Background crop, ground mesh, fruit, and USDZ parameters | `configs/simulation/*_asset.yaml`, `configs/segmentation/*_entities.toml` |
| Scene data root and stage configurations | `configs/scenes/*.yaml` |
| AprilTag sizes and orientation | `configs/alignment/*.yaml` |
| Video sampling parameters | `configs/preprocessing/video_sampling.yaml` |

Shell-based specialist workflows still expose path variables:

- `scripts/14_try_sugar_ground_mesh.sh`: `SUGAR_ROOT`, `SUGAR_PYTHON`.
- `scripts/15_install_sugar_rasterizer.sh`: `SUGAR_ROOT`, `SUGAR_PYTHON`, `SUGAR_PIP`.
- `scripts/18_run_saga_tomato.sh`: edit `PROJECT_ROOT`, `SAGA_ROOT`, `SAGA_PYTHON`, and `SAM_CHECKPOINT` near the top when moving to another machine.
- Pass the Isaac Sim launcher explicitly in commands, for example `/path/to/isaacsim/python.sh`.

### External library inventory

| Status | External project | Purpose and caller | Official GitHub |
| --- | --- | --- | --- |
| Required by current example | FFmpeg/ffprobe | Video compression, probing, and sampling; script 01 | [FFmpeg/FFmpeg](https://github.com/FFmpeg/FFmpeg) |
| Required by current example | COLMAP/pycolmap | Sparse reconstruction and Sim3 model transform; scripts 02 and 05 | [colmap/colmap](https://github.com/colmap/colmap) |
| Required by current example | OpenCV + contrib | AprilTag/ArUco detection; script 05 | [opencv/opencv](https://github.com/opencv/opencv), [opencv/opencv_contrib](https://github.com/opencv/opencv_contrib) |
| Required by current example | NVIDIA 3DGRUT | 3DGUT training, PLY, and NuRec USDZ; scripts 08 and 11 | [nv-tlabs/3dgrut](https://github.com/nv-tlabs/3dgrut) |
| Required by current example | Open3D | Ground-collision topology validation; script 16 | [isl-org/Open3D](https://github.com/isl-org/Open3D) |
| Required by current example | OpenUSD/`pxr` | USD/USDZ authoring, composition, and validation; scripts 11 and 21 | [PixarAnimationStudios/OpenUSD](https://github.com/PixarAnimationStudios/OpenUSD) |
| Optional segmentation | Segment Anything | Foundation model for 2D masks | [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) |
| Optional segmentation | SegmentAnything3D | NeRF/multi-view segmentation experiments | [Jumpat/SegmentAnythingin3D](https://github.com/Jumpat/SegmentAnythingin3D) |
| Optional segmentation | SegAnyGAussians/SAGA | Gaussian features and instance masks | [Jumpat/SegAnyGAussians](https://github.com/Jumpat/SegAnyGAussians) |
| Optional meshing | SuGaR | Experimental Gaussian-to-surface mesh path | [Anttwo/SuGaR](https://github.com/Anttwo/SuGaR) |
| Training not yet integrated | gsplat | Optional Gaussian backend placeholder | [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) |
| External-environment dependency | PyTorch/PyTorch3D/CUDA | GPU runtimes for 3DGRUT, SAGA, and SuGaR | [pytorch/pytorch](https://github.com/pytorch/pytorch), [facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d) |
| Downstream simulator | Isaac Sim | NuRec visual layer and USD Physics | [isaac-sim/IsaacSim](https://github.com/isaac-sim/IsaacSim) |
| Downstream simulator | Genesis | Standard meshes, masses, rigid bodies, and collision proxies | [Genesis-Embodied-AI/Genesis](https://github.com/Genesis-Embodied-AI/Genesis) |

NumPy, SciPy, plyfile, and PyYAML are Python package dependencies. Main-environment constraints live in `pyproject.toml`; their versions inside the dedicated 3DGRUT, SuGaR, and SAGA environments remain managed by those projects. See `examples/roman_tomato2/README.md` for independently callable stage commands.
