# SuGaR 地面网格试验 / SuGaR Ground-Mesh Experiment

[中文详细说明](#本机布局) · [English summary](#english-summary)

> 外部依赖与路径修改 / External dependencies and path configuration: [EXTERNAL_DEPENDENCIES.md](EXTERNAL_DEPENDENCIES.md)

## English summary

This experimental workflow adapts metric COLMAP cameras and ground Gaussians into the vanilla-3DGS checkpoint layout expected by SuGaR, then extracts a candidate ground mesh. Immutable images, cameras, and Gaussian inputs are linked rather than copied. The repository includes a local patch for configurable opacity filtering, true bounding-box background skipping, and an empty-background initialization bug.

Change the standard SuGaR repository and Python paths in `configs/tools.local.yaml`. The shell scripts additionally accept `SUGAR_ROOT`, `SUGAR_PYTHON`, and, for rasterizer installation, `SUGAR_PIP`. RTX 5090 requires a compatible recent PyTorch/CUDA environment and a rasterizer compiled for `sm_120`; see the Chinese operational details below.

## 本机布局

- SuGaR 仓库：`/home/linzz/Desktop/SuGaR`
- 固定测试提交：`7c10c4ae4a267dece512f5c7f40ed212a0a2ab44`
- Python：`/home/linzz/miniconda3/envs/sugar/bin/python`
- 数据适配包：`data/scenes/roman_tomato_02/09_sugar_ground`

适配包用符号链接引用约 2.2 GB 原图、722 MB metric COLMAP 模型和地面高斯，避免复制不可变输入。`checkpoint/cameras.json` 是从 metric COLMAP 文本模型确定性生成的。

RTX 5090 不能使用 SuGaR 官方锁定的 PyTorch 2.0.1/CUDA 11.8 环境。本机 `sugar` 环境从已有的 5090 兼容环境克隆，使用 PyTorch 2.12.0+cu130、源码编译的 PyTorch3D 0.7.9 和 Open3D 0.19.0。

从 SA3D 环境克隆时会继承要求 `mask` 参数的 SegmentAnything3D 光栅器，它与 SuGaR 原版接口不兼容。使用以下脚本替换并针对 RTX 5090 的 `sm_120` 重新编译：

```bash
scripts/15_install_sugar_rasterizer.sh
```

## SuGaR 补丁

当前地面高斯的激活 opacity 大多小于 SuGaR 写死的 `0.5`。补丁增加 `--opacity_threshold` 和真正裁掉 bbox 外部点的 `--skip_background` 参数，并修复没有背景点时 `decimated_o3d_bg_mesh` 未初始化的问题。

新克隆的 SuGaR 可以这样应用补丁：

```bash
git -C /home/linzz/Desktop/SuGaR apply --unidiff-zero \
  /home/linzz/Desktop/colmapcut_recon/patches/sugar_ground_mesh.patch
```

## 准备输入

```bash
cd /home/linzz/Desktop/colmapcut_recon

/home/linzz/3dgrut/.venv/bin/python scripts/13_prepare_sugar_ground.py \
  --adapter-root data/scenes/roman_tomato_02/09_sugar_ground \
  --images-dir /home/linzz/Desktop/realplantrecon_romantomato2/images \
  --sparse-dir /home/linzz/Desktop/realplantrecon_romantomato2/metric_scene/sparse \
  --ground-ply data/scenes/roman_tomato_02/08_asset_assembly/ground_gaussians.ply \
  --overwrite
```

## 检查与运行

```bash
scripts/14_try_sugar_ground_mesh.sh --check

# 快速验证：高斯中心 + Poisson
scripts/14_try_sugar_ground_mesh.sh --run-centers

# SuGaR 相机表面采样模式，明显更慢
scripts/14_try_sugar_ground_mesh.sh --run
```

可通过环境变量覆盖参数：

```bash
SUGAR_OPACITY_THRESHOLD=0.01 \
SUGAR_DECIMATION_TARGET=20000 \
SUGAR_BBOX_MIN='(-0.5,-0.5,-0.25)' \
SUGAR_BBOX_MAX='(0.5,0.5,0.25)' \
scripts/14_try_sugar_ground_mesh.sh --run-centers
```

最后一组范围对应 1×1 米地面；默认 `[-1,1]` 对应当前已经提取的 2×2 米地面。

## 当前试验结果

快速模式生成 `sugarmesh_vanilla3dgs_poissoncenters_decim20000.ply`：

- 19,618 个顶点、22,190 个三角形；
- 31 个连通分量，最大分量包含 21,679 个三角形；
- 当前不是 watertight，存在自相交且不可定向；
- 因此它证明 SuGaR 链路可运行，但尚不能直接作为最终碰撞网格。

完整相机表面采样模式也已成功生成 `sugarmesh_vanilla3dgs_level03_decim20000.ply`：

- 1,357,921 个采样表面点，其中 1,355,150 个位于 bbox 内；
- 14,618 个顶点、20,000 个三角形；
- 483 个连通分量，最大分量包含 9,603 个三角形；
- 仍然存在自相交且不是 watertight，需要进一步清理后才能作为碰撞网格。
