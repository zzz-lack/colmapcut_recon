# AprilTag 米制尺度与坐标系对齐 / AprilTag Metric Alignment

[中文详细说明](#坐标约定) · [English summary](#english-summary)

> 外部依赖与路径修改 / External dependencies and path configuration: [EXTERNAL_DEPENDENCIES.md](EXTERNAL_DEPENDENCIES.md)

## English summary

`scripts/05_align_scale_axes.py` integrates the former `plantwithtag_reconstruction` algorithm into stage `05_alignment`. It detects sub-pixel AprilTag corners, triangulates them with registered COLMAP cameras, derives metres per COLMAP unit from measured black-square edges, fits the tag plane as `+Z`, projects a configured tag-center direction as `+X`, and places the mean tag center at the origin. A pycolmap `Sim3d` transforms sparse points, cameras, rigs, and frames consistently.

Install the `alignment` optional dependency group from `pyproject.toml`. Change COLMAP paths in `configs/tools.local.yaml`, and change tag family, physical sizes, axis tag IDs, and stage paths in `configs/alignment/*.yaml`. The output contains binary/text COLMAP models, `transform.json`, `alignment_report.json`, detections, and annotated diagnostics.

`scripts/05_align_scale_axes.py` 已将 `plantwithtag_reconstruction` 的核心算法整合到主项目 `05_alignment` 阶段。原项目保持只读，运行时不再依赖它的脚本路径。

## 坐标约定

输出使用右手坐标系，变换约定为：

```text
plant_point_m = scale * rotation * colmap_point + translation
```

- 尺度：对每个 Tag 的四条黑色编码方框边分别估计“米/COLMAP 单位”，取中位数。
- 原点：所有指定 Tag 中心的平均值。
- `+Z`：用所有 Tag 角点拟合共面平面，法向选择朝向相机中心的一侧。
- `+X`：将 `x_axis_tags: [FROM, TO]` 的中心连线投影到 Tag 平面；方向为 `FROM → TO`。
- `+Y`：由 `Y = Z × X` 得到，保证右手坐标系。

变换通过 `pycolmap.Reconstruction.transform(Sim3d)` 一次性应用到稀疏点、相机位姿、rig 和 frame。输出同时包含二进制与文本 COLMAP 模型。

## 输入输出

默认读取：

```text
data/scenes/<scene>/04_colmap_foreground/sparse/0/
data/scenes/<scene>/01_frames/images/
```

输出到：

```text
data/scenes/<scene>/05_alignment/
├── sparse/0/                  # 米制、已定向的 COLMAP 模型
├── alignment_report.json      # 尺度、平面误差、重投影误差和验证量
├── transform.json             # 可供其他资产使用的统一 Sim3
├── tag_detections.json
└── tag_diagnostics/           # 前几张 Tag 检测标注图
```

## 配置与运行

通用配置位于 `configs/alignment/default.yaml`。必须填写 Tag 黑色编码方框的真实边长，不包括纸张白边。不同 Tag 尺寸不同时，使用 `tag_sizes_m` 分别设置。

针对原 `plantwithtag_reconstruction` 数据，已经记录配置 `configs/alignment/plantwithtag_36h11.yaml`：Tag family 为 `36h11`，ID 为 `0、1、2、3`，黑框边长为 `0.0998 m`，`+X` 从 Tag 3 指向 Tag 0。

先对主项目场景做预检：

```bash
uv run scripts/05_align_scale_axes.py \
  --alignment-config configs/alignment/plantwithtag_36h11.yaml \
  --dry-run
```

也可以直接读取原项目数据，并把结果写入主项目的 `05_alignment`：

```bash
uv run scripts/05_align_scale_axes.py \
  --alignment-config configs/alignment/plantwithtag_36h11.yaml \
  --model /home/linzz/Desktop/plantwithtag_reconstruction/sparse/0 \
  --images /home/linzz/Desktop/plantwithtag_reconstruction/images \
  --output-root data/scenes/plant_001/05_alignment
```

临时覆盖统一尺寸可使用 `--tag-size-m 0.0998`；不同尺寸可重复传入 `--tag-size 0=0.0998`。已有有效对齐模型时脚本拒绝覆盖，只有显式传入 `--overwrite` 才会替换该阶段管理的模型和诊断文件。

## 质量检查

运行后重点检查 `alignment_report.json`：

- `validation.mean_of_tag_centers_m` 应接近 `[0, 0, 0]`。
- `validation.metric_edge_length_median` 应接近实测 Tag 边长。
- `triangulation.*.reprojection_median_px` 通常应在数个像素以内。
- `alignment_diagnostics.scale_sample_relative_mad` 越小，16 条边给出的尺度越一致。
- `alignment_diagnostics.ground_plane_rms_m` 越小，Tag 共面性越好。

当前本机 3DGRUT 的 COLMAP loader 不再执行额外世界坐标归一化，因此 `05_alignment` 的米制坐标会原样进入训练。
