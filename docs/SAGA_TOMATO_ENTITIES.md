# SAGA 番茄提取与 Isaac Sim 实体 / SAGA Tomato Extraction and Isaac Sim Entities

[中文详细说明](#已部署路径) · [English summary](#english-summary)

> 外部依赖与路径修改 / External dependencies and path configuration: [EXTERNAL_DEPENDENCIES.md](EXTERNAL_DEPENDENCIES.md)

## English summary

SAGA/SegAnyGAussians produces a Boolean mask aligned one-to-one with the full metric 3DGS. This project maps that mask to the strict plant PLY, splits tomato instances, fits physical proxies, removes their volume from the static Gaussian plant, and authors independent Isaac Sim entities. The scene adapter uses symlinks rather than copying large immutable inputs.

Change the standard SegmentAnything3D repository and Python paths in `configs/tools.local.yaml`. Change scene PLY, mask, ground, and entity output paths in `configs/segmentation/saga_tomato.toml`. The shell workflow in `scripts/18_run_saga_tomato.sh` also exposes `PROJECT_ROOT`, `SAGA_ROOT`, `SAGA_PYTHON`, and `SAM_CHECKPOINT` near the top for machine-specific deployment.

本项目把 SAGA（SegAnyGAussians）的输出分成两个阶段处理：SAGA 负责输出与完整米制 3DGS 一一对应的布尔掩码；本项目再把掩码映射到 `plant_strict.ply`、分离番茄实例、拟合物理代理体，并从静态植物高斯中删除相应体积，避免实体和原始 3DGS 重影。

## 已部署路径

- SAGA 源码：`third_party/SegAnyGAussians`
- Python：`/home/linzz/miniconda3/envs/gaussian_splatting_sa3d/bin/python`
- 米制 SAGA 场景：`data/scenes/roman_tomato_02/07_saga`
- 番茄提取配置：`configs/segmentation/saga_tomato.toml`
- 实体输出：`data/scenes/roman_tomato_02/09_tomato_entities`

场景适配器只建立符号链接，不复制 1,000,000 个高斯、942 张原图或 COLMAP 模型。输入组合必须保持一致：

- 图像：`realplantrecon_romantomato2/images`
- 相机：`realplantrecon_romantomato2/metric_scene/sparse`
- 高斯：`plant_metric_original_coords.ply`

不要把旧 `data7_28/segmentation_res/final_mask.pt` 用到当前模型。旧掩码有 2,576,445 个元素，而当前米制高斯只有 1,000,000 个，索引不兼容。

## 执行流程

先检查部署并建立适配目录：

```bash
cd /home/linzz/Desktop/colmapcut_recon
scripts/18_run_saga_tomato.sh prepare
scripts/18_run_saga_tomato.sh smoke
```

为全部 942 帧生成 SAM masks 会占用约 74 GB。当前建议先均匀使用每 10 帧一张（约 95 个视角），既覆盖整圈相机又把磁盘和训练时间降到约十分之一：

```bash
scripts/18_run_saga_tomato.sh masks --stride 10
scripts/18_run_saga_tomato.sh scales
scripts/18_run_saga_tomato.sh train --iterations 2000 --save_iterations 2000
```

需要论文设置的完整覆盖和 10,000 次训练时，改用：

```bash
scripts/18_run_saga_tomato.sh masks --stride 1
scripts/18_run_saga_tomato.sh scales
scripts/18_run_saga_tomato.sh train
```

掩码生成支持断点续跑：已存在的 `.pt` 默认跳过。`--overwrite` 才会覆盖。

SAGA 本身是提示式实例特征场，不直接提供“tomato”类别。`scripts/19_saga_auto_tomato_mask.py` 默认只在 `plant_strict.ply` 范围内以成熟番茄的红色高斯作为自动正样本，在训练后的 SAGA 特征空间选择多组原型，再用特征相似度扩展到完整果实体：

```bash
/home/linzz/miniconda3/envs/gaussian_splatting_sa3d/bin/python \
  scripts/19_saga_auto_tomato_mask.py \
  --feature-ply data/scenes/roman_tomato_02/07_saga/point_cloud/iteration_1000/contrastive_feature_point_cloud.ply \
  --scale-gate data/scenes/roman_tomato_02/07_saga/point_cloud/iteration_1000/scale_gate.pt
```

然后生成 Isaac Sim 实体：

```bash
/home/linzz/miniconda3/envs/gaussian_splatting_sa3d/bin/python \
  scripts/17_extract_tomato_entities.py \
  --mask data/scenes/roman_tomato_02/07_saga/tomato_saga_mask.pt
```

## 输出含义

- `tomato_entities.usda`：每个番茄的可视椭球、球形碰撞体、质量和运动学刚体。
- `tomato_instances.json`：中心、半轴、旋转、质量、蒂部锚点和来源类型。
- `gaussians/tomato_NNN.ply`：每个实例对应的高斯子集。
- `tomatoes_combined.ply`：所有选中番茄高斯。
- `plant_without_tomatoes.ply`：已挖除番茄体积的静态植物高斯。
- `static_scene_without_tomatoes.ply`：上一个文件与原地面高斯进行精确 XYZ 去重后的完整静态场景，可直接重新打包成 NuRec 环境。
- `../10_simulation_asset/roman_tomato_02_without_tomatoes.usdz`：已打包的无番茄静态 NuRec 场景，含不可见地面碰撞网格。

`configs/simulation/tomato_orbit.toml` 已把无番茄 NuRec 场景和
`tomato_entities.usda` 分别挂到 `/World/Environment` 与
`/World/TomatoEntities`。运行 `scripts/isaac/03_orbit_plant.py` 时，方形绕行、
末端相机和 32 个独立物理代理体会加载到同一米制 Z-up 坐标系，无需额外平移。

每个实体初始具有：

```text
physics:kinematicEnabled = true
tomato:attached = true
tomato:stemAnchor = (...)  # 世界坐标，米制，Z-up
```

剪刀碰撞到蒂部切割区域后，把 `tomato:attached` 设为 `false`，并把 `physics:kinematicEnabled` 设为 `false`，果实就会交给 PhysX 重力和碰撞求解。

## 首轮颜色引导结果

在 SAGA 特征训练完成前，可以显式运行：

```bash
/home/linzz/miniconda3/envs/gaussian_splatting_sa3d/bin/python \
  scripts/17_extract_tomato_entities.py --bootstrap-colour
```

这个模式只用于检查坐标、实例拆分和 USD 物理链路。清单中的 `selection_source` 会写成 `ripe_colour_bootstrap`，不能当作最终 SAGA 检测精度。它也只覆盖红/橙色成熟果实，绿色番茄需要图像检测器或人工 SAGA 提示提供额外种子。

## 参数调整

实体误合并时，减小 `maximum_component_span_m` 或 `voxel_size_m`；一个番茄被拆成多个实体时反向调整。红叶或地面误检时，提高 `red_margin`、`opacity_minimum` 或 `minimum_height_m`。最终应依据 `tomato_instances.json`、相机重投影和 Isaac Sim 视图共同复核，而不能只看候选数量。
