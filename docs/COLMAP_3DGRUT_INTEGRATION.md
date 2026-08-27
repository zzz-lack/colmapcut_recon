# COLMAP 与 3DGRUT 集成 / COLMAP and 3DGRUT Integration

[中文详细说明](#数据流与目录契约) · [English summary](#english-summary)

> 外部依赖与路径修改 / External dependencies and path configuration: [EXTERNAL_DEPENDENCIES.md](EXTERNAL_DEPENDENCIES.md)

## English summary

COLMAP and NVIDIA 3DGRUT remain independent repositories. The project runs COLMAP as `feature_extractor → matcher → mapper`, records each invocation, and refuses to overwrite an existing reconstruction. The 3DGRUT adapter links composite images and the aligned `sparse/0` model into a standard COLMAP-style dataset, while training outputs stay under `runs/<scene>/3dgrut`.

Change repository, executable, and Python paths in `configs/tools.local.yaml`. Change COLMAP options in `configs/colmap/default.yaml` and 3DGRUT Hydra overrides in `configs/training/3dgrut.yaml`. Use `configs/tools.example.yaml` as the portable template. Preview both external commands with `--dry-run` before launching reconstruction or training.

主项目不复制或修改 `/home/linzz/colmap` 与 `/home/linzz/3dgrut`。二者保持独立环境，由 `configs/tools.local.yaml` 记录入口路径，主项目负责输入输出边界、命令组装和调用记录。

## 数据流与目录契约

```text
data/scenes/<scene>/
├── 01_frames/images/                 # COLMAP 输入：原始筛选帧
├── 02_colmap_full/
│   ├── database.db                   # feature_extractor/matcher 数据库
│   ├── sparse/0/
│   │   ├── cameras.bin
│   │   ├── images.bin
│   │   └── points3D.bin
│   ├── logs/*.json                   # 每个外部命令的参数、耗时、返回码
│   └── manifest.json
├── 05_alignment/sparse/0/            # 尺度与坐标轴对齐后的 COLMAP 模型
├── 06_composite/images/              # 植物前景与固定背景合成图
└── 07_datasets/3dgrut/
    ├── images/                        # 指向 06_composite 的文件级符号链接
    ├── sparse/0/                      # 指向 05_alignment 模型的符号链接
    └── dataset_manifest.json

runs/<scene>/3dgrut/
├── last_process.json                 # 最近一次外部进程调用记录
├── last_invocation.json              # 数据集、命令和新建运行目录
└── <dataset>-<timestamp>/             # 3DGRUT 自己生成的 checkpoint/PLY/日志
```

`02_colmap_full` 是未分割的完整稀疏重建；3DGRUT 默认使用后续完成米制对齐的 `05_alignment/sparse/0`，而不是直接使用未对齐模型。训练数据只保存可复现的输入链接，任何 checkpoint、PLY 或日志都不得写入 `data/`。

3DGRUT 会自动寻找与图片同名的 `_mask.png`。项目默认 `load_loss_mask: false`，因此数据准备阶段会排除这些文件；需要 loss mask 时可显式传入 `--load-loss-mask`。

## 配置

本机外部入口：

```yaml
colmap:
  repository: /home/linzz/colmap
  executable: /home/linzz/colmap/build/src/colmap/exe/colmap

threedgrut:
  repository: /home/linzz/3dgrut
  python: /home/linzz/3dgrut/.venv/bin/python
```

COLMAP 的相机模型、匹配器、GPU 开关与各阶段额外参数在 `configs/colmap/default.yaml` 中配置。3DGRUT 的 Hydra app、背景色、降采样、PLY 导出和额外覆盖项在 `configs/training/3dgrut.yaml` 中配置。

## 调用

先用 `--dry-run` 完成路径预检并查看不会经过 shell 插值的实际命令：

```bash
.venv/bin/python scripts/02_run_colmap.py --dry-run
```

确认后执行 COLMAP：

```bash
.venv/bin/python scripts/02_run_colmap.py
```

在掩膜过滤与尺度对齐阶段完成后，组装 3DGRUT 输入并预览训练命令：

```bash
.venv/bin/python scripts/07_prepare_datasets.py
.venv/bin/python scripts/08_train_3dgrut.py --dry-run
```

开始训练：

```bash
.venv/bin/python scripts/08_train_3dgrut.py
```

所有脚本都支持 `--scene-config`，也可用 `--images`、`--output`、`--dataset-root`、`--sparse-model`、`--repository` 或 `--python` 临时覆盖配置。已有 COLMAP 数据库或主模型时，适配器会拒绝覆盖，以免无意破坏重建结果。
