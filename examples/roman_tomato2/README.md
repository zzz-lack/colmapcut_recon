# roman_tomato2 可运行示例 / Runnable example

[中文](#中文) · [English](#english)

## 中文

### 示例输入

仓库跟踪的输入文件是 `input/roman_tomato2_example.mp4`。它由原始 4K 手机视频完整压缩而来，没有截短相机轨迹，并已删除 GPS、设备和拍摄时间元数据。

| 属性 | 值 |
| --- | --- |
| 时长 | 94.9949 秒 |
| 视频 | H.264、1920×1080、30000/1001 fps、约 1.71 Mbit/s |
| 音频 | 无 |
| 文件大小 | 20,304,540 字节（约 19.36 MiB） |
| SHA-256 | `8dce2fb9f574a7334b0f1284dab5a0c3d0d7b04917e96bcadc33f2a43845c673` |

最终文件使用两遍 H.264 编码，并在发布前清除元数据。可用以下等价流程从自己的源视频生成：

```bash
ffmpeg -y -i /path/to/source.mp4 -map 0:v:0 \
  -vf "scale=1920:-2:flags=lanczos,fps=30000/1001" \
  -c:v libx264 -preset medium -b:v 1800k -maxrate 2400k -bufsize 3600k \
  -pix_fmt yuv420p -pass 1 -passlogfile /tmp/example_pass -an -f mp4 /dev/null

ffmpeg -y -i /path/to/source.mp4 -map 0:v:0 \
  -vf "scale=1920:-2:flags=lanczos,fps=30000/1001" \
  -c:v libx264 -preset medium -b:v 1800k -maxrate 2400k -bufsize 3600k \
  -pix_fmt yuv420p -pass 2 -passlogfile /tmp/example_pass -an \
  -map_metadata -1 -map_metadata:s -1 -map_chapters -1 -movflags +faststart \
  examples/roman_tomato2/input/roman_tomato2_example.mp4
```

### 一键试运行

先复制并修改外部工具路径：

```bash
cp configs/tools.example.yaml configs/tools.local.yaml
```

然后运行：

```bash
uv run --extra alignment python scripts/run_pipeline.py \
  --scene-config configs/scenes/roman_tomato2_example.yaml \
  --video examples/roman_tomato2/input/roman_tomato2_example.mp4
```

默认开启恢复模式。完整命令和每阶段状态写入 `data/scenes/roman_tomato2_example/pipeline_manifest.json`，子进程记录写入同目录的 `pipeline_logs/`。

### 分阶段、独立调用

以下命令按顺序执行。前五步由主 `uv` 环境运行；后续专用操作分别使用 `configs/tools.local.yaml` 中 `threedgrut.python` 或 `sugar.python` 指向的解释器。

```bash
# 1. FFmpeg/ffprobe：抽帧
uv run python scripts/01_extract_frames.py \
  --scene-config configs/scenes/roman_tomato2_example.yaml \
  --sampling-config configs/preprocessing/video_sampling_roman_tomato2_example.yaml \
  --video examples/roman_tomato2/input/roman_tomato2_example.mp4

# 2. COLMAP：特征、顺序匹配和稀疏重建
uv run python scripts/02_run_colmap.py \
  --scene-config configs/scenes/roman_tomato2_example.yaml \
  --tools-config configs/tools.local.yaml \
  --colmap-config configs/colmap/default.yaml

# 3. OpenCV + pycolmap：AprilTag 米制尺度与坐标系对齐
uv run --extra alignment python scripts/05_align_scale_axes.py \
  --scene-config configs/scenes/roman_tomato2_example.yaml \
  --alignment-config configs/alignment/roman_tomato2_example_36h11.yaml \
  --model data/scenes/roman_tomato2_example/02_colmap_full/sparse/0

# 4. 为 3DGRUT 组装 images + sparse/0 数据集
uv run python scripts/07_prepare_datasets.py \
  --scene-config configs/scenes/roman_tomato2_example.yaml \
  --training-config configs/training/3dgrut_roman_tomato2_example_smoke.yaml

# 5. 调用外部 NVIDIA 3DGRUT
uv run python scripts/08_train_3dgrut.py \
  --scene-config configs/scenes/roman_tomato2_example.yaml \
  --tools-config configs/tools.local.yaml \
  --training-config configs/training/3dgrut_roman_tomato2_example_smoke.yaml
```

设定专用解释器和第 5 步生成的 PLY 后继续：

```bash
THREEDGRUT_PYTHON=/path/to/3dgrut/.venv/bin/python
SUGAR_PYTHON=/path/to/sugar/environment/bin/python
GAUSSIANS=/path/to/runs/roman_tomato2_example/3dgrut/smoke_100/<run>/export_last.ply

# 6. NumPy/plyfile：几何背景、地面与植物分离
"$THREEDGRUT_PYTHON" scripts/10_clean_gaussians.py \
  --full-scene "$GAUSSIANS" \
  --output-directory data/scenes/roman_tomato2_example/08_asset_separation \
  --x-min -1 --x-max 1 --y-min -1 --y-max 1 --z-min -0.15 --z-max 1.6

# 7. SciPy/Open3D：闭合地面碰撞 mesh
"$SUGAR_PYTHON" scripts/16_build_ground_collision_mesh.py \
  --ground-points data/scenes/roman_tomato2_example/08_asset_separation/ground_gaussians.ply \
  --ply-output data/scenes/roman_tomato2_example/09_ground_collision/ground_collision.ply \
  --obj-output data/scenes/roman_tomato2_example/09_ground_collision/ground_collision.obj \
  --report-output data/scenes/roman_tomato2_example/09_ground_collision/ground_collision_report.json \
  --x-min -1 --x-max 1 --y-min -1 --y-max 1

# 8. plyfile + OpenUSD：颜色引导果实 mesh/刚体
"$THREEDGRUT_PYTHON" scripts/17_extract_tomato_entities.py \
  --config configs/segmentation/roman_tomato2_example_entities.toml \
  --bootstrap-colour

# 9. 3DGRUT NuRec + OpenUSD：静态环境 USDZ
"$THREEDGRUT_PYTHON" scripts/11_export_simulation_asset.py \
  --gaussians data/scenes/roman_tomato2_example/10_fruit_entities/static_scene_without_tomatoes.ply \
  --collision-mesh data/scenes/roman_tomato2_example/09_ground_collision/ground_collision.ply \
  --output-usdz data/scenes/roman_tomato2_example/11_simulation_asset/roman_tomato2_example_environment.usdz \
  --threedgrut-root /path/to/3dgrut \
  --threedgrut-python "$THREEDGRUT_PYTHON"

# 10. OpenUSD：组合静态环境和动态果实实体
"$THREEDGRUT_PYTHON" scripts/21_compose_simulation_usdz.py \
  --environment-usdz data/scenes/roman_tomato2_example/11_simulation_asset/roman_tomato2_example_environment.usdz \
  --fruit-entities-usd data/scenes/roman_tomato2_example/10_fruit_entities/tomato_entities.usda \
  --output-usdz data/scenes/roman_tomato2_example/11_simulation_asset/roman_tomato2_example_simulation.usdz
```

在第 2 或第 5 步末尾添加 `--dry-run`，可以查看适配器将执行的外部 COLMAP 或 3DGRUT 原生命令。

### 本次实跑结果

- 抽取 95 张 1080p 图像；COLMAP 主模型注册 52 张，含 4,762 个稀疏点。
- 49 张已注册图像检测到 Tag；0/1/2/3 号分别有 5/6/35/25 次观测。
- 标定边长中位数为 0.0998002 m；Tag 平面 RMS 为 0.003012 m。
- 100 次 3DGRUT 冒烟训练导出 4,762 个 Gaussian。该设置只验证流水线，不代表最终视觉质量。
- 背景分离保留 4,460 个 Gaussian；地面碰撞 mesh 有 1,842 个顶点、3,680 个三角形，闭合、流形、watertight。
- 颜色引导阶段生成 1 个试验果实刚体；完整语义实例仍应使用 SAGA/SAM 流程。
- 最终 `roman_tomato2_example_simulation.usdz` 为 711,067 字节，包含 NuRec volume、不可见地面碰撞体和果实 mesh/刚体，并通过 OpenUSD 验证。

## English

### Example input

The tracked input is `input/roman_tomato2_example.mp4`. It preserves the complete camera path from the 4K phone capture while removing GPS, device, and capture-time metadata.

| Property | Value |
| --- | --- |
| Duration | 94.9949 seconds |
| Video | H.264, 1920×1080, 30000/1001 fps, about 1.71 Mbit/s |
| Audio | None |
| File size | 20,304,540 bytes (about 19.36 MiB) |
| SHA-256 | `8dce2fb9f574a7334b0f1284dab5a0c3d0d7b04917e96bcadc33f2a43845c673` |

The two-pass FFmpeg commands in the Chinese section reproduce the encoding and strip public metadata.

### One-command trial

Copy `configs/tools.example.yaml` to the Git-ignored `configs/tools.local.yaml`, edit the machine-local installations, and run:

```bash
uv run --extra alignment python scripts/run_pipeline.py \
  --scene-config configs/scenes/roman_tomato2_example.yaml \
  --video examples/roman_tomato2/input/roman_tomato2_example.mp4
```

Resume mode is enabled by default. The full command/status record is written to `data/scenes/roman_tomato2_example/pipeline_manifest.json`, with child-process records under `pipeline_logs/`.

### Independent stages and observed result

The ten commands in the Chinese “分阶段、独立调用” section are language-neutral shell commands. They independently call the FFmpeg, COLMAP, OpenCV/pycolmap, 3DGRUT, Open3D, and OpenUSD-backed stages; append `--dry-run` to stages 2 or 5 to print the underlying external COLMAP or 3DGRUT command without running it.

The exercised run sampled 95 images, registered 52 in the primary COLMAP model, detected all four tags across 49 registered images, recovered a 0.0998002 m median tag edge with 3.012 mm plane RMS, and exported 4,762 Gaussians after 100 smoke iterations. The closed 1,842-vertex/3,680-triangle ground collider passed the Open3D manifold/watertight checks. One colour-bootstrapped fruit rigid body was generated. The 711,067-byte combined USDZ contains the NuRec volume, hidden ground collider, and fruit mesh/rigid body and passed OpenUSD validation. The 100-iteration and colour-bootstrap settings validate integration only; use longer training and SAGA/SAM instances for production quality.
