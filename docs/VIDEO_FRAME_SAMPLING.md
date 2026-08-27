# 视频采样到重建图片 / Video Frame Sampling

[中文详细说明](#采样策略) · [English summary](#english-summary)

> 外部依赖与路径修改 / External dependencies and path configuration: [EXTERNAL_DEPENDENCIES.md](EXTERNAL_DEPENDENCIES.md)

## English summary

`scripts/01_extract_frames.py` deterministically samples a read-only video from `00_capture/videos` into `01_frames/images`. It supports presentation-timestamp intervals, every-N-frame sampling, and a fixed target count. Every output image is mapped to its source frame index and timestamp in `frame_manifest.json`; extraction is staged in a temporary directory and published only after FFmpeg succeeds and the frame count matches.

Change the `ffmpeg` and `ffprobe` executable paths in `configs/tools.local.yaml`. Change sampling method, time range, JPEG quality, and naming in `configs/preprocessing/video_sampling.yaml`. The referenced `realplantrecon_fruittomato` directory retained 1,444 images but no source video or extraction script, so this integration preserves its `frame_XXXXXX.jpg` naming contract without inventing an undocumented original FPS.

`scripts/01_extract_frames.py` 负责把 `00_capture/videos` 中的只读视频确定性地采样到 `01_frames/images`。

`realplantrecon_fruittomato` 当前保留了 1444 张连续编号的 `frame_000000.jpg` 至 `frame_001443.jpg`，但没有保留源视频或抽帧脚本。因此主项目复用了它的输出命名契约，没有假定无法从现有文件证明的原始采样 FPS。

## 采样策略

配置位于 `configs/preprocessing/video_sampling.yaml`，支持三种互斥策略：

- `interval_seconds`：根据视频逐帧 presentation timestamp 按时间间隔采样，适合普通及可变帧率视频。
- `every_n_frames`：在指定时间范围内每 N 个解码帧保留一个。
- `target_count`：在指定范围首尾之间均匀选出固定数量的图片。

输出文件始终重新连续编号，例如：

```text
data/scenes/plant_001/
├── 00_capture/videos/source.mp4       # 只读输入
└── 01_frames/
    ├── images/
    │   ├── frame_000000.jpg
    │   ├── frame_000001.jpg
    │   └── ...
    ├── frame_manifest.json
    └── ffmpeg_process.json
```

`frame_manifest.json` 为每个输出文件记录原视频帧序号、presentation timestamp 和关键帧标志，并记录视频分辨率、帧率、时长、采样配置以及可复现的 ffprobe/ffmpeg 参数。

## 使用

将一个视频放入场景的 `00_capture/videos` 后先预检：

```bash
uv run python scripts/01_extract_frames.py --dry-run
```

执行默认每秒一张的采样：

```bash
uv run python scripts/01_extract_frames.py
```

常用临时覆盖示例：

```bash
# 每 10 个解码帧保留一张
uv run python scripts/01_extract_frames.py \
  --method every_n_frames --every-n-frames 10

# 在第 5 到第 45 秒之间均匀选择 300 张
uv run python scripts/01_extract_frames.py \
  --method target_count --target-count 300 \
  --start-seconds 5 --end-seconds 45
```

如果视频目录中有多个视频，必须用 `--video` 明确选择。已有输出时默认拒绝覆盖；显式 `--overwrite` 只移除由当前命名配置管理的抽帧图片和 manifest，不触碰原视频或其他文件。

抽帧先写入同一阶段内的临时目录，只有 ffmpeg 成功且图片数量与选择清单一致后才发布到 `01_frames/images`，避免中断后留下半套数据。
