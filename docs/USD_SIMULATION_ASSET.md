# Isaac Sim 5.x 3DGS 与碰撞资产 / 3DGS + Collision Asset

[中文概述](#中文概述) · [English details](#english-details)

> 外部依赖与路径修改 / External dependencies and path configuration: [EXTERNAL_DEPENDENCIES.md](EXTERNAL_DEPENDENCIES.md)

## 中文概述

当前胶水流水线依次完成后训练背景分离、地面高斯提取、闭合地面碰撞网格、果实三角 mesh/刚体生成、静态 NuRec 环境导出和最终组合 USDZ。包内包含可渲染的 NuRec volume、标准静态地面碰撞、动态果实 mesh、质量、运动学状态与碰撞代理。

该流程使用 NVIDIA 3DGRUT 的转码工具；在 `configs/tools.local.yaml` 修改 `threedgrut.repository` 和 `threedgrut.python`。输入与输出资产路径通过 CLI 参数或场景配置传入。Isaac Sim/OpenUSD 属于独立外部运行时。

完整调用：

```bash
uv run --extra alignment python scripts/run_pipeline.py \
  --video /home/linzz/Desktop/simple_photo_capture/video/fruit_tomato.mp4
```

后处理参数在 `configs/simulation/fruit_tomato_asset.yaml`，果实实例参数在 `configs/segmentation/fruit_tomato_entities.toml`。最终文件为 `data/scenes/fruit_tomato/11_simulation_asset/fruit_tomato_simulation.usdz`。

Isaac Sim 5.x 可使用全部 NuRec 视觉和 USD Physics。Genesis 可使用标准地面/果实 mesh 与物理属性，但通常不会渲染 NVIDIA 专用 NuRec volume；若 Genesis 需要植物本体的标准 mesh，还需额外增加植物表面重建阶段。

## English details

The glue pipeline now performs post-training background separation, ground-Gaussian
extraction, closed ground collision meshing, fruit triangle-mesh/rigid-body generation,
static NuRec export, and final USDZ composition.

The converter uses the local NVIDIA 3DGRUT `transcode` and mesh-injection tools. The
lightweight transcode path avoids initializing the CUDA/Slang training renderer. The
final USDZ contains:

- a NuRec `UsdVol.Volume` and `.nurec` payload rendered by Omniverse RTX;
- `mesh.usd` with standard `UsdPhysics.CollisionAPI` and exact static mesh collision;
- `mesh.ply`, retained inside the package for traceability.
- standard dynamic fruit meshes with mass, kinematic state, and collision proxies.

Run the complete video-to-simulation-asset path with:

```bash
uv run --extra alignment python scripts/run_pipeline.py \
  --video /home/linzz/Desktop/simple_photo_capture/video/fruit_tomato.mp4
```

The final package is `data/scenes/fruit_tomato/11_simulation_asset/fruit_tomato_simulation.usdz`.
Isaac Sim 5.x can use both NuRec rendering and USD Physics. Genesis can consume the
standard ground/fruit meshes and physics metadata, but generally ignores the
NVIDIA-specific NuRec volume. A standard plant mesh requires a separate surface
reconstruction stage if Genesis must render the plant itself.

Run the current Roman tomato asset export with:

```bash
cd /home/linzz/Desktop/colmapcut_recon

/home/linzz/3dgrut/.venv/bin/python \
  scripts/11_export_simulation_asset.py \
  --gaussians data/scenes/roman_tomato_02/08_asset_assembly/combined_gaussians.ply \
  --collision-mesh data/scenes/roman_tomato_02/09_sugar_ground/mesh_output/ground_collision_2x2m.ply \
  --output-usdz data/scenes/roman_tomato_02/10_simulation_asset/roman_tomato_02_isaacsim51.usdz \
  --overwrite
```

Open or reference `roman_tomato_02_isaacsim51.usdz` in Isaac Sim 5.1. The collider is
invisible but active. It uses exact static triangle-mesh collision, so it must not be
attached to a dynamic rigid body.

The older `roman_tomato_02.usdc` uses custom `gaussian:*` primvars on `UsdGeom.Points`.
It remains a data/interchange file but is not a native 3DGS rendering asset for Isaac
Sim 5.1. Isaac Sim 6.x uses the newer ParticleField schema and needs a separate export.
