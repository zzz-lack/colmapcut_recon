# Isaac Sim 5.x 3DGS 与碰撞资产 / 3DGS + Collision Asset

[中文概述](#中文概述) · [English details](#english-details)

> 外部依赖与路径修改 / External dependencies and path configuration: [EXTERNAL_DEPENDENCIES.md](EXTERNAL_DEPENDENCIES.md)

## 中文概述

`scripts/11_export_simulation_asset.py` 将组合后的植物/地面 3DGS PLY 与闭合地面碰撞网格打包为 Isaac Sim 5.x 可用的 NVIDIA NuRec USDZ。包内包含可渲染的 NuRec volume、标准 PhysX 静态三角网格碰撞和用于追溯的 mesh PLY。

该流程使用 NVIDIA 3DGRUT 的转码工具；在 `configs/tools.local.yaml` 修改 `threedgrut.repository` 和 `threedgrut.python`。输入与输出资产路径通过 CLI 参数或场景配置传入。Isaac Sim/OpenUSD 属于独立外部运行时。

## English details

`scripts/11_export_simulation_asset.py` packages the combined plant/ground 3DGS PLY and
the closed ground collision mesh into an Isaac Sim 5.x-compatible NVIDIA NuRec USDZ.

The converter uses the local NVIDIA 3DGRUT `transcode` and mesh-injection tools. The
lightweight transcode path avoids initializing the CUDA/Slang training renderer. The
final USDZ contains:

- a NuRec `UsdVol.Volume` and `.nurec` payload rendered by Omniverse RTX;
- `mesh.usd` with standard `UsdPhysics.CollisionAPI` and exact static mesh collision;
- `mesh.ply`, retained inside the package for traceability.

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
