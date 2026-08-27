# Roman 番茄仿真引用 / Roman Tomato Simulation References

## 中文

可视化 NuRec 资产和调优后的不可见地面碰撞体保存在确定性重建输出目录 `data/scenes/roman_tomato_02/10_simulation_asset/`。本目录预留给附加植物碰撞代理体和独立番茄刚体。场景资产路径在 `configs/simulation/tomato_orbit.toml` 修改；外部 Isaac Sim/3DGRUT 依赖见 `docs/EXTERNAL_DEPENDENCIES.md`。

## English

The visual NuRec asset and its tuned, invisible ground collider remain in the
deterministic reconstruction output tree:

`data/scenes/roman_tomato_02/10_simulation_asset/roman_tomato_02_isaacsim51_tuned_collision.usdz`

This directory is reserved for additional plant collision proxies and independent
tomato target rigid bodies. The monolithic NuRec volume is visual-only apart from its
embedded ground collider.
