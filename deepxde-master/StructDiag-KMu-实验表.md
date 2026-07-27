# StructDiag-KMu 实验表

| 编号 | 实验名 | 结构 | 目的 | 实验所在目录 |
|---|---|---|---|---|
| A | `pinn_mlp_kmu_3load` | 单网络，`1` 个共享 `MLP` 同时输出 `u_x,u_y,\sigma_{xx},\sigma_{yy},\sigma_{xy},K,\mu` | 第 1 组基线，代表“单网络包揽物理场和材料场” | `/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master/exp/StructDiag-KMu-60k/single_inclusion/pinn_mlp_kmu_3load_60k` |
| B | `twobranch_compact_mlp_kmu_3load` | 两个分支，各 `1` 个 `MLP`：`state_net -> (u_x,u_y)`，`material_net -> (K,\mu)`，不输出应力 | 和 A 对比，单独检验“位移场 / 材料场解耦”是否有价值 | `/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master/exp/StructDiag-KMu-60k/single_inclusion/twobranch_compact_mlp_kmu_3load_60k` |
| C | `twobranch_stress_mlp_kmu_3load` | 两个分支，各 `1` 个 `MLP`：`state_net -> (u_x,u_y,\sigma_{xx},\sigma_{yy},\sigma_{xy})`，`material_net -> (K,\mu)` | 和 B 对比，检验“不输出应力的紧凑型方程”与“显式输出应力的标准方程”谁更划算 | `/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master/exp/StructDiag-KMu-60k/single_inclusion/twobranch_stress_mlp_kmu_3load_60k` |
| D | `pfnn_scalar_kmu_3load` | `PFNN` 风格 `5` 个物理子网分别输出 `u_x,u_y,\sigma_{xx},\sigma_{yy},\sigma_{xy}`，材料为全局标量 `K,\mu` | 第 2 组基线，代表“老派多网络 + 全局平均材料” | `/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master/exp/StructDiag-KMu-60k/single_inclusion/pfnn_scalar_kmu_3load_60k` |
| E | `fivestate_stress_mlp_kmu_3load` | `5` 个物理 `MLP` 分别输出 `u_x,u_y,\sigma_{xx},\sigma_{yy},\sigma_{xy}`，另加 `1` 个材料 `MLP -> (K,\mu)` | 和 D 对比，检验“多物理子网 + 空间材料分支”是否优于“多物理子网 + 全局材料” | `/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master/exp/StructDiag-KMu-60k/single_inclusion/fivestate_stress_mlp_kmu_3load_60k` |

## 能回答的问题

- `A vs C`：单网络 vs 两分支，判断“把物理场和材料场拆开算”是否必要。
- `B vs C`：紧凑型方程 vs 显式应力方程，判断是否值得显式输出应力。
- `D vs E`：全局平均材料 vs 空间材料场，判断第 6 个材料分支是否必要。
- `C vs E`：`1` 个物理 `MLP` vs `5` 个物理 `MLP`，判断物理场是否值得彻底拆开。

## 统一设置

- case: `single_inclusion`
- 训练变量: `K, \mu`
- 工况: `biaxial_bulk + uniaxial_x + pure_shear`
- seed: `42`
- 观测划分标签: `kmu_structdiag_v2`
- 采样点: `num_domain=8000, num_boundary=800, num_observe=1000, num_val_observe=400, num_eval_observe=1000`
- 流程: `smoke2k -> formal60k`

## 启动脚本

- `/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master/launch_structdiag_kmu_combo.sh`
