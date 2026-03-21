# GeoIAMINN 设计与公平诊断计划

## 1. 为什么从 IAMINN-v2.2 转向 GeoIAMINN
前面的 `IAMINN-v2.1 / v2.2` 已经说明一个事实：
- 纯 MLP 的界面分支即使加了 staged training，也容易塌缩
- 它没有充分利用当前 benchmark 本身的几何先验

而我们当前的核心任务恰好都是**几何结构很明确的两相材料问题**：
- `layered`
- `single_inclusion`
- `double_inclusion`

因此，下一步不再让网络“自由猜一整张界面图”，而是直接把界面写成**少量几何参数**：
- layered：一条可学习分层界面
- single_inclusion：一个可学习圆夹杂
- double_inclusion：两个可学习圆夹杂

这不是作弊，而是把问题从“自由场回归”改写成“几何参数 + 区域参数识别”。

## 2. GeoIAMINN 的核心思路
GeoIAMINN 仍然保留：
- `state_net` 学位移场
- 区域材料参数 `lambda_regions / mu_regions`

但界面不再由 `interface MLP` 生成，而是由显式几何参数生成：
- `layer_y`
- `circle center`
- `radius`

然后再通过 sigmoid / softmax 形成区域概率，得到：
- `lambda(x,y)`
- `mu(x,y)`

## 3. 它和 IAMINN-v2 的区别
### 3.1 自由度更小
IAMINN-v2：
- 让一个 MLP 去表示整张界面

GeoIAMINN：
- 只学习少量几何参数

### 3.2 更贴合当前 benchmark
对于 layered / inclusion 类任务，真实界面本来就接近：
- 直线
- 圆

所以显式几何参数化不是额外作弊，而是更合理的结构先验。

### 3.3 更有希望“碾压 PINN”
如果任务真的是少量结构参数控制的两相材料识别，那么：
- `PINN` 需要在整张空间里拟合自由材料场
- GeoIAMINN 只需要识别少数界面参数和区域材料参数

从可识别性角度看，后者更有可能明显占优。

## 4. 第一轮公平诊断计划
### 4.1 先做 layered
先不扩展更多 case，只做：
- `layered`
- `PINN baseline` vs `GeoIAMINN`

### 4.2 公平性约束
保持与现有 baseline 完全同口径：
- 同一个二维方板
- 同一个 observation cache
- 同一个 soft BC
- 同一个 `seed`
- 同样 `5000` 步总预算
- 同样 train/val/eval 观测点
- 同样用 validation observation MSE 选 best checkpoint

### 4.3 三阶段训练仍保留
GeoIAMINN 仍使用三阶段训练：
1. warmup
2. material stage
3. joint fine-tuning

但这一次 material stage 的目标不再是逼一个 MLP 学整张界面，而是优化少量几何参数和区域参数。

## 5. 成功判据
GeoIAMINN 只有在 layered 5k 公平诊断里满足以下条件，才值得进 50000：

1. `lambda` 误差优于 `PINN`
2. `mu` 误差优于 `PINN`
3. 材料场向量误差优于 `PINN`
4. observation MSE 不明显差于 `PINN`
5. 学出的界面参数与真实几何一致性合理

## 6. 当前中期信号
300 步 smoke 已经给出积极信号：
- material stage 中两相占比接近 `50/50`
- 没有再塌成全背景相
- 学出的 `layer_y` 接近 `0.5`
- `lambda/mu/material vector error` 明显优于之前的 interface-MLP 版本

这说明 GeoIAMINN 至少值得进入 layered 5000 步公平诊断。
