# IAMINN-v1 失败分析

## 1. 实验背景
本次分析对应的正式公平实验为二维分层材料反演（`layered`）任务，目标是在相同训练预算、相同观测点、相同边界条件和相同优化器配置下，对比 `PINN` 与 `IAMINN-v1` 的空间材料参数识别能力。

对应结果目录：
- PINN：`/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master/exp/spatial_material_inverse/layered/pinn/layered_bc400_obs500_val200_eval500_noise0_seed42_iter50000_softbc_node29`
- IAMINN-v1：`/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master/exp/spatial_material_inverse/layered/iaminn/layered_bc400_obs500_val200_eval500_noise0_seed42_iter50000_softbc_node30`

## 2. 公平性说明
本次结论基于严格一致的实验设置：
- 同一个二维单位方板 `[0,1] x [0,1]`
- 同一个 `layered` 材料分布
- 同一个随机种子 `42`
- 同一个观测点缓存文件
- 同一个软约束边界条件
- 同一个训练预算 `50000` 步
- 同一个优化器 `Adam`
- 同一个学习率 `1e-3`
- 同样的 `num_domain=4000`
- 同样的 `num_boundary=400`
- 同样的 `num_observe=500`
- 同样的 `num_val_observe=200`
- 同样的 `num_eval_observe=500`
- 都按 `validation_observation_mse` 选择 best checkpoint

因此，本次分析结论可视为真实有效，而不是由训练预算不一致或数据划分不一致造成。

## 3. 关键结果
### 3.1 主指标对比
| 指标 | PINN | IAMINN-v1 | 更优者 |
|---|---:|---:|---|
| lambda 相对 L2 误差 | 0.2735 | 0.3624 | PINN |
| mu 相对 L2 误差 | 0.1346 | 0.2370 | PINN |
| 材料场向量平均绝对误差 | 0.3384 | 0.5201 | PINN |
| evaluation observation MSE | 1.38e-06 | 1.37e-05 | PINN |
| validation selection MSE | 1.23e-06 | 1.44e-05 | PINN |
| PDE residual mean abs | 1.454e-01 | 4.234e-03 | IAMINN-v1 |

### 3.2 现象总结
IAMINN-v1 在 PDE residual 上明显更低，但在材料参数识别精度、观测拟合精度上明显差于 PINN。这说明它学到了“更容易满足 PDE 的解”，但没有学到“更准确的材料场”。

## 4. 核心失败原因
### 4.1 材料场塌缩为近乎常数场
对正式实验保存的稠密网格进行检查后发现：
- PINN 预测的 `lambda(x,y)`、`mu(x,y)` 仍然有明显空间变化
- IAMINN-v1 预测的 `lambda(x,y)`、`mu(x,y)` 几乎退化为常数场

具体表现：
- IAMINN-v1 的 `lambda_pred_min/max` 几乎相同
- IAMINN-v1 的 `mu_pred_min/max` 几乎相同
- 这意味着它几乎没有真正识别出分层界面

这不是一个小误差问题，而是结构性失败。

### 4.2 v1 结构允许“状态分支绕开材料分支”
IAMINN-v1 的网络结构中：
- `state_net` 直接输出 `ux, uy, sxx, syy, sxy`
- `interface_net` 再去决定 `lambda(x,y), mu(x,y)`

这种结构的问题是，状态场已经被单独拟合到较合理的程度后，材料分支只需要提供一个能让 PDE 损失变小的折中解即可。结果就是：
- PDE residual 可以很小
- 但材料场不一定对

换句话说，材料分支没有被“强制”承担足够的物理解释责任。

### 4.3 v1 的界面建模过弱
IAMINN-v1 采用了“区域 softmax 混合”的材料表示，但没有配套：
- 显式界面正则
- 界面锐化调度
- 分阶段训练
- level-set 约束
- 交替优化

因此，它的 interface branch 很容易退化成：
- 整个区域几乎都属于同一类
- 最终材料参数场接近全局常数

### 4.4 结构先验方向没错，但 v1 的实现不够强
IAMINN-v1 的出发点“不要自由回归材料场，而是利用区域/界面先验”是合理的。
但是 v1 的实际实现：
- 先验太弱，学不出清晰界面
- 同时又因为全局区域参数过少，表达能力容易塌缩

所以它落入了“既没有真正学出界面，也失去了自由场回归能力”的尴尬区间。

## 5. 结论
### 5.1 当前结论
IAMINN-v1 当前不能作为论文主方法，原因不是它“稍微差一点”，而是：
- 主逆问题指标明显落后于 PINN
- 已经出现材料场塌缩的结构性失败
- 即使 PDE residual 更好，也不足以支撑论文创新成立

### 5.2 是否继续在 v1 上调参
不建议继续把主要时间花在 v1 上做超参数微调，原因有三点：
1. 失败模式已经不是简单调学习率就能解决的
2. 结构绕行问题依然存在
3. 再继续烧卡，很可能只会得到“PDE 更漂亮，但反演仍不准”的结果

## 6. 对下一版方法的要求
为了让下一版方法真正有机会碾压 PINN，下一版必须满足：
1. **材料场与位移场耦合更强**
2. **界面表示更明确，而不是容易塌缩的软混合**
3. **不能让状态分支轻易绕开材料分支**
4. **训练过程需要更适合 inverse problem，而不是一锅端地联合训练**

因此，下一版方法改为 `IAMINN-v2`，方向是：
- 显式界面表示
- 更强的物理耦合
- 先在 `layered` 上做小规模公平诊断
- 只有赢了 PINN，再进入正式 `50000` 步实验
