# IAMINN-v2.2 设计与公平诊断计划

## 1. v2.2 要解决什么
`IAMINN-v2.1` 的核心失败模式已经明确：
- 不是完全不会学区域参数
- 而是 explicit region 几乎在全域塌为 `0`
- 导致材料场退化成近似单相常值

因此，`v2.2` 的目标不是继续盲调学习率，而是针对这个失败机理做结构化修正：
- 给界面/材料分支一个独立优化阶段
- 防止显式区域整体塌缩
- 再回到联合训练

## 2. v2.2 的核心思路
### 2.1 三阶段训练
`v2.2` 采用三阶段训练，而不是 `v2.1` 的两阶段。

#### 阶段 A：state warmup
目标：
- 先把位移场训稳

设置：
- 冻结材料/界面分支
- physics scale 设低或为 `0`
- 只让 state 分支先吸收 boundary + observation 信号

#### 阶段 B：material stage
目标：
- 单独激活界面分支和区域参数
- 让第二相真正被“用起来”

设置：
- 冻结 `state_net`
- 只训练 `interface_net + raw_lambda_params + raw_mu_params`
- 使用基于 domain 点的 physics objective
- 增加 region usage penalty，阻止显式区域塌到 `0`
- 增加轻量 binary penalty，让区域分配逐步更清晰

#### 阶段 C：joint fine-tuning
目标：
- 在界面与区域参数已经被激活后
- 再做完整联合优化

设置：
- 解冻全部分支
- 恢复标准 soft BC + observation + PDE 损失
- 用 validation observation MSE 选 best checkpoint

## 3. v2.2 的新增约束
### 3.1 region usage penalty
作用：
- 防止显式区域在整张图上都变成 `0`

做法：
- 对每个 class 的平均占据率施加最低占据率约束
- 采用归一化 shortfall penalty

解释：
- 这不是使用真值监督界面
- 只是告诉模型“每个材料相都应该有非零体积分数”

### 3.2 binary penalty
作用：
- 减少界面概率长期停留在模糊灰区

做法：
- 对显式区域概率加轻量 `p(1-p)` 惩罚

解释：
- 它不是强制特定位置是某个相
- 只是鼓励分区更明确

## 4. 与 v2.1 的主要区别
### 4.1 训练策略不同
`v2.1`：
- warmup -> joint

`v2.2`：
- warmup -> material stage -> joint

### 4.2 新增防塌缩机制
`v2.1`：
- 没有单独防止 explicit region 消失

`v2.2`：
- 增加 usage penalty
- 增加 binary penalty

### 4.3 目标更明确
`v2.2` 不再假设 joint training 自己就能把界面学出来，而是承认：
- inverse 材料识别需要显式的材料阶段

## 5. 第一轮公平诊断怎么做
### 5.1 诊断对象
先只做：
- `layered`
- `PINN baseline` vs `IAMINN-v2.2`

### 5.2 统一设置
保持与现有 `PINN 5000` baseline 一致：
- 同一个二维单位方板
- 同一个 `layered` case
- 同一个 soft boundary
- 同一个 observation cache
- 同一个 seed
- 同一个 `bc=400, obs=500, val=200, eval=500`
- 同一个总预算 `5000` 步

注意：
- `v2.2` 的三阶段训练加起来总步数仍然是 `5000`
- 不额外增加训练预算

### 5.3 默认三阶段划分
第一轮建议：
- warmup: `1000`
- material stage: `1000`
- joint: `3000`

如果这轮已经明显优于 `PINN`，再考虑更长预算。

## 6. 成功判据
`v2.2` 只有在满足以下条件时，才值得继续上 `50000`：

1. `lambda` 相对误差优于 `PINN`
2. `mu` 相对误差优于 `PINN`
3. 材料场向量误差优于 `PINN`
4. observation MSE 不劣于 `PINN`
5. PDE residual 不出现明显恶化
6. class probability 不再整体塌为背景相

## 7. 失败判据
如果出现以下任一情况，则 `v2.2` 不进入正式实验：

1. explicit region 概率仍然整体接近 `0`
2. 材料场仍然接近单相常值
3. 只提升 `lambda`，但 `mu / observation / PDE` 仍然系统性更差
4. 需要更大参数量却没有更强效果

## 8. 本轮需要保存的额外结果
除现有保存内容外，`v2.2` 还要额外保存：
- `json/material_stage_history.json`
- `txt/material_stage_history.txt`
- `npz/material_stage_points.npz`
- `txt/material_stage_points.txt`

这些文件用于回答：
- 材料阶段到底有没有把 explicit region 激活起来
- 区域参数有没有逐步拉开
- usage penalty 是否真的起作用

## 9. 当前工作原则
这一轮仍然保持同一原则：
- 不为了结果好看而作弊
- 不根据 test 结果反向挑模型
- 不因为想发论文就忽视失败信号

只有 `v2.2` 在 layered 的公平小实验里真实赢过 `PINN`，我们才继续向正式主实验推进。
