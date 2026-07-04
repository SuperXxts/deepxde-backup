# 第二篇 PINN 反演论文实验设计

本文档只记录第二篇论文的科学问题、方法原理、损失函数、实验矩阵和后续执行顺序。目标是把实验组织成一篇可以投稿 `Computer Methods in Applied Mechanics and Engineering` 或 `Computers and Geotechnics` 的论文，而不是简单堆模型改进。

## 1. 当前论文主线

拟研究的问题是：

```text
在空间变 K(x,y), mu(x,y) 材料反演中，如果边界位移或荷载信息存在尺度错误、形状错误、缺失或只知道一部分，标准 PINN/PFNN 是否会把边界误差错误解释成材料非均质性？

如果会，能否通过低维边界不确定性建模、边界-材料残差解耦训练和全局反力信息，减少这种材料场污染并恢复反问题的可识别性？
```

这篇论文不能写成“又提出一个更复杂的 MLP”。真正的科学问题是：

```text
同一个位移观测残差，可能来自材料参数错误，也可能来自边界条件错误。
标准 PINN/PFNN 没有机制区分这两类误差，因此高自由度材料场可能吸收边界误差，产生虚假的材料非均质性。
```

因此本文的核心贡献应围绕：

```text
1. 揭示边界误差污染空间变材料反演的机制。
2. 提出低维边界参数化，使边界误差有自己的解释通道。
3. 提出边界-材料残差解耦训练，避免材料场优先吸收边界误差。
4. 说明全局反力观测如何补充绝对刚度尺度信息。
```

## 2. 与已有工作的关系

必须谨慎表述创新性。已有研究已经覆盖了很多相近问题，不能把常规 PINN 设置写成创新。

已经存在的相关方向包括：

```text
1. PINN/PFNN 用于固体力学材料参数反演。
2. PINN 中同时输出位移和应力，并用平衡方程与本构一致性残差训练。
3. 用 PointSetBC、OperatorBC、外部可训练变量处理观测、边界和未知参数。
4. 传统 FEMU、IDIC、VFM 中联合或顺序识别边界条件和材料参数。
5. 用低阶多项式、刚体运动、POD 模态或灵敏度矩阵描述边界条件误差。
```

因此，以下内容不能单独作为本文创新点：

```text
PFNN 五输出。
同时预测位移和应力。
平衡残差 + 本构一致性残差。
位移观测损失。
边界损失。
反力积分损失。
Fourier feature。
残差 MLP。
普通自适应损失权重。
单纯把材料从一个尺度场改成 K 和 mu 两个场。
```

本文可以成立的新角度是：

```text
面向空间变 K(x,y), mu(x,y) 反演，系统证明错误边界会诱发虚假材料非均质性，并提出一个低维边界模式 + 残差投影/交替训练的边界-材料解耦 PINN 框架。
```

文献检索目前显示：传统实验力学和 FEMU 文献中已有“边界条件与材料参数共同识别”的工作；PINN 固体力学中也已有未知边界、边界约束影响和材料识别研究。但尚未看到与本文完全相同的组合：

```text
空间变 K/mu 场反演
+ 错误边界导致虚假材料非均质性的系统诊断
+ 低维边界模式显式进入 PINN 位移场
+ 观测残差按边界模式子空间投影拆分
+ 全局反力恢复绝对刚度尺度
+ 与标准 DeepXDE PFNN 和传统反演基线对比。
```

论文中不能写“首次提出边界和材料联合反演”，更稳妥的表述是：

```text
已有工作认识到边界条件对参数识别的重要性，但边界误差在 PINN 空间变材料场反演中如何被吸收到材料非均质性、以及如何通过边界模式投影与反力约束进行机制性解耦，仍缺少系统研究。
```

## 3. 基本物理方程与网络输出

采用二维小变形线弹性。位移为：

```text
d = [u(x,y), v(x,y)]
```

应变由位移自动微分得到：

```text
epsilon_xx = partial u / partial x
epsilon_yy = partial v / partial y
epsilon_xy = 1/2 (partial u / partial y + partial v / partial x)
```

材料参数建议正式实验使用体积模量和剪切模量：

```text
K(x,y)  = K_ref  exp(k_theta(x,y))
mu(x,y) = mu_ref exp(g_theta(x,y))
lambda(x,y) = K(x,y) - 2 mu(x,y) / 3
```

再换算展示：

```text
E  = 9 K mu / (3 K + mu)
nu = (3 K - 2 mu) / [2(3 K + mu)]
```

当前第一阶段解析实验使用单尺度材料场：

```text
K(x,y)  = K0  exp(m(x,y))
mu(x,y) = mu0 exp(m(x,y))
```

这个设置只能作为机制验证。正式论文主实验应至少包含独立 K/mu 的结果，否则审稿人可能质疑“泊松比被人为锁死，反问题过度简化”。

DeepXDE PFNN 弹性板示例的输出是：

```text
[u, v, sigma_xx, sigma_yy, sigma_xy]
```

本文正式模型建议输出：

```text
[u, v, sigma_xx, sigma_yy, sigma_xy, k, g]
```

其中 `k, g` 分别控制 `K` 和 `mu`。

## 4. 损失函数应该怎样写

### 4.1 不要把几何方程误写成独立损失

DeepXDE 官方弹性板 PFNN 并没有显式返回一个几何残差损失。几何关系被嵌入自动微分计算中：

```text
epsilon = sym(grad u)
```

也就是说，当前 PFNN 的损失不是：

```text
L = L_geometry + L_constitutive + L_equilibrium + ...
```

更准确的写法是：

```text
L_phys = L_equilibrium + L_stress_consistency
```

其中 `L_stress_consistency` 里面用到了几何关系。

如果以后网络额外输出应变：

```text
[u, v, epsilon_xx, epsilon_yy, epsilon_xy, sigma_xx, sigma_yy, sigma_xy, K, mu]
```

并且加入：

```text
epsilon_theta - sym(grad u_theta) = 0
```

那时才可以单独写 `L_geometry`。当前不建议这样增加复杂度。

### 4.2 平衡残差

网络输出应力：

```text
sigma_theta = [sigma_xx, sigma_yy, sigma_xy]
```

平衡方程残差：

```text
r_eq_x = partial sigma_xx / partial x + partial sigma_xy / partial y - f_x
r_eq_y = partial sigma_xy / partial x + partial sigma_yy / partial y - f_y
```

损失：

```text
L_eq = MSE(r_eq_x) + MSE(r_eq_y)
```

### 4.3 本构一致性残差

由位移自动微分得到应变：

```text
epsilon_theta = sym(grad d_theta)
```

由材料参数和应变计算本构应力：

```text
sigma_const = C[K_theta, mu_theta] : epsilon_theta
```

本构一致性残差：

```text
r_const = sigma_theta - sigma_const
```

损失：

```text
L_const = MSE(r_const_xx) + MSE(r_const_yy) + MSE(r_const_xy)
```

### 4.4 位移观测残差

观测点为 `x_i`，观测位移为：

```text
d_obs(x_i) = [u_obs, v_obs]
```

普通 PFNN 的观测损失：

```text
L_obs = MSE(d_theta(x_i) - d_obs(x_i))
```

本文的解耦方法中，观测残差要拆成边界可解释部分和材料可解释部分，详见第 6 节。

### 4.5 边界残差

已知底部边界：

```text
y = 0: u = 0, v = 0
```

顶部水平位移：

```text
y = 1: u = 0
```

顶部竖向位移存在不确定性：

```text
y = 1: v = b_beta(x)
```

边界损失：

```text
L_bc = MSE(u_theta_bottom)
     + MSE(v_theta_bottom)
     + MSE(u_theta_top)
     + MSE(v_theta_top - b_beta)
```

如果某个工况给定左右边界真实位移，则额外加入左右边界位移损失；否则不加入。

### 4.6 全局反力残差

顶部竖向总反力：

```text
R_y = integral_top sigma_yy(x,1) dx
```

数值积分：

```text
R_y_pred = sum_i w_i sigma_yy(x_i,1)
```

其中 `x_i` 是积分点，`w_i` 是积分权重。积分点不是反力传感器。

反力损失：

```text
L_R = [(R_y_pred - R_y_obs) / R_scale]^2
```

这里的观测信息只有一个总反力标量，不是 200 个反力观测点。

### 4.7 正则项

为了避免边界修正系数过度自由，建议加入小权重正则：

```text
L_beta = ||c||_2^2
```

为了避免 K/mu 出现高频假振荡，可选材料平滑正则：

```text
L_mat = ||grad log K||_2^2 + ||grad log mu||_2^2
```

材料正则不能过强，否则会抹掉真实软弱夹层或透镜体。正式实验必须做正则权重敏感性。

### 4.8 总损失

标准 PFNN baseline：

```text
L_total =
  w_eq    L_eq
+ w_const L_const
+ w_obs   L_obs
+ w_bc    L_bc
+ w_R     L_R
+ w_mat   L_mat
```

边界-材料解耦模型：

```text
L_total =
  w_eq     L_eq
+ w_const  L_const
+ w_obs_m  L_obs_material
+ w_obs_b  L_obs_boundary
+ w_bc     L_bc
+ w_R      L_R
+ w_beta   L_beta
+ w_mat    L_mat
```

其中：

```text
L_obs_material = 只让材料/状态网络解释不属于边界模式的残差
L_obs_boundary = 只让边界参数解释属于边界模式的残差
```

## 5. 低维边界参数化

### 5.1 为什么不能让每个边界点都自由学习

如果把顶部边界每个点的位移都作为未知量自由学习，边界自由度太高，模型几乎可以任意拟合观测数据。这会导致：

```text
边界解释过强；
材料反演失去物理意义；
审稿人会质疑这是过拟合，不是可识别性恢复。
```

因此边界误差必须低维化。

### 5.2 整体幅值 A

当前顶部边界基准形状：

```text
g0(x) = 0.15 + 0.04 sin(pi x) + 0.02 sin(2 pi x)
```

真实边界：

```text
v_top_true(x) = A_true g0(x), A_true = 1.0
```

错误边界：

```text
v_top_wrong(x) = 0.85 g0(x)
```

可学习幅值：

```text
v_top(x) = A_theta g0(x)
```

`A_theta` 表示整体幅值标定误差。

### 5.3 形状修正系数 c_i

真实边界误差不一定只是整体放大或缩小，也可能存在形状误差。例如：

```text
加载板轻微转动；
加载面接触不均匀；
左侧位移稍大、右侧稍小；
中部接触更强；
边界简化模型与真实试验夹具不一致。
```

因此可以把边界写成：

```text
b_beta(x) = A g0(x) + c1 phi1(x) + c2 phi2(x) + c3 phi3(x)
```

其中：

```text
beta = [A, c1, c2, c3]
```

`phi_i(x)` 是预先给定的低维边界形状基函数，`c_i` 是这些形状的权重。

推荐第一版使用：

```text
phi1(x) = 2x - 1              # 左右倾斜误差
phi2(x) = sin(pi x)           # 中部鼓起或凹陷误差
phi3(x) = sin(2 pi x)         # 左右双峰型误差
```

这不是唯一正确选择，但属于实验力学和 FEMU 中常见的低维边界正则化思想。更工程化的 FEM 算例中，可以使用：

```text
刚体运动模式；
低阶多项式模式；
POD 边界模式；
由参考 FEM 辅助问题得到的边界影响模式。
```

论文中必须说明：`c_i` 不是材料参数，也不是让边界任意自由，它们只是少数几个边界误差形状的控制旋钮。

## 6. 边界-材料解耦训练器

### 6.1 标准 PFNN 为什么会混淆

观测残差：

```text
r_u = d_theta(x_obs) - d_obs
```

这个残差可能来自：

```text
材料参数错误；
边界幅值错误；
边界形状错误；
观测噪声；
模型表达能力不足。
```

标准 PFNN 只看到 `r_u`，不会自动知道应该由材料场还是边界参数解释。因为材料场是空间函数，自由度比少数边界参数大，所以材料场很容易吸收边界错误，形成虚假软弱区或虚假硬化区。

### 6.2 为了做解耦，边界参数必须进入位移场

如果 `A, c_i` 只出现在顶部边界损失：

```text
v_theta(x,1) - b_beta(x)
```

那么内部观测点的瞬时预测 `d_theta(x_obs)` 对 `beta` 没有直接导数。此时很难严格把内部观测残差投影到边界参数方向。

因此，真正的解耦模型建议使用边界模式增强位移场：

```text
d_total(x,y) = d_core(x,y) + sum_j beta_j Psi_j(x,y)
```

其中：

```text
d_core      = PFNN 核心网络预测的位移场
Psi_j       = 第 j 个边界误差模式在域内的影响场
beta_j      = 对应边界模式系数
d_total     = 用于观测、边界、本构和平衡计算的总位移
```

这样，边界参数不只是边界损失中的标量，而是直接控制一种低维、物理可解释的位移模式。

### 6.3 Psi_j 如何得到

第一阶段解析算例可以先用简单运动学扩展：

```text
Psi_j^v(x,y) = y phi_j(x)
Psi_j^u(x,y) = 0
```

它满足底部为 0、顶部为 `phi_j(x)`。

正式 FEM 岩土算例更建议使用参考弹性辅助问题生成：

```text
对每个边界模式 phi_j，在参考均质材料上求解一次线弹性问题；
底部和其他已知边界按真实约束处理；
顶部施加单位边界模式 phi_j；
得到整个区域内的位移响应 Psi_j(x,y)。
```

这相当于用低维边界影响函数描述边界误差，比随意让网络学习整条边界更容易被审稿人接受。

### 6.4 残差投影是什么意思

把所有观测点上的边界影响模式组装成矩阵：

```text
Psi_obs = [Psi_1(x_obs), Psi_2(x_obs), ..., Psi_r(x_obs)]
```

观测残差：

```text
r = d_total(x_obs) - d_obs
```

边界模式子空间投影矩阵：

```text
P_beta = Psi_obs (Psi_obs^T Psi_obs + rho I)^(-1) Psi_obs^T
```

则：

```text
r_boundary = P_beta r
r_material = (I - P_beta) r
```

通俗解释：

```text
r_boundary 是长得像边界误差模式的那部分观测残差；
r_material 是去掉边界误差模式之后剩下的观测残差。
```

训练时：

```text
边界参数 beta 优先解释 r_boundary；
材料场 K/mu 主要解释 r_material；
```

这就是边界-材料解耦。

### 6.5 交替训练比单次联合训练更稳

建议训练流程：

```text
Stage 0: smoke 训练，确认所有损失和导数正常。

Stage 1: PFNN baseline 预训练。
  使用标准 L_eq + L_const + L_obs + L_bc。
  目标是获得基本位移/应力场。

Stage 2: 边界参数更新。
  冻结材料分支或强正则材料变化；
  更新 beta，使边界模式解释 r_boundary；
  同时使用 L_bc 和 L_R。

Stage 3: 材料场更新。
  固定 beta；
  更新 K/mu，使材料场解释 r_material；
  使用 L_eq、L_const、L_obs_material 和材料正则。

Stage 4: 小学习率联合微调。
  同时优化网络和 beta，但保留投影拆分损失。
```

如果直接所有变量一起训练，材料场仍然可能因为自由度高而优先吸收边界误差。

## 7. 模型结构选择

### 7.1 Baseline

必须保留 DeepXDE PFNN 作为 baseline：

```text
DeepXDE PFNN
输出 [u, v, sigma_xx, sigma_yy, sigma_xy, K/mu 或 m]
标准损失 L_eq + L_const + L_obs + L_bc + L_R
```

这用于证明标准 PFNN 在错误边界下会出现材料污染。

### 7.2 正式方法采用四模块框架

为了与 TBA-PINN 的博士论文主线形成承接，同时避免把本文写成普通 MLP 调参，正式方法采用：

```text
状态分支
材料分支
边界分支
解耦训练器
```

四个模块的职责必须清楚分开。

状态分支：

```text
输入: x, y
输出: d_core = [u_core, v_core], sigma = [sigma_xx, sigma_yy, sigma_xy]
作用: 表达满足力学方程的位移场和应力场。
```

材料分支：

```text
输入: x, y
输出: K(x,y), mu(x,y)
作用: 表达真实材料非均质性，而不是吸收边界误差。
```

边界分支：

```text
输入: 不一定需要 x, y；核心是少量可训练参数 beta = [A, c1, c2, ...]
输出: d_beta = sum_j beta_j Psi_j(x,y)
作用: 给边界误差一个低维、物理可解释的解释通道。
```

解耦训练器：

```text
输入: 观测残差 r_u、边界模式矩阵 Psi_obs、各损失项
输出: r_boundary, r_material，以及分阶段/交替优化策略
作用: 防止材料分支优先吸收边界误差。
```

总位移写成：

```text
d_total(x,y) = d_core(x,y) + d_beta(x,y)
```

物理损失中的应变由 `d_total` 计算：

```text
epsilon = sym(grad d_total)
```

这样边界分支不仅影响边界损失，也影响内部观测残差和物理残差。它才能真正参与“边界误差与材料误差拆分”。

### 7.3 与 DeepXDE PFNN 的关系

本文仍以 DeepXDE PFNN 作为 baseline。正式模型不是抛弃 PFNN，而是在 PFNN 思路上进行任务化分解：

```text
PFNN baseline: 多输出并行子网，所有解释权通过统一损失竞争。
本文方法: 状态、材料、边界分别有明确职责，并通过解耦训练器分配观测残差。
```

因此消融必须包含：

```text
1. 标准 DeepXDE PFNN。
2. PFNN + K/mu 双参数材料场。
3. 状态分支 + 材料分支，但无边界分支。
4. 状态分支 + 材料分支 + 边界分支，但无投影解耦。
5. 状态分支 + 材料分支 + 边界分支 + 解耦训练器。
6. 上述模型再加入全局反力。
```

这样可以回答三个问题：

```text
材料双参数化是否让问题更真实但更病态？
边界分支是否能吸收真实边界误差？
解耦训练器是否真的减少材料污染，而不仅是参数量增加带来的提升？
```

### 7.4 不建议作为主创新的网络堆叠

不建议把以下内容作为本文主创新：

```text
很深的 ResNet；
复杂 attention；
大规模 Fourier feature；
普通多分支网络；
普通自适应权重。
```

原因：

```text
这些容易把论文写成普通深度学习调参；
也容易与 TBA-PINN 的 Fourier、残差、多分支、梯度归一化重复；
更重要的是，它们不一定直接解决边界-材料混淆问题。
```

如果后续使用残差层或 Fourier feature，只作为表达能力增强消融，不能抢走主线。

### 7.5 与 TBA-PINN 的承接与区别

承接性：

```text
仍然是 PINN 反演；
仍然关注 K/mu 或弹性参数反演；
仍然保留物理方程、观测数据和材料场可视化。
从“材料场反演模型设计”自然延伸到“边界不确定条件下的材料场可识别性”。
```

区别：

```text
TBA-PINN 重点是材料场反演中的训练/结构增强；
本文重点是边界不确定性导致的可识别性破坏，以及边界误差与材料误差的解耦。
```

因此这篇不能把 TBA-PINN 的原有改进再包装一遍。

## 8. 已完成的解析 10 组实验

当前已完成的解析机制实验使用：

```text
DeepXDE PFNN
单尺度材料场 K=K0 exp(m), mu=mu0 exp(m)
seed = 42
150000 steps
PDE 内点 = 4000
边界配点 = 800
内部位移观测点 = 15 x 15 = 225
验证点 = 41 x 41
测试点 = 201 x 201
```

关键结果：

| 工况 | A 结果 | E 误差 | ux 误差 | uy 误差 | 总反力误差 |
|---|---:|---:|---:|---:|---:|
| Full boundary | 1.0000 | 5.91% | 2.25% | 0.39% | 4.36% |
| Full boundary + reaction | 1.0000 | 3.42% | 1.68% | 0.29% | 0.85% |
| Correct top | 1.0000 | 7.84% | 3.74% | 0.39% | 5.53% |
| Correct top + reaction | 1.0000 | 2.53% | 3.14% | 0.41% | 0.45% |
| Wrong fixed A | 0.8500 | 50.52% | 15.73% | 7.26% | 16.25% |
| Wrong fixed A + reaction | 0.8500 | 63.16% | 10.86% | 7.51% | 0.57% |
| Learnable A | 0.9977 | 7.05% | 3.62% | 0.39% | 5.38% |
| Learnable A + anchors | 1.0004 | 12.85% | 6.88% | 0.35% | 10.68% |
| Learnable A + reaction | 0.9980 | 3.67% | 3.29% | 0.32% | 0.35% |
| Learnable A + anchors + reaction | 1.0004 | 3.37% | 2.62% | 0.31% | 0.75% |

目前结论：

```text
1. 固定错误 A=0.85 会显著污染材料场。
2. 可学习 A 可以恢复整体边界尺度。
3. 反力能显著增强刚度尺度识别。
4. 如果 A 被强行固定错误，即使加入反力也不能恢复正确材料，甚至可能进一步扭曲材料场。
5. 当前单尺度材料和标准 PFNN 仍然只是机制证明，反演效果还不够作为顶刊主结果。
```

## 9. 下一阶段解析强化实验

在进入 FEM 之前，必须先把模型机制做扎实。建议按以下顺序执行。

### 9.1 双参数 K/mu 诊断

目的：

```text
判断从单尺度材料场扩展到 K/mu 独立场后，标准 PFNN 是否更容易发生边界-材料混淆。
```

工况：

```text
correct_top_amp_only, material_mode=dual
wrong_fixed_A, material_mode=dual
learnable_A, material_mode=dual
learnable_A_reaction, material_mode=dual
learnable_A_anchor_reaction, material_mode=dual
```

评价：

```text
RelL2_K, RelL2_mu, RelL2_E, RelL2_nu
A error
reaction error
材料场云图和误差图
```

### 9.2 边界形状错误实验

目的：

```text
证明只学习整体 A 不足以处理非均匀边界误差。
```

真值边界：

```text
v_top_true = g0(x) + 0.03 phi1(x) - 0.02 phi2(x)
```

模型对比：

```text
错误固定边界：只用 g0(x)
只学习 A：A g0(x)
学习 A + c_i：A g0(x) + c1 phi1 + c2 phi2 + c3 phi3
```

预期：

```text
只学习 A 能修复整体尺度误差，但不能修复形状误差；
A + c_i 能显著降低材料污染。
```

### 9.3 边界模式增强 PFNN

目的：

```text
验证边界参数直接进入位移场后，是否比只在边界损失中学习 A 更稳定。
```

模型：

```text
d_total = d_core + sum beta_j Psi_j
```

对比：

```text
PFNN + learnable A in BC only
PFNN + boundary mode layer
PFNN + boundary mode layer + reaction
```

### 9.4 投影解耦训练

目的：

```text
验证投影拆分 residual 后，材料场是否更少吸收边界误差。
```

对比：

```text
联合训练，不投影
交替训练，不投影
交替训练 + 投影解耦
交替训练 + 投影解耦 + 反力
```

核心指标：

```text
材料误差是否下降；
虚假软弱区是否减少；
A 和 c_i 是否接近真值；
反力误差是否下降；
不同 seed 是否稳定。
```

## 10. FEM 岩土主算例

解析实验完成后，进入 FEM 主算例。解析实验只证明机制，FEM 算例才是工程可信度的主证据。

推荐 FEM 场景：

```text
二维平面应变地基
基础宽度 B = 1.0 m
计算域宽度 = 6B
计算域深度 = 3B
顶部中心刚性加载板，宽度 = B
底部固定
左右侧水平约束、竖向自由
顶部非加载区自由
加载方式：位移控制
```

材料场：

```text
背景 E0 = 50 MPa, nu0 = 0.30
软弱夹层
局部软弱透镜体
可选平滑随机场
```

观测方式：

```text
地表沉降线
加载板附近少量边界位移锚点
1-3 条竖向测线
低分辨率 DIC-like 内部位移点
一个加载板竖向总反力
```

注意：

```text
DIC-like 是数值模拟观测，不是真实实验 DIC。
不能写成真实实验数据。
```

## 11. 对比方法

必须至少包含：

```text
DeepXDE PFNN baseline
PFNN + learnable boundary amplitude
PFNN + low-dimensional boundary shape parameters
PFNN + boundary mode layer
PFNN + boundary mode layer + projection decoupling
PFNN + boundary mode layer + projection decoupling + reaction
```

传统数值反演对比：

```text
FEMU-like low-dimensional material parameter inversion
FEMU-like KLE/Tikhonov material field inversion
可选 VFM/EGM 类方法，视实现难度决定
```

传统方法不是陪衬。若传统方法在低维、边界准确、材料分区已知时更好，应如实报告。本文要强调的是：

```text
高维材料场 + 边界错误/缺失 + 稀疏含噪观测时，如何避免边界误差污染材料场。
```

## 12. 图片与产物要求

每组实验至少保存：

```text
json/config.json
json/run_config.json
json/loss_history.json
txt/train.log
txt/driver.log
metrics/metrics.json
metrics/evaluation_metrics.json
metrics/validation_metrics.json
model/best_model.*
model/final_model.*
dat/loss_history.dat
dat/amplitude_history.dat
dat/observation_points.dat
dat/reaction_points.dat
dat/anchor_points.dat
npz/eval_fields.npz
npz/loss_history.npz
png/loss_history.png
png/amplitude_history.png
png/sampling_layout.png
png/*field*.png
png/top_uy_curve.png
png/top_reaction_density_curve.png
```

当前正式代码实际使用中文图名，同时保存 `png` 和 `pdf`，核心包括：

```text
png/损失历史图.png
png/损失分量图.png
png/材料参数演化图.png
png/边界参数演化图.png
png/采样点分布图.png
png/评估_K与μ真值预测对比图.png
png/评估_*_真值预测误差图.png
png/验证_*_真值预测误差图.png
png/位移矢量场图_评估集.png
png/切片对比图_y0.50.png
png/材料参数切片对比图_y0p25.png
png/材料参数切片对比图_y0p50.png
png/材料参数切片对比图_y0p75.png
png/顶部竖向位移曲线图.png
png/顶部反力密度曲线图.png
png/应力分支顶部反力密度曲线图.png
png/本构应力顶部反力密度曲线图.png
txt/关键指标摘要.txt
```

汇总图目录要保持清晰：

```text
exp/01.AnalyticalBenchmark/07.ComparisonSummary/
  image/
  csv/
```

不要再生成繁杂的多层临时汇总目录。

## 13. 长跑规则

```text
禁止在本地电脑跑 PINN 长实验。
长实验必须在远程计算节点 GPU/DCU 上跑。
长实验前必须先 smoke。
启动长实验后必须给预计完成时间。
判断任务是否 live，必须同时检查活进程、日志时间戳、step 是否增长。
不能只看 screen 是否存在。
不能用旧日志或历史 screen.log 当作当前状态。
不能重发同名实验覆盖正在运行的任务。
失败重跑前必须保留失败目录或失败快照。
```

## 14. 立即执行顺序

当前不建议马上进入 FEM。先做解析强化，因为现有标准 PFNN 的反演效果还不够强。

立即顺序：

```text
1. 实现三分支代码骨架：状态分支、材料分支、边界分支，但先关闭边界分支，确保可退化为 PFNN baseline。
2. 做 smoke：1000-3000 steps，检查输出维度、损失项、K/mu 正值、A/c_i 记录、图片和 metrics 是否正常。
3. 跑 dual K/mu 的 5 组关键解析诊断，先判断标准 PFNN 在更真实材料自由度下是否更容易混淆。
4. 打开边界分支，先只学习 A，复现当前 learnable_A 和 learnable_A_reaction 结论。
5. 增加 A + c_i 边界形状参数，构造非均匀边界误差实验。
6. 实现 boundary mode layer，使 beta 直接进入位移场 d_total = d_core + d_beta。
7. 实现投影/交替解耦训练器。
8. 做解耦消融：无边界分支、有边界分支但无投影、有投影解耦、有投影解耦 + 反力。
9. 当解析机制稳定后，再进入 FEM 岩土主算例。
```

阶段性目标：

```text
在无噪声解析算例中，正确边界或可恢复边界工况下，K/mu/E 的相对误差应明显低于当前标准 PFNN。
在错误边界工况下，标准 PFNN 应产生明显材料污染；解耦模型应显著降低这种污染。
```

## 15. A13-A15 当前诊断结论与下一步 smoke

已经检查 DeepXDE 官方弹性板 PFNN 示例和旧 TBA-PINN 相关实现。标准混合形式通常让网络输出：

```text
[u, v, sigma_xx, sigma_yy, sigma_xy]
```

平衡方程使用网络应力分支：

```text
partial sigma_xx / partial x + partial sigma_xy / partial y = f_x
partial sigma_xy / partial x + partial sigma_yy / partial y = f_y
```

本构一致性再把网络应力与由位移梯度、材料参数计算出的本构应力拉近。这个写法对正问题和低维常数材料反演是常见且合理的，但对本文的空间变 `K(x,y), mu(x,y)` 反演有一个关键风险：

```text
平衡残差主要训练应力分支；
总反力若也用网络应力分支积分，则反力残差也主要训练应力分支；
材料分支 K、mu 主要只通过本构一致性间接获得梯度。
```

3000 步梯度诊断已经验证这个问题。旧 `mixed + stress` 形式中，`momentum` 和 `reaction` 对材料分支的梯度接近 `1e-15`。因此 A13-A15 效果差不能简单归因于“模型结构不行”，首先是物理残差到材料分支的梯度路径不正确。

正式方法需要区分两种平衡形式：

```text
mixed:
  平衡方程使用网络应力分支 sigma_theta。
  这是 DeepXDE/PFNN 的强 baseline 和失败消融。

direct:
  先由预测位移梯度和预测 K、mu 计算本构应力 sigma_const。
  再对 sigma_const 求散度进入平衡方程。
  这样平衡残差会直接训练材料分支。
```

也需要区分两种反力形式：

```text
stress:
  顶部总反力由网络应力分支 sigma_yy 积分。
  适合作为传统混合形式 baseline。

constitutive:
  顶部总反力由预测位移梯度和预测 K、mu 计算出的 sigma_yy_const 积分。
  这是本文方法中更合理的反力约束，因为它直接约束材料刚度尺度。
```

下一步不再盲目跑 A0-A15 全部组合，而先跑关键 smoke：

| 运行名 | 工况 | 平衡形式 | 反力形式 | 目的 |
|---|---|---|---|---|
| A5 | PFNN + 可学习 A + 锚点 + 反力 | mixed | stress | 最强传统 PFNN 基线 |
| A13 | 边界分支 + 解耦训练器，无锚点无反力 | direct | stress | 单独检查解耦训练器 |
| A14 | A13 + 反力 | direct | constitutive | 检查本构反力是否恢复刚度尺度 |
| A15 | A14 + 锚点，反力权重 5 | direct | constitutive | 当前候选最终方法 |
| A15_mixed | A15 但退回旧形式 | mixed | stress | 失败消融，证明旧梯度路径不足 |

这 5 组先跑 3000 步 smoke。smoke 通过标准不是材料误差已经足够低，而是：

```text
1. 所有组能完成训练并保存完整产物；
2. direct 组的 momentum 对材料分支梯度非零；
3. constitutive 反力组的 reaction 对材料分支梯度非零；
4. 中文动态图、材料演化图、K/mu 对比图、两类反力图正常生成；
5. A15 direct/constitutive 的材料误差趋势优于 A15_mixed。
```
