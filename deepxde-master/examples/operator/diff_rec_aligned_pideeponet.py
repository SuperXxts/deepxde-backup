"""
扩散-反应方程的物理信息DeepONet（PIDeepONet）示例

⚠️ 重要区别：传统DeepONet vs PIDeepONet
===========================================
传统DeepONet（有监督学习）：
  - 需要输入函数u和对应的真实输出函数G(u)的样本对
  - 使用 dde.data.Triple 或 dde.data.TripleCartesianProd
  - Loss函数：MSE(y_true, y_pred)，其中y_true是真实的G(u)
  - 就像：给你很多题目和标准答案，学习答案

PIDeepONet（物理信息DeepONet，无监督学习）：
  - 不需要真实的G(u)，只需要PDE方程
  - 使用 dde.data.PDEOperatorCartesianProd（本代码使用）
  - Loss函数：MSE(0, PDE残差)，通过物理约束学习
  - 就像：给你很多题目和规则（方程），学习满足规则

本代码使用的是PIDeepONet！
===========================================

这个脚本演示了如何使用PIDeepONet学习扩散-反应方程的解算子。
方程形式：u_t - D*u_xx + k*u^2 = v(x,t)
其中 v(x,t) 是源项（输入函数），u(x,t) 是解（输出函数）。

PIDeepONet学习的是从输入函数v(x)到解函数u(x,t)的映射（算子），
但不需要真实的u(x,t)作为标签，只需要PDE方程作为约束。

Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle
"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# 添加utils目录到路径，用于导入工具函数
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils'))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
from utils.save_results import save_all_results
from utils.device_utils import print_gpu_info, set_random_seed
from utils.loss_callback import LossHistoryCallback

from ADR_solver import solve_ADR


# ============================================================================
# 0. 设置随机种子（用于结果可复现）
# ============================================================================

# 设置随机种子，确保实验结果可复现
# 注意：固定seed可能会降低训练速度，但能保证结果的一致性
# 如果不需要可复现性，可以设置为None
RANDOM_SEED = 42  # 可以修改为任意整数
set_random_seed(RANDOM_SEED)


# ============================================================================
# 重要说明：传统DeepONet vs PIDeepONet
# ============================================================================
#
# 传统DeepONet（有监督学习）：
#   - 需要输入函数u和对应的真实输出函数G(u)的样本对
#   - 使用 dde.data.Triple 或 dde.data.TripleCartesianProd
#   - Loss函数：MSE(y_true, y_pred)，其中y_true是真实的G(u)
#   - 就像：给你很多题目和标准答案，学习答案
#
# PIDeepONet（物理信息DeepONet，本代码使用）：
#   - 不需要真实的G(u)，只需要PDE方程
#   - 使用 dde.data.PDEOperatorCartesianProd（本代码使用）
#   - Loss函数：MSE(0, PDE残差)，通过物理约束学习
#   - 就像：给你很多题目和规则（方程），学习满足规则
#
# 本代码使用的是PIDeepONet，所以是无监督学习（基于物理约束）！
#
# ============================================================================
# 1. 定义PDE方程（物理方程）
# ============================================================================
def pde(x, y, v):
    """
    定义扩散-反应方程：u_t - D*u_xx + k*u^2 = v(x,t)
    
    这是一个时间相关的偏微分方程，描述物理量的扩散和反应过程。
    
    参数:
        x: 空间-时间坐标，形状为 (N, 2)
           - x[:, 0] 是空间坐标（0到1之间）
           - x[:, 1] 是时间坐标（0到1之间）
        y: 网络预测的解 u(x,t)，形状为 (N,)
           这是DeepONet的输出，表示在点x处的解值
        v: 源项函数 v(x,t)，形状为 (N,)
           这是输入函数在点x处的值，作为PDE的右端项
    
    返回:
        PDE残差：dy_t - D * dy_xx + k * y^2 - v
        如果网络预测正确，这个值应该接近0
    """
    D = 0.01  # 扩散系数（diffusion coefficient）
    k = 0.01  # 反应系数（reaction coefficient）
    
    # 计算时间导数：∂u/∂t
    # jacobian(y, x, j=1) 表示对x的第1维（时间维度）求一阶导数
    dy_t = dde.grad.jacobian(y, x, j=1)
    
    # 计算空间二阶导数：∂²u/∂x²
    # hessian(y, x, j=0) 表示对x的第0维（空间维度）求二阶导数
    dy_xx = dde.grad.hessian(y, x, j=0)
    
    # PDE方程：u_t - D*u_xx + k*u^2 - v = 0
    # 如果预测正确，残差应该为0
    return dy_t - D * dy_xx + k * y**2 - v


# ============================================================================
# 2. 定义几何域和边界条件
# ============================================================================

# 定义空间域：一维区间 [0, 1]
geom = dde.geometry.Interval(0, 1)

# 定义时间域：时间区间 [0, 1]
timedomain = dde.geometry.TimeDomain(0, 1)

# 组合成空间-时间域
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# 定义边界条件：边界上的值为0（Dirichlet边界条件）
# lambda _: 0 表示边界值恒为0
# lambda _, on_boundary: on_boundary 表示判断点是否在边界上
bc = dde.icbc.DirichletBC(geomtime, lambda _: 0, lambda _, on_boundary: on_boundary)

# 定义初始条件：t=0时刻的值为0
# lambda _, on_initial: on_initial 表示判断点是否在初始时刻
ic = dde.icbc.IC(geomtime, lambda _: 0, lambda _, on_initial: on_initial)

# ============================================================================
# 3. 定义PDE数据对象（包含训练点和测试点）
# ============================================================================
pde = dde.data.TimePDE(
    geomtime,      # 空间-时间域
    pde,           # PDE方程函数
    [bc, ic],      # 边界条件和初始条件列表
    num_domain=200,    # 域内训练点数：用于计算PDE损失
    num_boundary=40,   # 边界训练点数：用于计算边界条件损失
    num_initial=20,    # 初始条件训练点数：用于计算初始条件损失
    num_test=500,      # 测试点数：用于评估模型性能（不参与训练）
)

# ============================================================================
# 4. 定义函数空间（输入函数的空间）
# ============================================================================

# 使用高斯随机场（Gaussian Random Field）作为函数空间
# length_scale=0.2 控制函数的平滑程度，值越小函数变化越快
func_space = dde.data.GRF(length_scale=0.2)

# ============================================================================
# 5. 定义PDE算子数据（用于训练DeepONet）
# ============================================================================

# 评估点：用于离散化输入函数v(x)，作为branch net的输入
# 在空间[0,1]上均匀采样50个点
eval_pts = np.linspace(0, 1, num=50)[:, None]  # 形状：(50, 1)

# 创建PDE算子数据对象
# 这个对象会：
# 1. 从函数空间中随机采样1000个函数v(x)
# 2. 对每个函数，在训练点上计算PDE损失
# 3. 组织成DeepONet需要的格式（对齐格式）
data = dde.data.PDEOperatorCartesianProd(
    pde,                    # PDE对象
    func_space,             # 函数空间
    eval_pts,               # 评估点（50个点）
    1000,                   # 训练函数数量：1000个不同的v(x)
    function_variables=[0], # 函数变量索引：[0]表示函数只依赖于空间x（不依赖时间t）
    num_test=100,           # 测试函数数量：100个函数用于测试
    batch_size=50           # 批量大小：每次训练使用50个函数
)

# ============================================================================
# 6. 定义DeepONet网络结构
# ============================================================================

# 创建DeepONet网络（对齐格式）
net = dde.nn.DeepONetCartesianProd(
    [50, 128, 128, 128],    # Branch Net结构：输入50维 → 128 → 128 → 输出128维
                            # 输入：函数v在50个评估点的值
    [2, 128, 128, 128],     # Trunk Net结构：输入2维 → 128 → 128 → 输出128维
                            # 输入：空间-时间坐标(x, t)
    "tanh",                 # 激活函数：tanh（隐藏层使用，输出层不用）
    "Glorot normal",        # 权重初始化方法：Glorot正态分布初始化
)

# ============================================================================
# 7. 创建模型并编译
# ============================================================================

# 创建模型：将数据和网络组合
model = dde.Model(data, net)

# 检查GPU使用情况（仅检查PyTorch）
# DeepXDE会自动检测并使用GPU（如果可用），无需手动配置
# 对于PyTorch后端：如果检测到CUDA，会自动设置默认设备为cuda
print_gpu_info()

# 编译模型：配置优化器、学习率、损失函数和评估指标
# "adam": 使用Adam优化器
# lr=0.0005: 学习率为0.0005
# loss="MSE": 损失函数（默认值，可省略）
#   重要说明：
#   - 训练loss和测试loss都使用相同的损失函数（默认是MSE）
#   - 可用的loss包括："MSE", "MAE", "mean l2 relative error"等
# metrics: 评估指标（对于算子学习问题，通常不需要）
#   重要说明：
#   - 算子学习是无监督学习，测试集没有真实标签（y_test=None）
#   - 因此不能使用需要真实标签的metrics（如"MSE", "l2 relative error"等）
#   - 如果没有指定metrics，训练输出中的"Test metric"列会是空的
#   - 但Test loss仍然会计算并显示（使用MSE），这已经足够评估模型性能
#   - Test loss通过PDE残差、边界条件误差等计算，不需要真实解
#   如果需要评估指标，可以在训练后手动计算（使用数值解作为参考）
model.compile("adam", lr=0.0005, loss="MSE", metrics=None)

# ============================================================================
# 8. 准备模型配置（用于保存）
# ============================================================================

# 保存模型的所有超参数和配置信息
config = {
    "model_type": "DeepONetCartesianProd",
    "branch_net": [50, 128, 128, 128],
    "trunk_net": [2, 128, 128, 128],
    "activation": "tanh",
    "kernel_initializer": "Glorot normal",
    "optimizer": "adam",
    "learning_rate": 0.0005,
    "metrics": None,  # 评估指标（算子学习问题通常不需要，因为y_test=None）
    "iterations": 40000,
    "num_domain": 200,
    "num_boundary": 40,
    "num_initial": 20,
    "num_test": 500,
    "num_function": 1000,
    "batch_size": 50,
    "eval_points": 50,
    "function_space": "GRF",
    "length_scale": 0.2,
    "random_seed": RANDOM_SEED,  # 保存随机种子，便于复现
}

# ============================================================================
# 9. 训练模型
# ============================================================================

# ============================================================================
# 训练、验证、测试函数的调用说明
# ============================================================================
# 
# 在DeepXDE中，训练、验证、测试都在 model.train() 内部自动完成：
#
# 1. 训练函数（_train_step）：
#    - 位置：model.train() -> _train_sgd() -> _train_step()
#    - 调用时机：每次迭代都调用
#    - 功能：
#      * 从训练数据中采样一个batch（50个函数）
#      * 前向传播：branch net处理函数值，trunk net处理空间-时间点
#      * 计算损失：PDE损失 + BC损失 + IC损失
#      * 反向传播：计算梯度
#      * 更新权重：使用Adam优化器更新网络参数
#
# 2. 测试函数（_test）：
#    - 位置：model.train() -> _train_sgd() -> _test()
#    - 调用时机：
#      * 训练开始前调用一次（初始测试）
#      * 每 display_every 次迭代调用一次（默认1000次）
#      * 最后一次迭代后调用
#    - 功能：
#      * 在测试数据上评估模型性能（100个测试函数）
#      * 计算测试损失（不更新权重）
#      * 计算测试指标（如L2相对误差）
#      * 更新最佳模型（如果当前测试损失更小）
#      * 记录到losshistory中
#
# 3. 验证：
#    - DeepXDE中没有单独的验证集
#    - 测试集（test set）同时用于：
#      * 训练过程中的验证（监控过拟合）
#      * 最终模型评估
#    - 测试数据不参与训练，只用于评估
#
# 训练流程示例（20000次迭代）：
#   Iteration 0:    调用 _test() [初始测试]
#   Iteration 1-999:   只调用 _train_step() [训练]
#   Iteration 1000: 调用 _test() [测试] + _train_step() [训练]
#   Iteration 1001-1999: 只调用 _train_step() [训练]
#   Iteration 2000: 调用 _test() [测试] + _train_step() [训练]
#   ...
#   Iteration 20000: 调用 _test() [最终测试]
#
# ============================================================================

# 设置保存目录（用于实时保存loss可视化）
save_dir = "/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master/exp/deeponet2"

# 创建实时保存loss可视化的callback
# 每1000次迭代（display_every的默认值）会自动更新并保存loss图片
loss_callback = LossHistoryCallback(
    save_dir=save_dir,
    period=1000,  # 每1000次迭代保存一次（与display_every一致）
    filename="loss_history.png"
)

# 调用训练函数
# 这个函数内部会自动调用训练和测试函数
# callbacks参数：传入callback列表，用于在训练过程中执行额外操作
losshistory, train_state = model.train(iterations=20000, callbacks=[loss_callback])

# ============================================================================
# 10. 预测阶段：使用训练好的模型进行预测
# ============================================================================

# 10.1 生成一个新的输入函数v(x)
# 从函数空间中随机采样1个函数
func_feats = func_space.random(1)  # 生成1个函数的特征

# 在100个空间点上计算函数值（用于生成真实解）
xs = np.linspace(0, 1, num=100)[:, None]
v = func_space.eval_batch(func_feats, xs)[0]  # 形状：(100,)

# 10.2 使用数值方法求解真实解（用于最终评估对比）
# 
# ⚠️ 重要说明：训练、测试、验证、评估的区别
# ============================================
# 
# 1. 训练阶段（Training）：
#    - 使用训练数据（1000个函数）
#    - 模型通过PDE残差、边界条件误差等学习
#    - 训练数据中没有真实标签（y_train=None）
#    - 模型学习算子映射：v(x) -> u(x,t)
#
# 2. 测试阶段（Test，训练过程中）：
#    - 使用测试数据（100个函数）
#    - 在测试数据上计算Test loss（PDE残差、BC误差等）
#    - 测试数据中也没有真实标签（y_test=None）
#    - 用于监控训练过程，不更新权重
#    - 每1000次迭代调用一次
#
# 3. 验证（Validation）：
#    - DeepXDE中没有单独的验证集
#    - Test set同时用于验证（监控过拟合）
#    - 所以Test loss既用于测试，也用于验证
#
# 4. 最终评估（Evaluation，训练完成后）：
#    - 这是训练完成后的独立评估步骤
#    - 使用数值求解器（solve_ADR）计算真实解作为"标准答案"
#    - 这是评估方法，不是训练/测试数据！
#    - 用于验证模型是否学到了正确的算子映射
#    - 就像考试时需要有标准答案来评分一样
#
# 为什么可以用数值方法？
#    - 对于这个扩散-反应方程，可以用有限差分等数值方法精确求解
#    - 数值解可以作为"真实解"的近似（网格足够密时误差很小）
#    - 这样就能评估DeepONet的预测是否准确
#
# 总结：
#    - 训练/测试：无监督，不需要真实解
#    - 最终评估：有监督（用数值方法计算真实解），用于验证效果
#
# ============================================
#
# solve_ADR 是数值求解器，用于计算扩散-反应方程的真实解
# 参数说明：
#   - 前4个参数：空间域[0,1]和时间域[0,1]
#   - lambda x: 0.01*ones: 扩散系数k(x) = 0.01
#   - lambda x: zeros: 对流速度v(x) = 0
#   - lambda u: 0.01*u^2: 反应项g(u) = 0.01*u^2
#   - lambda u: 0.02*u: 反应项导数dg/du = 0.02*u
#   - lambda x,t: ...: 源项f(x,t) = v(x)（我们生成的函数）
#   - lambda x: zeros: 初始条件u(x,0) = 0
#   - 100, 100: 空间和时间网格点数
x, t, u_true = solve_ADR(
    0, 1,  # 空间域 [0, 1]
    0, 1,  # 时间域 [0, 1]
    lambda x: 0.01 * np.ones_like(x),      # 扩散系数 k(x) = 0.01
    lambda x: np.zeros_like(x),            # 对流速度 v(x) = 0
    lambda u: 0.01 * u**2,                # 反应项 g(u) = 0.01*u^2
    lambda u: 0.02 * u,                   # 反应项导数 dg/du = 0.02*u
    lambda x, t: np.tile(v[:, None], (1, len(t))),  # 源项 f(x,t) = v(x)
    lambda x: np.zeros_like(x),            # 初始条件 u(x,0) = 0
    100,  # 空间网格点数
    100,  # 时间网格点数
)
u_true = u_true.T  # 转置，形状变为 (100, 100)：100个时间点 × 100个空间点

# 10.3 准备模型预测所需的输入
# 在50个评估点上计算函数值（与训练时一致）
v_branch = func_space.eval_batch(func_feats, np.linspace(0, 1, num=50)[:, None])
# v_branch形状：(1, 50) - 1个函数在50个评估点的值

# 创建空间-时间网格点（100×100=10000个点）
xv, tv = np.meshgrid(x, t)  # xv和tv都是(100, 100)
x_trunk = np.vstack((np.ravel(xv), np.ravel(tv))).T
# x_trunk形状：(10000, 2) - 10000个点，每个点有(x, t)两个坐标

# 10.4 使用训练好的模型进行预测
# model.predict() 会：
#   1. 将v_branch输入到branch net，得到branch输出
#   2. 将x_trunk输入到trunk net，得到trunk输出
#   3. 通过点积合并两个输出，得到最终预测
u_pred = model.predict((v_branch, x_trunk))
# u_pred形状：(1, 10000) - 1个函数在10000个点的预测值

# 重塑为网格形状，便于可视化
u_pred = u_pred.reshape((100, 100))  # 形状：(100, 100)

# ============================================================================
# 11. 保存所有训练和预测结果
# ============================================================================

# 注意：save_dir已经在训练前定义，用于实时保存loss可视化
# 这里直接使用同一个目录保存所有结果

# ============================================================================
# 模型效果评估说明
# ============================================================================
# 
# 1. 可视化图的样式说明：
# ============================================
# 
# u_true.png 和 u_pred.png 都是热力图（heatmap）：
#   - X轴：空间坐标 x（0到1）
#   - Y轴：时间坐标 t（0到1）
#   - 颜色：表示 u(x,t) 的值
#     * 蓝色/紫色：值较小
#     * 绿色/黄色：值中等
#     * 红色：值较大
#   - 颜色条（colorbar）：显示数值范围
#
# 例如：
#   - 如果真实解在某个区域是红色（值大），预测解也应该是红色
#   - 如果两者颜色分布相似，说明预测准确
#   - 如果颜色分布差异很大，说明预测不准确
#
# error_map.png：
#   - 同样也是热力图
#   - 颜色表示误差大小：|u_true - u_pred|
#   - 红色区域：误差大（预测不准）
#   - 蓝色区域：误差小（预测准确）
#
# comparison.png：
#   - 四宫格布局
#   - 左上：真实解热力图
#   - 右上：预测解热力图
#   - 左下：误差分布热力图
#   - 右下：数值统计信息（L2误差、最大误差等）
#
# 2. 数值评估：
#    - L2相对误差：通过 save_error_info() 计算并保存
#    - 最大/平均绝对误差：在 comparison.png 中显示
#    - 均方根误差：在 comparison.png 中显示
#
# 3. 如何查看效果：
#    - 打开 comparison.png：可以同时看到真实解、预测解、误差分布和统计信息
#    - 对比 u_true.png 和 u_pred.png：直观比较预测是否准确
#    - 查看 error_map.png：误差大的区域会显示为红色（热力图）
#    - 查看 error_info.json：包含详细的数值评估指标
#
# 4. 评估标准：
#    - L2相对误差 < 1%：模型效果很好
#    - L2相对误差 < 5%：模型效果较好
#    - L2相对误差 < 10%：模型效果一般
#    - L2相对误差 > 10%：需要改进模型或训练参数
#
# ============================================================================

# 调用工具函数保存所有结果，包括：
#   - 模型配置（model_config.json）
#   - Loss历史数据（loss_history.dat）和可视化（loss_history.png）
#   - 模型检查点（model_checkpoint-*.pt）
#   - 预测结果数据（u_true.npy, u_pred.npy等）
#   - 可视化图像：
#     * u_true.png: 真实解热力图
#     * u_pred.png: 预测解热力图
#     * error_map.png: 误差分布热力图
#     * comparison.png: 四宫格对比图（推荐查看这个！）
#   - 误差信息（error_info.json）
save_all_results(
    model=model,           # 训练好的模型
    losshistory=losshistory,  # 损失历史
    train_state=train_state,  # 训练状态
    u_true=u_true,        # 真实解（形状：(100, 100)）
    u_pred=u_pred,        # 预测解（形状：(100, 100)）
    config=config,        # 模型配置
    x_trunk=x_trunk,      # 空间-时间点坐标（形状：(10000, 2)）
    v_branch=v_branch,    # 输入函数值（形状：(1, 50)）
    save_dir=save_dir,    # 保存目录
    x=x,                  # 空间坐标数组（用于设置热力图坐标轴）
    t=t                   # 时间坐标数组（用于设置热力图坐标轴）
)
