"""
================================================================================
问题描述：2D线性弹性板问题（Linear Elasticity Plate Problem）
================================================================================

【问题背景】
    本代码求解一个2D线性弹性力学问题，计算在给定外力作用下弹性板的位移场和应力场。

【力的施加方向】
    1. 主要边界载荷：在上边界（y=1）施加y方向的正应力
       - Syy = (2*mu + lambda) * Q * sin(πx)
       - 这是垂直方向（y方向）的拉力或压力，沿x方向呈正弦分布
       - 这是导致板变形的主要原因
    2. 边界约束：
       - 左、右边界：Sxx = 0（无x方向表面力）
       - 各边界有位移约束（ux=0或uy=0）
    3. 体积力：在域内每一点都作用有体积力
       - fx(x,y)：x方向的体积力（非零）
       - fy(x,y)：y方向的体积力（非零）
    4. 总结：主要外力在y方向（垂直方向），通过上边界的正应力施加

【物理问题】
    1. 求解域：单位正方形板 [0,1] × [0,1]
    2. 未知量：
       - 位移场：u = (ux, uy)  [x方向和y方向的位移]
       - 应力场：σ = (Sxx, Syy, Sxy)  [正应力和剪应力]
    3. 控制方程：
       a) 应变-位移关系（几何关系）：
          E_xx = ∂ux/∂x          (x方向正应变)
          E_yy = ∂uy/∂y          (y方向正应变)
          E_xy = 0.5*(∂ux/∂y + ∂uy/∂x)  (剪应变)
       
       b) 应力-应变关系（本构关系，胡克定律）：
          Sxx = E_xx*(2*mu + lambda) + E_yy*lambda
          Syy = E_yy*(2*mu + lambda) + E_xx*lambda
          Sxy = 2*mu*E_xy
          其中：lambda (lmbd) = 拉梅常数，mu = 剪切模量
       
       c) 平衡方程（动量守恒）：
          ∂Sxx/∂x + ∂Sxy/∂y = fx  (x方向力平衡)
          ∂Sxy/∂x + ∂Syy/∂y = fy  (y方向力平衡)
          其中：fx, fy 是体积力（body force）

【求解方法】
    使用PINN（Physics-Informed Neural Network）方法：
    1. 神经网络结构：PFNN（Physics-Informed Feedforward Neural Network）
    2. 输入：空间坐标 (x, y)
    3. 输出：5个物理量 [ux, uy, Sxx, Syy, Sxy]
    4. 损失函数：
       - PDE残差：平衡方程和本构关系的残差
       - 边界条件误差：边界上的位移和应力约束
    5. 边界条件处理方式：
       - 软边界（soft BC）：通过DirichletBC显式施加，需要优化器学习满足
       - 硬边界（hard BC）：通过输出变换函数自动满足，无需优化器学习

【参考文献】
    Paper: https://doi.org/10.1016/j.cma.2021.113741
    Reference: https://github.com/sciann/sciann-applications/blob/master/SciANN-Elasticity/Elasticity-Forward.ipynb

Backend supported: pytorch, jax, paddle
================================================================================
"""
import deepxde as dde
import numpy as np
import sys
import os

# 添加utils目录到路径
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../utils'))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.device_utils import print_gpu_info, set_random_seed
from utils.save_results import (
    plot_and_save_loss_history,
    plot_all_elasticity_fields_with_train,
    plot_deformed_mesh,
    plot_displacement_vector_field
)

# ============================================================================
# 0. 设置随机种子（用于结果可复现）
# ============================================================================
RANDOM_SEED = 42
set_random_seed(RANDOM_SEED)
print(f"Random seed set to: {RANDOM_SEED} (for reproducibility)")
print_gpu_info()

# ============================================================================
# 材料参数（Material Parameters）
# ============================================================================
lmbd = 1.0  # 拉梅常数（Lamé constant）lambda，用于计算体积模量
mu = 0.5    # 剪切模量（Shear modulus），用于计算剪切应力
Q = 4.0     # 问题参数，用于构造精确解和边界条件

# ============================================================================
# 后端函数定义（Backend Functions）
# ============================================================================
# 使用后端无关的函数，确保代码可以在不同后端（PyTorch, JAX, Paddle）上运行
sin = dde.backend.sin
cos = dde.backend.cos
stack = dde.backend.stack  # 用于堆叠张量

# ============================================================================
# 几何定义（Geometry Definition）
# ============================================================================
# 定义求解域：单位正方形 [0,1] × [0,1]
geom = dde.geometry.Rectangle([0, 0], [1, 1])

# ============================================================================
# 边界条件类型选择（Boundary Condition Type）
# ============================================================================
# "hard": 硬边界条件，通过输出变换函数自动满足边界条件（更高效）
# "soft": 软边界条件，通过DirichletBC显式施加边界条件（更灵活）
BC_type = ["hard", "soft"][0]  # 当前选择硬边界条件


# ============================================================================
# 边界识别函数（Boundary Identification Functions）
# ============================================================================
# 这些函数用于识别不同的边界，返回True表示该点在指定边界上

def boundary_left(x, on_boundary):
    """
    识别左边界：x = 0
    Args:
        x: 空间坐标点 [x, y]
        on_boundary: 布尔值，表示该点是否在边界上
    Returns:
        True: 如果点在左边界上（x=0）
    """
    return on_boundary and dde.utils.isclose(x[0], 0.0)


def boundary_right(x, on_boundary):
    """
    识别右边界：x = 1
    """
    return on_boundary and dde.utils.isclose(x[0], 1.0)


def boundary_top(x, on_boundary):
    """
    识别上边界：y = 1
    """
    return on_boundary and dde.utils.isclose(x[1], 1.0)


def boundary_bottom(x, on_boundary):
    """
    识别下边界：y = 0
    """
    return on_boundary and dde.utils.isclose(x[1], 0.0)


# ============================================================================
# 精确解函数（Exact Solution Function）
# ============================================================================
# 注意：这个函数仅用于验证和评估PINN的精度，不是训练所必需的
# 在实际工程问题中，通常没有精确解，只能依赖PINN求解
def func(x):
    """
    计算精确解：位移场和应力场
    Args:
        x: 空间坐标点，形状为 (N, 2)，其中 N 是点的数量
    Returns:
        np.array: 形状为 (N, 5)，包含 [ux, uy, Sxx, Syy, Sxy]
    """
    # 1. 计算位移场（Displacement Field）
    # x方向位移：ux(x,y) = cos(2πx) * sin(πy)
    ux = np.cos(2 * np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2])
    # y方向位移：uy(x,y) = sin(πx) * Q * y^4 / 4
    uy = np.sin(np.pi * x[:, 0:1]) * Q * x[:, 1:2] ** 4 / 4

    # 2. 计算应变场（Strain Field）
    # x方向正应变：E_xx = ∂ux/∂x
    E_xx = -2 * np.pi * np.sin(2 * np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2])
    # y方向正应变：E_yy = ∂uy/∂y
    E_yy = np.sin(np.pi * x[:, 0:1]) * Q * x[:, 1:2] ** 3
    # 剪应变：E_xy = 0.5*(∂ux/∂y + ∂uy/∂x)
    E_xy = 0.5 * (
        np.pi * np.cos(2 * np.pi * x[:, 0:1]) * np.cos(np.pi * x[:, 1:2])
        + np.pi * np.cos(np.pi * x[:, 0:1]) * Q * x[:, 1:2] ** 4 / 4
    )

    # 3. 计算应力场（Stress Field）
    # 使用本构关系（胡克定律）计算应力
    Sxx = E_xx * (2 * mu + lmbd) + E_yy * lmbd
    Syy = E_yy * (2 * mu + lmbd) + E_xx * lmbd
    Sxy = 2 * E_xy * mu

    return np.hstack((ux, uy, Sxx, Syy, Sxy))


# ============================================================================
# 软边界条件（Soft Boundary Conditions）
# ============================================================================
# 软边界条件通过DirichletBC显式施加，需要优化器在训练过程中学习满足这些约束
# component参数指定约束的物理量：
#   component=0: ux (x方向位移)
#   component=1: uy (y方向位移)
#   component=2: Sxx (x方向正应力)
#   component=3: Syy (y方向正应力)
#   component=4: Sxy (剪应力)

# 位移边界条件（Displacement Boundary Conditions）
# 上边界：ux = 0
ux_top_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_top, component=0)
# 下边界：ux = 0
ux_bottom_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_bottom, component=0)
# 左边界：uy = 0
uy_left_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_left, component=1)
# 下边界：uy = 0
uy_bottom_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_bottom, component=1)
# 右边界：uy = 0
uy_right_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_right, component=1)

# 应力边界条件（Stress Boundary Conditions）
# 左边界：Sxx = 0（x方向正应力为零，无x方向表面力）
sxx_left_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_left, component=2)
# 右边界：Sxx = 0（x方向正应力为零，无x方向表面力）
sxx_right_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_right, component=2)
# 上边界：Syy = (2*mu + lambda) * Q * sin(πx)  （非零应力边界条件）
# 这是主要的载荷边界条件：在上边界施加y方向的正应力（垂直方向的力）
# 应力沿x方向呈正弦分布，这是导致板变形的主要原因
syy_top_bc = dde.icbc.DirichletBC(
    geom,
    lambda x: (2 * mu + lmbd) * Q * np.sin(np.pi * x[:, 0:1]),
    boundary_top,
    component=3,
)


# ============================================================================
# 硬边界条件（Hard Boundary Conditions）
# ============================================================================
# 硬边界条件通过输出变换函数自动满足，无需优化器学习
# 优点：边界条件自动满足，减少优化难度，提高训练效率
# 缺点：需要构造合适的变换函数，可能不够灵活
def hard_BC(x, f):
    """
    硬边界条件的输出变换函数
    通过构造特定的函数形式，使得边界条件自动满足
    
    Args:
        x: 空间坐标点，形状为 (N, 2)
        f: 神经网络原始输出，形状为 (N, 5)，包含 [ux_raw, uy_raw, Sxx_raw, Syy_raw, Sxy_raw]
    
    Returns:
        变换后的输出，形状为 (N, 5)，包含 [Ux, Uy, Sxx, Syy, Sxy]
        这些输出自动满足边界条件
    """
    # 1. x方向位移：Ux = f[:,0] * y * (1-y)
    #    在 y=0 和 y=1 处，Ux = 0（自动满足上、下边界的ux=0条件）
    Ux = f[:, 0] * x[:, 1] * (1 - x[:, 1])
    
    # 2. y方向位移：Uy = f[:,1] * x * (1-x) * y
    #    在 x=0, x=1 处，Uy = 0（自动满足左、右边界的uy=0条件）
    #    在 y=0 处，Uy = 0（自动满足下边界的uy=0条件）
    Uy = f[:, 1] * x[:, 0] * (1 - x[:, 0]) * x[:, 1]

    # 3. x方向正应力：Sxx = f[:,2] * x * (1-x)
    #    在 x=0 和 x=1 处，Sxx = 0（自动满足左、右边界的Sxx=0条件）
    Sxx = f[:, 2] * x[:, 0] * (1 - x[:, 0])
    
    # 4. y方向正应力：Syy = f[:,3] * (1-y) + (lambda + 2*mu) * Q * sin(πx)
    #    在 y=1 处，Syy = (lambda + 2*mu) * Q * sin(πx)（自动满足上边界的Syy条件）
    #    这是主要的载荷边界条件：在上边界施加y方向的正应力（垂直方向的力）
    Syy = f[:, 3] * (1 - x[:, 1]) + (lmbd + 2 * mu) * Q * sin(np.pi * x[:, 0])
    
    # 5. 剪应力：Sxy = f[:,4]（无边界约束，直接使用原始输出）
    Sxy = f[:, 4]
    
    # 堆叠所有物理量
    return stack((Ux, Uy, Sxx, Syy, Sxy), axis=1)


# ============================================================================
# 体积力函数（Body Force Functions）
# ============================================================================
# 体积力是作用在物体内部每一点的力（如重力、电磁力等）
# 这些函数是根据精确解反推出来的，用于构造一个已知精确解的问题
# 在实际工程问题中，体积力通常是给定的（如重力：fx=0, fy=-ρg）

def fx(x):
    """
    计算x方向的体积力 fx(x,y)
    这个函数是根据精确解反推出来的，使得精确解满足平衡方程
    
    Args:
        x: 空间坐标点，形状为 (N, 2)
    Returns:
        x方向的体积力，形状为 (N, 1)
    """
    return (
        -lmbd
        * (
            4 * np.pi**2 * cos(2 * np.pi * x[:, 0:1]) * sin(np.pi * x[:, 1:2])
            - Q * x[:, 1:2] ** 3 * np.pi * cos(np.pi * x[:, 0:1])
        )
        - mu
        * (
            np.pi**2 * cos(2 * np.pi * x[:, 0:1]) * sin(np.pi * x[:, 1:2])
            - Q * x[:, 1:2] ** 3 * np.pi * cos(np.pi * x[:, 0:1])
        )
        - 8 * mu * np.pi**2 * cos(2 * np.pi * x[:, 0:1]) * sin(np.pi * x[:, 1:2])
    )


def fy(x):
    """
    计算y方向的体积力 fy(x,y)
    这个函数是根据精确解反推出来的，使得精确解满足平衡方程
    
    Args:
        x: 空间坐标点，形状为 (N, 2)
    Returns:
        y方向的体积力，形状为 (N, 1)
    """
    return (
        lmbd
        * (
            3 * Q * x[:, 1:2] ** 2 * sin(np.pi * x[:, 0:1])
            - 2 * np.pi**2 * cos(np.pi * x[:, 1:2]) * sin(2 * np.pi * x[:, 0:1])
        )
        - mu
        * (
            2 * np.pi**2 * cos(np.pi * x[:, 1:2]) * sin(2 * np.pi * x[:, 0:1])
            + (Q * x[:, 1:2] ** 4 * np.pi**2 * sin(np.pi * x[:, 0:1])) / 4
        )
        + 6 * Q * mu * x[:, 1:2] ** 2 * sin(np.pi * x[:, 0:1])
    )


# ============================================================================
# 雅可比矩阵计算函数（Jacobian Computation）
# ============================================================================
def jacobian(f, x, i, j):
    """
    计算雅可比矩阵的某个元素：∂f_i/∂x_j
    这是一个辅助函数，用于处理不同后端的差异
    
    Args:
        f: 函数值，形状为 (N, 5) 或 (N, 5) 的元组（JAX后端）
        x: 空间坐标点，形状为 (N, 2)
        i: 函数分量的索引（0=ux, 1=uy, 2=Sxx, 3=Syy, 4=Sxy）
        j: 坐标分量的索引（0=x, 1=y）
    Returns:
        偏导数，形状为 (N, 1)
    """
    if dde.backend.backend_name == "jax":
        # JAX后端返回元组，需要取第一个元素
        return dde.grad.jacobian(f, x, i=i, j=j)[0]
    else:
        return dde.grad.jacobian(f, x, i=i, j=j)


# ============================================================================
# PDE方程定义（PDE Equation Definition）
# ============================================================================
def pde(x, f):
    """
    定义线性弹性力学的PDE方程残差
    这个函数计算PDE残差，用于构造损失函数
    
    控制方程包括：
    1. 应变-位移关系（几何关系）
    2. 应力-应变关系（本构关系）
    3. 平衡方程（动量守恒）
    
    Args:
        x: 空间坐标点，形状为 (N, 2)
        f: 神经网络输出，形状为 (N, 5)，包含 [ux, uy, Sxx, Syy, Sxy]
    Returns:
        list: 包含5个PDE残差的列表
            [momentum_x, momentum_y, stress_x, stress_y, stress_xy]
    """
    # ========================================================================
    # 1. 计算应变场（Strain Field）
    # ========================================================================
    # x方向正应变：E_xx = ∂ux/∂x
    E_xx = jacobian(f, x, i=0, j=0)
    # y方向正应变：E_yy = ∂uy/∂y
    E_yy = jacobian(f, x, i=1, j=1)
    # 剪应变：E_xy = 0.5*(∂ux/∂y + ∂uy/∂x)
    E_xy = 0.5 * (jacobian(f, x, i=0, j=1) + jacobian(f, x, i=1, j=0))

    # ========================================================================
    # 2. 计算应力场（Stress Field）- 从应变计算
    # ========================================================================
    # 使用本构关系（胡克定律）计算应力
    # Sxx = E_xx*(2*mu + lambda) + E_yy*lambda
    S_xx = E_xx * (2 * mu + lmbd) + E_yy * lmbd
    # Syy = E_yy*(2*mu + lambda) + E_xx*lambda
    S_yy = E_yy * (2 * mu + lmbd) + E_xx * lmbd
    # Sxy = 2*mu*E_xy
    S_xy = E_xy * 2 * mu

    # ========================================================================
    # 3. 计算应力的空间导数（用于平衡方程）
    # ========================================================================
    # ∂Sxx/∂x
    Sxx_x = jacobian(f, x, i=2, j=0)
    # ∂Syy/∂y
    Syy_y = jacobian(f, x, i=3, j=1)
    # ∂Sxy/∂x
    Sxy_x = jacobian(f, x, i=4, j=0)
    # ∂Sxy/∂y
    Sxy_y = jacobian(f, x, i=4, j=1)

    # ========================================================================
    # 4. 平衡方程残差（Momentum Balance Equations）
    # ========================================================================
    # x方向平衡方程：∂Sxx/∂x + ∂Sxy/∂y - fx = 0
    momentum_x = Sxx_x + Sxy_y - fx(x)
    # y方向平衡方程：∂Sxy/∂x + ∂Syy/∂y - fy = 0
    momentum_y = Sxy_x + Syy_y - fy(x)

    # ========================================================================
    # 5. 本构关系残差（Constitutive Relations）
    # ========================================================================
    # 处理JAX后端的特殊情况
    if dde.backend.backend_name == "jax":
        f = f[0]  # f[1] is the function used by jax to compute the gradients

    # 应力一致性条件：从应变计算的应力 = 神经网络输出的应力
    # S_xx（从应变计算）应该等于 f[:,2]（神经网络输出的Sxx）
    stress_x = S_xx - f[:, 2:3]
    # S_yy（从应变计算）应该等于 f[:,3]（神经网络输出的Syy）
    stress_y = S_yy - f[:, 3:4]
    # S_xy（从应变计算）应该等于 f[:,4]（神经网络输出的Sxy）
    stress_xy = S_xy - f[:, 4:5]

    # 返回所有PDE残差
    # 这些残差应该等于0，PINN的目标是让这些残差尽可能小
    return [momentum_x, momentum_y, stress_x, stress_y, stress_xy]


# ============================================================================
# 边界条件选择（Boundary Condition Selection）
# ============================================================================
if BC_type == "hard":
    # 硬边界条件：通过输出变换函数自动满足，无需显式边界条件
    bcs = []
else:
    # 软边界条件：显式列出所有边界条件，让优化器学习满足
    bcs = [
        ux_top_bc,
        ux_bottom_bc,
        uy_left_bc,
        uy_bottom_bc,
        uy_right_bc,
        sxx_left_bc,
        sxx_right_bc,
        syy_top_bc,
    ]

# ============================================================================
# 数据定义（Data Definition）
# ============================================================================
# 创建PDE数据对象，包含：
#   - 几何域
#   - PDE方程
#   - 边界条件
#   - 采样点数量
#   - 精确解（用于验证，可选）
data = dde.data.PDE(
    geom,              # 几何域：单位正方形 [0,1] × [0,1]
    pde,               # PDE方程函数
    bcs,               # 边界条件列表（硬边界时为空列表）
    num_domain=500,    # 域内训练点数量：用于计算PDE残差
    num_boundary=500,  # 边界训练点数量：用于计算边界条件误差
    solution=func,     # 精确解函数（仅用于验证和评估，不是训练所必需的）
    num_test=100,      # 测试点数量：用于评估模型精度
)

# ============================================================================
# 神经网络定义（Neural Network Definition）
# ============================================================================
# PFNN: Physics-Informed Feedforward Neural Network
# 网络结构：[输入层, 隐藏层1, 隐藏层2, 隐藏层3, 隐藏层4, 输出层]
#   - 输入层：2个神经元（x, y坐标）
#   - 隐藏层：每层40个神经元，共4层
#   - 输出层：5个神经元（ux, uy, Sxx, Syy, Sxy）
layers = [2, [40] * 5, [40] * 5, [40] * 5, [40] * 5, 5]
activation = "tanh"           # 激活函数：双曲正切
initializer = "Glorot uniform" # 权重初始化：Glorot均匀分布（Xavier初始化）
net = dde.nn.PFNN(layers, activation, initializer)

# 如果使用硬边界条件，应用输出变换函数
if BC_type == "hard":
    net.apply_output_transform(hard_BC)

# ============================================================================
# 模型编译和训练（Model Compilation and Training）
# ============================================================================
# 创建模型
model = dde.Model(data, net)

# 编译模型
#   - 优化器：Adam
#   - 学习率：0.001
#   - 评估指标：L2相对误差（用于评估模型精度）
model.compile("adam", lr=0.001, metrics=["l2 relative error"])

# 训练模型
#   - iterations: 训练迭代次数
#   - 返回：损失历史记录和训练状态
losshistory, train_state = model.train(iterations=50000)

# ============================================================================
# 结果保存和可视化（Results Saving and Visualization）
# ============================================================================
# 保存目录
save_dir = "/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master/exp/pinn_elasticity_plate"
os.makedirs(save_dir, exist_ok=True)

# 1. 保存损失历史曲线和训练状态（使用DeepXDE默认函数）
dde.saveplot(
    losshistory, train_state, 
    issave=True, 
    isplot=True,
    output_dir=save_dir
)

# 2. 绘制详细的损失历史（三张图：PDE Loss、BC Loss、总Loss，包含训练和测试阶段）
print("\n" + "="*60)
print("正在生成详细的损失历史图（PDE Loss、BC Loss、总Loss）...")
print("="*60)
# 对于弹性板问题：5个PDE残差（momentum_x, momentum_y, stress_x, stress_y, stress_xy）
# + 多个BC损失（取决于边界条件数量）
# 注意：如果使用硬边界条件，BC损失为0（因为边界条件通过输出变换自动满足）
num_pde_losses = 5  # 5个PDE残差：momentum_x, momentum_y, stress_x, stress_y, stress_xy
num_bc_losses = len(bcs)  # BC损失数量（硬边界时为0，软边界时为边界条件数量）

# 检查损失历史的结构，自动推断损失分量数量
if len(losshistory.loss_train) > 0:
    first_loss = losshistory.loss_train[0]
    if isinstance(first_loss, (list, np.ndarray)) and len(first_loss) > 0:
        total_loss_components = len(first_loss)
        print(f"检测到 {total_loss_components} 个损失分量")
        print(f"  - PDE损失: {num_pde_losses}")
        print(f"  - BC损失: {num_bc_losses}")
        if total_loss_components != num_pde_losses + num_bc_losses:
            print(f"  警告: 总分量数 ({total_loss_components}) != PDE ({num_pde_losses}) + BC ({num_bc_losses})")
            # 如果数量不匹配，使用自动推断
            num_bc_losses = max(0, total_loss_components - num_pde_losses)

plot_and_save_loss_history(
    losshistory, save_dir,
    filename="损失历史详细图.png",
    num_pde_losses=num_pde_losses,
    num_bc_losses=num_bc_losses
)

# 3. 提取训练集和测试数据，生成专业的应力场和位移场可视化
print("\n" + "="*60)
print("正在生成专业的应力场和位移场可视化图（训练集+测试集）...")
print("="*60)

# 3.1 获取测试点坐标和预测值
X_test = train_state.X_test
print(f"测试点形状: {X_test.shape}")

# 3.2 使用模型预测测试集
y_pred_test = model.predict(X_test)
print(f"测试集预测值形状: {y_pred_test.shape}")

# 3.3 计算测试集精确解（用于对比）
y_true_test = func(X_test)
print(f"测试集真实值形状: {y_true_test.shape}")

# 3.4 获取训练点坐标和预测值（用于验证阶段可视化）
X_train = train_state.X_train
print(f"训练点形状: {X_train.shape}")

# 3.5 使用模型预测训练集
y_pred_train = model.predict(X_train)
print(f"训练集预测值形状: {y_pred_train.shape}")

# 3.6 计算训练集精确解（用于对比）
y_true_train = func(X_train)
print(f"训练集真实值形状: {y_true_train.shape}")

# 3.7 生成所有场的专业可视化图（包含训练集和测试集）
# 包括：三列对比图（真值、预测值、绝对误差）、相对误差图、切片对比图
field_names = ['ux', 'uy', 'Sxx', 'Syy', 'Sxy']
plot_all_elasticity_fields_with_train(
    X_test, y_true_test, y_pred_test,  # 测试集
    X_train, y_true_train, y_pred_train,  # 训练集（验证阶段）
    save_dir,
    field_names=field_names,
    nx=100, ny=100,  # 插值网格分辨率
    x_min=0.0, x_max=1.0,
    y_min=0.0, y_max=1.0,
    slice_x=0.5  # 在x=0.5处切片（中线）
)

# 3.8 生成变形网格可视化（直观显示板子变形）
print("\n" + "="*60)
print("正在生成变形网格可视化图（直观显示板子变形）...")
print("="*60)

# 导入变形可视化函数
from utils.save_results import plot_deformed_mesh, plot_displacement_vector_field

# 提取测试集的位移场
ux_test = y_pred_test[:, 0]
uy_test = y_pred_test[:, 1]

# 计算合适的放大系数（使变形可见）
max_displacement = max(np.max(np.abs(ux_test)), np.max(np.abs(uy_test)))
scale_factor = min(5.0, 0.3 / max_displacement) if max_displacement > 0 else 1.0
print(f"位移放大系数: {scale_factor:.2f} (最大位移: {max_displacement:.6f})")

# 绘制变形网格对比图
plot_deformed_mesh(
    X_test, ux_test, uy_test, save_dir,
    filename="变形网格对比图_测试集.png",
    scale_factor=scale_factor,
    nx=20, ny=20,
    x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0
)

# 绘制位移矢量场
plot_displacement_vector_field(
    X_test, ux_test, uy_test, save_dir,
    filename="位移矢量场图_测试集.png",
    nx=20, ny=20,
    x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0,
    scale=1.0
)

print("\n" + "="*60)
print("所有可视化图生成完成！")
print(f"结果已保存到: {save_dir}")
print("="*60)
print("\n生成的文件:")
print("  - 损失历史详细图.png: 详细损失历史图（三张子图：PDE Loss、BC Loss、总Loss）")
print("  - 场对比图_测试集_*.png: 测试集场对比图（5个场：ux, uy, Sxx, Syy, Sxy）")
print("  - 场对比图_训练集_*.png: 训练集场对比图（5个场：ux, uy, Sxx, Syy, Sxy）")
print("  - 相对误差图_测试集_*.png: 测试集相对误差图（5个场）")
print("  - 相对误差图_训练集_*.png: 训练集相对误差图（5个场）")
print("  - 切片对比图_测试集_x0.50.png: 测试集切片对比图（沿x=0.5，包含所有5个场）")
print("  - 切片对比图_训练集_x0.50.png: 训练集切片对比图（沿x=0.5，包含所有5个场）")
print("\n总计：1张损失图 + 10张场对比图 + 10张相对误差图 + 2张切片对比图 = 23张图")
