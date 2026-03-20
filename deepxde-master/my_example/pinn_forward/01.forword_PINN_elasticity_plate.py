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
import sys
import os
# Add the project root directory to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import deepxde as dde
import numpy as np
import argparse
import torch
from utils.device_utils import print_gpu_info, set_random_seed
from utils.evaluation_utils import run_evaluation
from utils.save_results import (
    plot_and_save_loss_history,
    plot_all_loss_components,
    plot_all_elasticity_fields_with_train,
    plot_deformed_mesh,
    plot_displacement_vector_field,
    plot_point_distribution
)
from utils.loss_callback import LossHistoryCallback
from utils.checkpoint_utils import BestModelCheckpoint, resolve_model_path
from utils.progress_callback import TqdmProgressCallback
RANDOM_SEED = 42
set_random_seed(RANDOM_SEED)
print(f"Random seed set to: {RANDOM_SEED} (for reproducibility)")
print_gpu_info()
# ============================================================================
# 参数解析（Argument Parsing）
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='PINN Elasticity Plate Problem')
    
    # 运行模式：训练(train)或测试(test)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Running mode: train or test')
    
    # 模型加载路径（仅测试模式使用），若不指定则尝试加载默认路径
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to load model (for test mode). If not specified, tries to load from default save directory.')
    
    # 是否禁用可视化绘图
    parser.add_argument('--no_plot', action='store_true',
                        help='Disable plotting visualization')
    
    # 是否仅计算评估指标（不进行可视化，等同于--no_plot）
    parser.add_argument('--metrics_only', action='store_true',
                        help='Only calculate metrics, no visualization (equivalent to --no_plot)')

    
    # 训练迭代次数
    parser.add_argument('--iterations', type=int, default=50000,
                        help='Number of training iterations')

    # 显示和评估频率
    parser.add_argument('--display_every', type=int, default=1000,
                        help='Display and evaluation frequency')

    # 域内采样点数量
    parser.add_argument('--num_domain', type=int, default=2000,
                        help='Number of domain points')

    # 边界采样点数量
    parser.add_argument('--num_boundary', type=int, default=2000,
                        help='Number of boundary points')

    # 测试点数量
    parser.add_argument('--num_test', type=int, default=5000,
                        help='Number of test points')

    # 学习率
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    
    # 结果保存目录，若不指定则使用默认路径
    parser.add_argument('--save_dir', type=str, default="/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master/exp/01.PINN_forword_elasticity_plate",
                        help='Directory to save results. If None, uses default path.')
    return parser.parse_args()

args = parse_args()

# ============================================================================
# 材料参数（Material Parameters）
# ============================================================================
lmbd = 1.0  # 拉梅常数（Lamé constant）lambda，用于计算体积模量
mu = 0.5    # 剪切模量（Shear modulus），用于计算剪切应力
Q = 4.0     # 问题参数，用于构造精确解和边界条件

# 使用后端无关的函数，确保代码可以在不同后端（PyTorch, JAX, Paddle）上运行
sin = dde.backend.sin
cos = dde.backend.cos
stack = dde.backend.stack  

# ============================================================================
# 几何定义（Geometry Definition）
# ============================================================================
# 定义求解域：单位正方形 [0,1] × [0,1]
geom = dde.geometry.Rectangle([0, 0], [1, 1])

# ============================================================================
# 可视化点分布（Point Distribution Visualization）
# ============================================================================
if not args.no_plot:
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print(f"Visualizing point distribution to {args.save_dir}...")
    plot_point_distribution(
        args.save_dir,
        geom,
        args.num_domain,
        args.num_boundary,
        args.num_test,
        observe_x=None
    )

# ============================================================================
# 边界条件类型选择（Boundary Condition Type）
# ============================================================================
# "hard": 硬边界条件，通过输出变换函数自动满足边界条件（更高效）
# "soft": 软边界条件，通过DirichletBC显式施加边界条件（更灵活）
BC_type = ["soft", "hard"][0]  # 当前选择软边界条件


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
    num_domain=args.num_domain,    # 域内训练点数量：用于计算PDE残差
    num_boundary=args.num_boundary,  # 边界训练点数量：用于计算边界条件误差
    num_test=args.num_test,      # 测试点数量：用于评估模型精度
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
model.compile("adam", lr=args.lr, metrics=[])

# ============================================================================
# 设置保存目录和回调函数
# ============================================================================
# 保存目录
save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)
print(f"Results will be saved to: {save_dir}")
model_dir = os.path.join(save_dir, "model")
os.makedirs(model_dir, exist_ok=True)
for fname in os.listdir(model_dir):
    if fname.startswith("best_model-"):
        os.remove(os.path.join(model_dir, fname))

# 定义损失项名称
pde_loss_names = ['momentum_x', 'momentum_y', 'stress_x', 'stress_y', 'stress_xy']
bc_loss_names = None
if len(bcs) > 0:
    bc_loss_names = [f'BC_{i+1}' for i in range(len(bcs))]

# 创建实时保存loss可视化的callback
loss_callback = LossHistoryCallback(
    save_dir=save_dir,
    period=args.display_every,
    filename="损失历史详细图.png",
    num_pde_losses=5,
    num_bc_losses=len(bcs),
    pde_loss_names=pde_loss_names,
    bc_loss_names=bc_loss_names,
    save_all_components=True
)
best_model_callback = BestModelCheckpoint(os.path.join(model_dir, "best_model"), monitor="test loss", verbose=0)
progress_callback = TqdmProgressCallback(
    total_steps=args.iterations,
    display_every=args.display_every,
    true_solution_fn=func,
    metric_name="metric",
)

# ============================================================================
# 训练/测试流程控制
# ============================================================================
losshistory = None
train_state = None

if args.mode == 'train':
    print("\n" + "="*60)
    print("开始训练模型...")
    print("="*60)
    
    # 训练模型
    losshistory, train_state = model.train(
        iterations=args.iterations,
        display_every=args.display_every,
        callbacks=[loss_callback, best_model_callback, progress_callback],
        verbose=0,
    )
    
    # 保存模型
    model_save_path = os.path.join(model_dir, "model")
    model.save(model_save_path)
    print(f"模型已保存到: {model_save_path}")
    if best_model_callback.best_file:
        model.restore(best_model_callback.best_file, verbose=1)

elif args.mode == 'test':
    print("\n" + "="*60)
    print("开始测试模型...")
    print("="*60)
    
    model_load_path = resolve_model_path(model_dir, save_dir, args.model_path)
            
    print(f"正在加载模型: {model_load_path}")
    try:
        model.restore(model_load_path, verbose=1)
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("请使用 --model_path 指定正确的模型路径")
        sys.exit(1)

# ============================================================================
# 评估与可视化
# ============================================================================
# 准备评估数据
X_test = None
X_train = None

if train_state is not None:
    X_test = train_state.X_test
    X_train = train_state.X_train
else:
    if hasattr(data, "test_x") and data.test_x is not None:
        X_test = data.test_x
    else:
        print("重新生成测试点...")
        X_test = geom.random_points(args.num_test, random="pseudo")
    
    # 只有当需要画图且没有提供训练集时，尝试获取训练集
    if not args.no_plot and not args.metrics_only:
        if hasattr(data, "train_x") and data.train_x is not None:
            X_train = data.train_x

# 运行评估流程
run_evaluation(
    model, 
    X_test, 
    X_train, 
    func, 
    save_dir, 
    losshistory, 
    train_state, 
    bcs,
    plot_visualization=not args.no_plot and not args.metrics_only,
    calculate_metrics=True,
    save_data=True
)
