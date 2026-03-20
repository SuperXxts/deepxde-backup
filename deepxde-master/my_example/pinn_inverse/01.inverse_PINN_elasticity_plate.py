"""
================================================================================
问题描述：2D线性弹性板反演问题（Inverse Linear Elasticity Plate Problem）
================================================================================

【问题背景】
    本代码求解一个2D线性弹性力学反演问题。
    与正演问题（已知材料参数求位移和应力）不同，反演问题的目标是：
    根据观测到的部分位移数据，识别出未知的材料参数（拉梅常数 λ 和 剪切模量 μ）。

【物理问题】
    1. 求解域：单位正方形板 [0,1] × [0,1]
    2. 真实参数（Ground Truth）：
       - lambda (lmbd) = 1.0
       - mu = 0.5
    3. 待反演参数（Unknown Parameters）：
       - lmbd: 初始猜测值为 0.8
       - mu: 初始猜测值为 0.4
       - 目标：通过训练，使这两个参数收敛到真实值
    4. 观测数据（Observation Data）：
       - 在域内随机选取 num_observe 个点
       - 观测这些点的 x方向位移 (ux) 和 y方向位移 (uy)
       - 这些观测值作为训练数据，约束神经网络的输出

【求解方法】
    使用PINN（Physics-Informed Neural Network）方法：
    1. 神经网络结构：PFNN（Physics-Informed Feedforward Neural Network）
       - 输入：空间坐标 (x, y)
       - 输出：5个物理量 [ux, uy, Sxx, Syy, Sxy]
    2. 可学习变量（Trainable Variables）：
       - 神经网络的权重和偏置
       - 物理参数 lmbd 和 mu（定义为 dde.Variable）
    3. 损失函数（Loss Function）：
       - PDE残差：动量守恒方程 + 本构关系（包含未知的 lmbd 和 mu）
       - 观测数据误差：模型预测位移与观测位移的差异
    4. 边界条件：
       - 使用硬边界条件（Hard BC）自动满足边界约束，减少优化难度

【输出内容】
    1. 训练过程中的 Loss 曲线（PDE Loss 和 Data Loss）
    2. 反演参数（lmbd, mu）随迭代次数的变化曲线
    3. 最优模型权重
    4. 最终预测的位移场和应力场与真实解的对比
    5. 反演指标（JSON）和损失历史（JSON）

================================================================================
"""
import sys
import os
import json

# 将项目根目录添加到python路径，以便导入utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import numpy as np
import deepxde as dde
import torch
import matplotlib.pyplot as plt

from utils.device_utils import print_gpu_info, set_random_seed
from utils.checkpoint_utils import BestModelCheckpoint, resolve_model_path
from utils.loss_callback import LossHistoryCallback, ParameterPlottingCallback
from utils.progress_callback import TqdmProgressCallback
from utils.save_results import (
    plot_and_save_loss_history,
    plot_all_loss_components,
    plot_all_elasticity_fields_with_train,
    plot_parameter_history,
    plot_point_distribution,
    save_loss_history_json,
    save_loss_history,
    plot_deformed_mesh
)
from utils.data_saving_utils import _get_save_path
from utils.true_solution_utils import calculate_accuracy_metrics
from utils.evaluation_utils import run_evaluation

# ============================================================================
# 参数解析（Argument Parsing）
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="PINN inverse: elasticity plate (infer lambda, mu)")
    
    # 学习率
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    
    # 训练迭代次数
    parser.add_argument("--iterations", type=int, default=50000, help="Number of training iterations")
    
    # 显示和记录频率
    parser.add_argument("--display_every", type=int, default=1000, help="Display and recording frequency")
    
    # 域内配点数量（用于计算PDE残差）
    parser.add_argument("--num_domain", type=int, default=2000, help="Number of domain collocation points")
    
    # 边界配点数量
    parser.add_argument("--num_boundary", type=int, default=2000, help="Number of boundary points")
    
    # 测试点数量（用于评估模型精度）
    parser.add_argument("--num_test", type=int, default=5000, help="Number of test points")
    
    # 观测点数量（用于反演的训练数据）
    parser.add_argument("--num_observe", type=int, default=500, help="Number of observation points for inverse problem")
    
    # 结果保存目录
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master/exp/01.PINN_inverse_elasticity_plate_2.0_1.0",
        help="Directory to save results"
    )
    
    # 运行模式：训练(train)或测试(test)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Running mode: train or test')
    
    # 模型加载路径（仅测试模式使用），若不指定则尝试加载默认路径
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to load model (for test mode). If not specified, tries to load from default save directory.')
                        
    return parser.parse_args()


args = parse_args()
# 设置随机种子以保证结果可复现
set_random_seed(42)
print_gpu_info()

# 使用后端无关的数学函数
sin = dde.backend.sin
cos = dde.backend.cos
stack = dde.backend.stack

# ============================================================================
# 真实参数与待反演参数（Parameters）
# ============================================================================
# 真实值（Ground Truth）- 用于生成观测数据和验证反演结果
mu_true = 0.5
lmbd_true = 1.0
Q = 4.0

# 待反演参数（Trainable Variables）- 初始猜测值
# 我们希望通过训练，这两个变量能收敛到 mu_true 和 lmbd_true
lmbd = dde.Variable(2.0)  # 初始猜测 lambda = 0.8 (真实值 1.0)
mu = dde.Variable(1.0)    # 初始猜测 mu = 0.4 (真实值 0.5)

# ============================================================================
# 几何定义（Geometry）
# ============================================================================
# 定义求解域：单位正方形 [0,1] × [0,1]
geom = dde.geometry.Rectangle([0, 0], [1, 1])

BC_type = ["soft", "hard"][0]

def boundary_left(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0.0)

def boundary_right(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1.0)

def boundary_top(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 1.0)

def boundary_bottom(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 0.0)


# ============================================================================
# 精确解函数（Exact Solution）- 用于生成观测数据
# ============================================================================
def func(x):
    """
    计算精确解：位移场和应力场
    用于生成观测数据，以及最后验证模型预测的精度
    """
    # 计算位移场
    ux = np.cos(2 * np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2])
    uy = np.sin(np.pi * x[:, 0:1]) * Q * x[:, 1:2] ** 4 / 4

    # 计算应变场
    E_xx = -2 * np.pi * np.sin(2 * np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2])
    E_yy = np.sin(np.pi * x[:, 0:1]) * Q * x[:, 1:2] ** 3
    E_xy = 0.5 * (
        np.pi * np.cos(2 * np.pi * x[:, 0:1]) * np.cos(np.pi * x[:, 1:2])
        + np.pi * np.cos(np.pi * x[:, 0:1]) * Q * x[:, 1:2] ** 4 / 4
    )

    # 计算应力场（使用真实参数）
    Sxx = E_xx * (2 * mu_true + lmbd_true) + E_yy * lmbd_true
    Syy = E_yy * (2 * mu_true + lmbd_true) + E_xx * lmbd_true
    Sxy = 2 * E_xy * mu_true

    return np.hstack((ux, uy, Sxx, Syy, Sxy))


# ============================================================================
# 硬边界条件（Hard Boundary Conditions）
# ============================================================================
def hard_BC(x, f):
    """
    硬边界条件的输出变换函数
    强制模型输出满足特定的边界约束，减少优化难度
    
    硬边界条件构造逻辑：
    1. Ux: 在 y=0 和 y=1 处，x[:, 1] * (1 - x[:, 1]) 为 0，强制 Ux=0
    2. Uy: 在 x=0, x=1, y=0 处，x[:, 0] * (1 - x[:, 0]) * x[:, 1] 为 0，强制 Uy=0
    3. Sxx: 在 x=0, x=1 处，x[:, 0] * (1 - x[:, 0]) 为 0，强制 Sxx=0
    4. Syy: 在 y=1 处，(1 - x[:, 1]) 为 0，加上后面的项，满足特定边界应力
    5. Sxy: 无特殊硬约束，由网络自由学习
    """
    Ux = f[:, 0] * x[:, 1] * (1 - x[:, 1])
    Uy = f[:, 1] * x[:, 0] * (1 - x[:, 0]) * x[:, 1]
    Sxx = f[:, 2] * x[:, 0] * (1 - x[:, 0])
    Syy = f[:, 3] * (1 - x[:, 1]) + (lmbd_true + 2 * mu_true) * Q * sin(np.pi * x[:, 0])
    Sxy = f[:, 4]
    return stack((Ux, Uy, Sxx, Syy, Sxy), axis=1)

ux_top_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_top, component=0)
ux_bottom_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_bottom, component=0)
uy_left_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_left, component=1)
uy_bottom_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_bottom, component=1)
uy_right_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_right, component=1)
sxx_left_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_left, component=2)
sxx_right_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_right, component=2)
syy_top_bc = dde.icbc.DirichletBC(
    geom,
    lambda x: (2 * mu_true + lmbd_true) * Q * np.sin(np.pi * x[:, 0:1]),
    boundary_top,
    component=3,
)


# ============================================================================
# PDE 右端项（Body Forces）
# ============================================================================
def fx(x):
    """x方向的体积力，根据真实参数计算"""
    return (
        -lmbd_true
        * (
            4 * np.pi**2 * cos(2 * np.pi * x[:, 0:1]) * sin(np.pi * x[:, 1:2])
            - Q * x[:, 1:2] ** 3 * np.pi * cos(np.pi * x[:, 0:1])
        )
        - mu_true
        * (
            np.pi**2 * cos(2 * np.pi * x[:, 0:1]) * sin(np.pi * x[:, 1:2])
            - Q * x[:, 1:2] ** 3 * np.pi * cos(np.pi * x[:, 0:1])
        )
        - 8 * mu_true * np.pi**2 * cos(2 * np.pi * x[:, 0:1]) * sin(np.pi * x[:, 1:2])
    )


def fy(x):
    """y方向的体积力，根据真实参数计算"""
    return (
        lmbd_true
        * (
            3 * Q * x[:, 1:2] ** 2 * sin(np.pi * x[:, 0:1])
            - 2 * np.pi**2 * cos(np.pi * x[:, 1:2]) * sin(2 * np.pi * x[:, 0:1])
        )
        - mu_true
        * (
            2 * np.pi**2 * cos(np.pi * x[:, 1:2]) * sin(2 * np.pi * x[:, 0:1])
            + (Q * x[:, 1:2] ** 4 * np.pi**2 * sin(np.pi * x[:, 0:1])) / 4
        )
        + 6 * Q * mu_true * x[:, 1:2] ** 2 * sin(np.pi * x[:, 0:1])
    )


# ============================================================================
# PDE 定义（Physics Equation）
# ============================================================================
def pde(x, f):
    """
    定义控制方程（线弹性力学方程）
    注意：这里使用的是待反演的变量 lmbd 和 mu，而不是真实值
    """
    # f = [ux, uy, Sxx, Syy, Sxy]，其中位移和应力都由网络输出
    # 通过自动微分得到应变，再用本构关系计算应力，形成应力一致性约束
    # 计算应变（自动微分）
    E_xx = dde.grad.jacobian(f, x, i=0, j=0)
    E_yy = dde.grad.jacobian(f, x, i=1, j=1)
    E_xy = 0.5 * (dde.grad.jacobian(f, x, i=0, j=1) + dde.grad.jacobian(f, x, i=1, j=0))

    # 本构关系：应力 = f(应变, lmbd, mu)
    # 这里的 lmbd 和 mu 是可学习的变量
    S_xx = E_xx * (2 * mu + lmbd) + E_yy * lmbd
    S_yy = E_yy * (2 * mu + lmbd) + E_xx * lmbd
    S_xy = E_xy * 2 * mu

    # 计算应力的导数（用于平衡方程）
    Sxx_x = dde.grad.jacobian(f, x, i=2, j=0)
    Syy_y = dde.grad.jacobian(f, x, i=3, j=1)
    Sxy_x = dde.grad.jacobian(f, x, i=4, j=0)
    Sxy_y = dde.grad.jacobian(f, x, i=4, j=1)

    # 平衡方程（动量守恒），考虑体力项 fx, fy
    momentum_x = Sxx_x + Sxy_y - fx(x)
    momentum_y = Sxy_x + Syy_y - fy(x)

    # 兼容性处理（JAX后端）
    if dde.backend.backend_name == "jax":
        f = f[0]

    # 应力约束：网络输出的应力应与本构关系计算的应力一致
    stress_x = S_xx - f[:, 2:3]
    stress_y = S_yy - f[:, 3:4]
    stress_xy = S_xy - f[:, 4:5]
    
    # 返回所有PDE残差
    return [momentum_x, momentum_y, stress_x, stress_y, stress_xy]


# ============================================================================
# 观测数据准备（Observation Data）
# ============================================================================
# 随机生成观测点
rs = np.random.RandomState(42)
eps = 1e-6
observe_x = eps + (1.0 - 2.0 * eps) * rs.rand(args.num_observe, 2).astype(np.float32)
# 计算这些点的真实位移（模拟实验观测值），仅使用位移分量作为观测数据
observe_y = func(observe_x)

# 将观测数据作为点集边界条件（PointSetBC）加入训练
# component=0 对应 ux，component=1 对应 uy
observe_ux = dde.icbc.PointSetBC(observe_x, observe_y[:, 0:1], component=0)
observe_uy = dde.icbc.PointSetBC(observe_x, observe_y[:, 1:2], component=1)

if BC_type == "hard":
    bcs = [observe_ux, observe_uy]
    bc_loss_names = ["obs_ux", "obs_uy"]
else:
    bcs = [
        ux_top_bc,
        ux_bottom_bc,
        uy_left_bc,
        uy_bottom_bc,
        uy_right_bc,
        sxx_left_bc,
        sxx_right_bc,
        syy_top_bc,
        observe_ux,
        observe_uy,
    ]
    bc_loss_names = ["ux_top", "ux_bottom", "uy_left", "uy_bottom", "uy_right", "sxx_left", "sxx_right", "syy_top", "obs_ux", "obs_uy"]

# ============================================================================
# 可视化点分布（Point Distribution Visualization）
# ============================================================================
# 确保保存目录存在
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
    
print(f"Visualizing point distribution to {args.save_dir}...")
plot_point_distribution(
    args.save_dir,
    geom,
    args.num_domain,
    args.num_boundary,
    args.num_test,
    observe_x=observe_x,
    filename="point_distribution.png"
)

# ============================================================================
# 数据集构建（Data）
# ============================================================================
data = dde.data.PDE(
    geom,
    pde,
    bcs,
    num_domain=args.num_domain,
    num_boundary=args.num_boundary,
    anchors=observe_x,  # 将观测点作为锚点，确保训练采样包含观测点
    solution=func,      # 提供参考解（用于测试误差计算）
    num_test=args.num_test,
)

# ============================================================================
# 模型构建（Model）
# ============================================================================
# 网络结构：2输入 -> 多层全连接 -> 5输出（位移+应力）
layers = [2, [40] * 5, [40] * 5, [40] * 5, [40] * 5, 5]
net = dde.nn.PFNN(layers, "tanh", "Glorot uniform")
if BC_type == "hard":
    net.apply_output_transform(hard_BC)

model = dde.Model(data, net)

# 编译模型
# external_trainable_variables=[lmbd, mu] 告诉优化器这两个变量也需要更新
model.compile("adam", lr=args.lr, metrics=[], external_trainable_variables=[lmbd, mu])

# ============================================================================
# 回调函数与训练（Callbacks & Training）
# ============================================================================
save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)
model_dir = os.path.join(save_dir, "model")
os.makedirs(model_dir, exist_ok=True)

# 1. 损失历史记录（Loss History）
loss_callback = LossHistoryCallback(
    save_dir=save_dir,
    period=args.display_every,
    filename="损失历史详细图.png",
    num_pde_losses=5,
    num_bc_losses=len(bcs),
    pde_loss_names=["momentum_x", "momentum_y", "stress_x", "stress_y", "stress_xy"],
    bc_loss_names=bc_loss_names,
    bc_label="BC Loss",
    save_all_components=True,
)

# 2. 最优模型保存（Best Model Checkpoint）
best_model_callback = BestModelCheckpoint(
    os.path.join(model_dir, "best_model"),
    monitor="test loss",
    verbose=0,
)

# 3. 变量值记录（Variable Value）
# 记录 lmbd 和 mu 的变化历史
variable_callback = dde.callbacks.VariableValue(
    [lmbd, mu],
    period=args.display_every,
    filename=os.path.join(save_dir, "inferred_params.txt"),
    precision=8,
)

# 3.5. 参数变化实时绘图
param_plot_callback = ParameterPlottingCallback(
    param_file=os.path.join(save_dir, "inferred_params.txt"),
    save_path=_get_save_path(save_dir, "png", "parameter_convergence.png"),
    true_values=[lmbd_true, mu_true],
    param_names=["lambda", "mu"],
    period=args.display_every
)

# 4. 进度条（Progress Bar）
progress_callback = TqdmProgressCallback(
    total_steps=args.iterations,
    display_every=args.display_every,
    true_solution_fn=func,
    metric_name="metric",
)

print(f"\nStart training for {args.iterations} iterations...")
print(f"Results will be saved to: {save_dir}")
lmbd_val = lmbd.item() if hasattr(lmbd, "item") else lmbd
mu_val = mu.item() if hasattr(mu, "item") else mu
print(f"Initial parameters: lmbd={lmbd_val:.4f}, mu={mu_val:.4f}")
print(f"True parameters:    lmbd={lmbd_true:.4f}, mu={mu_true:.4f}")

# ============================================================================
# 训练/测试流程控制（Training/Testing Logic）
# ============================================================================
losshistory = None
train_state = None
if args.mode == "train":
    print("\n" + "="*60)
    print("Start Training...")
    print("="*60)
    
    # 开始训练
    losshistory, train_state = model.train(
        iterations=args.iterations,
        callbacks=[loss_callback, best_model_callback, variable_callback, param_plot_callback, progress_callback],
        display_every=args.display_every,
    )
    
    # 训练结束后进行后处理
    print("\nTraining finished. Post-processing results...")
    
    # 1. 保存DeepXDE默认的 loss 图和数据
    dde_dir = os.path.join(save_dir, "dde")
    os.makedirs(dde_dir, exist_ok=True)
    dde.saveplot(losshistory, train_state, issave=True, isplot=False, output_dir=dde_dir)
    
    save_loss_history(losshistory, save_dir, filename="loss_history.dat")
    save_loss_history_json(losshistory, save_dir, filename="loss_history.json")

    # 3. 绘制参数反演收敛曲线（重要！）
    # 读取 inferred_params.txt 并绘图
    param_file = os.path.join(save_dir, "inferred_params.txt")
    plot_parameter_history(
        param_file, 
        true_values=[lmbd_true, mu_true], 
        param_names=["lambda", "mu"], 
        save_path=_get_save_path(save_dir, "png", "parameter_convergence.png")
    )
    
    # 4. 绘制详细的 Loss 曲线（包含所有分量）
    plot_all_loss_components(
        losshistory, 
        save_dir, 
        num_pde_losses=5, 
        num_bc_losses=len(bcs),
        pde_loss_names=["momentum_x", "momentum_y", "stress_x", "stress_y", "stress_xy"],
        bc_loss_names=bc_loss_names
    )
    
    # 加载最优模型用于最终评估
    print("Loading best model for evaluation...")
    best_model_path = os.path.join(model_dir, "best_model-" + str(best_model_callback.best_step) + ".pt")
    model.restore(best_model_path, verbose=1)
    
    # 5. 计算并保存反演精度指标（基于最优模型）
    lmbd_pred = lmbd.item() if hasattr(lmbd, "item") else lmbd
    mu_pred = mu.item() if hasattr(mu, "item") else mu
    
    inverse_metrics = {
        "lambda": {
            "true": float(lmbd_true),
            "pred": float(lmbd_pred),
            "abs_error": float(abs(lmbd_true - lmbd_pred)),
            "rel_error": float(abs(lmbd_true - lmbd_pred) / lmbd_true)
        },
        "mu": {
            "true": float(mu_true),
            "pred": float(mu_pred),
            "abs_error": float(abs(mu_true - mu_pred)),
            "rel_error": float(abs(mu_true - mu_pred) / mu_true)
        }
    }
    
    # 保存反演指标到JSON
    metrics_path = _get_save_path(save_dir, "json", "inverse_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(inverse_metrics, f, indent=4)
    print(f"Inverse metrics saved to {metrics_path}")

    # 6. 计算物理场精度指标
    print("\nCalculating field accuracy metrics...")
    x_test = geom.random_points(args.num_test)
    y_true_test = func(x_test)
    y_pred_test_best = model.predict(x_test)
    calculate_accuracy_metrics(y_true_test, y_pred_test_best, save_dir=save_dir)

elif args.mode == "test":
    print("\n" + "="*60)
    print("Start Testing...")
    print("="*60)
    
    model_load_path = resolve_model_path(model_dir, save_dir, args.model_path)
            
    print(f"Loading model from: {model_load_path}")
    try:
        model.restore(model_load_path, verbose=1)
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Please specify correct model path using --model_path")
        sys.exit(1)

# ============================================================================
# 最终评估与可视化（Final Evaluation & Visualization）
# ============================================================================
X_test = None
X_train = None

if train_state is not None:
    X_test = train_state.X_test
    X_train = observe_x
else:
    if hasattr(data, "test_x") and data.test_x is not None:
        X_test = data.test_x
    else:
        X_test = geom.random_points(args.num_test)
    X_train = observe_x

run_evaluation(
    model,
    X_test,
    X_train,
    func,
    save_dir,
    losshistory,
    train_state,
    bcs,
    bc_loss_names=bc_loss_names,
    bc_label="BC Loss",
    data_loss_prefix="obs_",
    plot_visualization=True,
    calculate_metrics=True,
    save_data=True
)

print(f"\nAll results saved to {save_dir}")
