"""
传统DeepONet示例：学习反导数算子（有监督学习）

这个脚本演示了如何使用传统DeepONet学习反导数算子 G: v -> u，
其中 du/dx = v(x)，u(0) = 0。

⚠️ 重要说明：
- 这是传统DeepONet（有监督学习），需要真实的G(u)作为标签
- 训练数据文件需要从DeepXDE官方下载
- 数据文件下载地址：https://yaleedu-my.sharepoint.com/:f:/g/personal/lu_lu_yale_edu/EnTn0aLimaRJuNKDOc0lfHkB2MXK8n8vAO1oV5cWVdJo3w?e=OLp80r

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

# ============================================================================
# 0. 设置随机种子（用于结果可复现）
# ============================================================================
RANDOM_SEED = 42
set_random_seed(RANDOM_SEED)
print(f"Random seed set to: {RANDOM_SEED} (for reproducibility)")
print()

# ============================================================================
# 1. 加载数据集
# ============================================================================
# 数据文件路径
# 数据文件已下载到：/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master/data/deeponet_antiderivative_aligned/
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/deeponet_antiderivative_aligned'))
train_file = os.path.join(data_dir, "antiderivative_aligned_train.npz")
test_file = os.path.join(data_dir, "antiderivative_aligned_test.npz")

# 检查数据文件是否存在
if not os.path.exists(train_file) or not os.path.exists(test_file):
    print("=" * 60)
    print("ERROR: Data files not found!")
    print("=" * 60)
    print(f"Missing files:")
    if not os.path.exists(train_file):
        print(f"  - {train_file}")
    if not os.path.exists(test_file):
        print(f"  - {test_file}")
    print()
    print("Please download the data files from:")
    print("https://yaleedu-my.sharepoint.com/:f:/g/personal/lu_lu_yale_edu/EnTn0aLimaRJuNKDOc0lfHkB2MXK8n8vAO1oV5cWVdJo3w?e=OLp80r")
    print()
    print("And place them in the following directory:")
    print(f"  {data_dir}")
    print("=" * 60)
    sys.exit(1)

# Load training dataset
print("Loading training dataset...")
d = np.load(train_file, allow_pickle=True)
X_train = (d["X"][0].astype(np.float32), d["X"][1].astype(np.float32))
y_train = d["y"].astype(np.float32)
print(f"  Training data shape: X_train[0]={X_train[0].shape}, X_train[1]={X_train[1].shape}, y_train={y_train.shape}")

# Load testing dataset
print("Loading testing dataset...")
d = np.load(test_file, allow_pickle=True)
X_test = (d["X"][0].astype(np.float32), d["X"][1].astype(np.float32))
y_test = d["y"].astype(np.float32)
print(f"  Testing data shape: X_test[0]={X_test[0].shape}, X_test[1]={X_test[1].shape}, y_test={y_test.shape}")
print()

# ============================================================================
# 2. 创建数据对象
# ============================================================================
data = dde.data.TripleCartesianProd(
    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
)

# ============================================================================
# 3. 定义网络结构
# ============================================================================
m = 100  # Branch net输入维度（函数在100个评估点的值）
dim_x = 1  # Trunk net输入维度（空间坐标x）

net = dde.nn.DeepONetCartesianProd(
    [m, 40, 40],      # Branch Net: 100 -> 40 -> 40
    [dim_x, 40, 40],  # Trunk Net: 1 -> 40 -> 40
    "relu",
    "Glorot normal",
)

# ============================================================================
# 4. 创建模型并编译
# ============================================================================
model = dde.Model(data, net)

# 检查GPU使用情况
print_gpu_info()

# 编译模型
# metrics=["mean l2 relative error"]: 使用L2相对误差作为评估指标
model.compile("adam", lr=0.001, metrics=["mean l2 relative error"])

# ============================================================================
# 5. 准备模型配置（用于保存）
# ============================================================================
config = {
    "model_type": "DeepONetCartesianProd",
    "problem": "Antiderivative operator",
    "branch_net": [m, 40, 40],
    "trunk_net": [dim_x, 40, 40],
    "activation": "relu",
    "kernel_initializer": "Glorot normal",
    "optimizer": "adam",
    "learning_rate": 0.001,
    "metrics": ["mean l2 relative error"],
    "iterations": 10000,
    "train_size": y_train.shape[0],
    "test_size": y_test.shape[0],
    "random_seed": RANDOM_SEED,
}

# ============================================================================
# 6. 训练模型
# ============================================================================
# 设置保存目录
save_dir = "/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master/exp/antiderivative_aligned"

# 创建实时保存loss可视化的callback
loss_callback = LossHistoryCallback(
    save_dir=save_dir,
    period=1000,  # 每1000次迭代保存一次
    filename="loss_history.png"
)

print("Starting training...")
print(f"Results will be saved to: {save_dir}")
print()

losshistory, train_state = model.train(iterations=10000, callbacks=[loss_callback])

# ============================================================================
# 7. 预测和评估
# ============================================================================
print()
print("=" * 60)
print("Prediction and Evaluation")
print("=" * 60)

# 从测试集中选择一个样本进行预测和可视化
# 选择第一个测试样本
test_idx = 0
v_branch_test = X_test[0][test_idx:test_idx+1]  # 形状: (1, 100) - 1个函数在100个点的值
x_trunk_test = X_test[1]  # 形状: (N, 1) - N个空间位置
y_true_test = y_test[test_idx:test_idx+1]  # 形状: (1, N) - 真实解

# 使用模型进行预测
print(f"Predicting for test sample {test_idx}...")
y_pred_test = model.predict((v_branch_test, x_trunk_test))  # 形状: (1, N)

# 重塑为1D数组，便于可视化
y_true_1d = y_true_test.flatten()  # 形状: (N,)
y_pred_1d = y_pred_test.flatten()  # 形状: (N,)
x_1d = x_trunk_test.flatten()  # 形状: (N,)

# 计算误差
l2_error = np.linalg.norm(y_true_1d - y_pred_1d) / np.linalg.norm(y_true_1d)
max_error = np.max(np.abs(y_true_1d - y_pred_1d))
mean_error = np.mean(np.abs(y_true_1d - y_pred_1d))

print(f"  L2 relative error: {l2_error:.6f}")
print(f"  Max absolute error: {max_error:.6f}")
print(f"  Mean absolute error: {mean_error:.6f}")
print()

# ============================================================================
# 8. 保存所有结果
# ============================================================================
# 注意：对于传统DeepONet，我们使用测试数据中的真实解
# 不需要像PIDeepONet那样用数值方法求解真实解

# 为了兼容save_all_results函数，我们需要将数据转换为合适的格式
# save_all_results期望u_true和u_pred是2D数组（用于热力图）
# 但这里我们只有1D数据，所以创建一个简单的2D表示

# 将1D数据转换为2D（用于可视化）
# 假设我们想显示函数曲线，可以创建一个简单的2D表示
# 或者直接保存1D数据

# ============================================================================
# 8. 保存所有结果
# ============================================================================
# 使用工具函数保存所有结果，包括：
#   - 模型配置（model_config.json）
#   - Loss历史数据（loss_history.dat）和可视化（loss_history.png）
#   - 模型检查点（model_checkpoint-*.pt）
#   - 预测结果数据（u_true_1d.npy, u_pred_1d.npy, x_1d.npy等）
#   - 1D可视化图像（comparison_1d.png：函数曲线对比和误差分布）
#   - 误差信息（error_info.json）
# 
# 注意：save_all_results会自动检测数据维度（1D或2D），并相应地处理
# 对于1D问题（t=None），会自动创建函数曲线对比图
save_all_results(
    model=model,           # 训练好的模型
    losshistory=losshistory,  # 损失历史
    train_state=train_state,  # 训练状态
    u_true=y_true_1d,      # 真实解（1D数组）
    u_pred=y_pred_1d,      # 预测解（1D数组）
    config=config,        # 模型配置
    x_trunk=x_trunk_test, # 空间点坐标
    v_branch=v_branch_test,  # 输入函数值
    save_dir=save_dir,    # 保存目录
    x=x_1d,              # 空间坐标数组（用于可视化）
    t=None               # 时间坐标（None表示1D问题）
)
