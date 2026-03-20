# DeepONet 运行指南

## 什么是 DeepONet？

DeepONet（Deep Operator Network）是一种深度学习架构，用于学习从函数到函数的映射（算子学习）。它是 DeepXDE 库中实现的一个重要算法，可以用于：

- 学习偏微分方程（PDE）的解算子
- 学习各种函数到函数的映射关系
- 物理信息算子学习（Physics-Informed DeepONet）

## 环境要求

DeepXDE 需要以下依赖之一作为后端：
- TensorFlow 1.x (>=2.7.0)
- TensorFlow 2.x (>=2.3.0)
- PyTorch (>=2.0.0)
- JAX
- PaddlePaddle (>=2.6.0)

其他依赖：
- numpy
- matplotlib
- scipy
- scikit-learn
- scikit-optimize

## 如何运行 DeepONet 示例

### 方法1：设置 PYTHONPATH 后运行（推荐）

由于 deepxde 是本地源码，需要将其添加到 Python 路径中：

```bash
cd /public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master
export PYTHONPATH=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master:$PYTHONPATH
python examples/operator/poisson_1d_pideeponet.py
```

或者一行命令：

```bash
cd /public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master && PYTHONPATH=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master:$PYTHONPATH python examples/operator/poisson_1d_pideeponet.py
```

### 方法2：在 Python 脚本中设置路径

创建一个运行脚本 `run_deeponet.py`：

```python
import sys
import os
# 添加 deepxde 目录到 Python 路径
sys.path.insert(0, '/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master')

import deepxde as dde
# 然后运行你的代码或导入示例
exec(open('examples/operator/poisson_1d_pideeponet.py').read())
```

### 方法3：安装 deepxde（如果已安装 pip）

如果你想全局安装：

```bash
cd /public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master
pip install -e .
```

然后就可以直接运行示例了。

## 可用的 DeepONet 示例

在 `examples/operator/` 目录下有以下 DeepONet 示例：

### 1. 简单示例（推荐初学者）
- **poisson_1d_pideeponet.py** - 一维泊松方程的物理信息 DeepONet
  ```bash
  python examples/operator/poisson_1d_pideeponet.py
  ```
  这个示例不需要额外的数据文件，可以直接运行。

### 2. 其他示例
- `advection_aligned_pideeponet.py` - 平流方程（对齐）
- `advection_unaligned_pideeponet.py` - 平流方程（非对齐）
- `antiderivative_aligned_pideeponet.py` - 反导数（对齐）
- `antiderivative_unaligned_pideeponet.py` - 反导数（非对齐）
- `stokes_aligned_pideeponet.py` - Stokes 方程
- `diff_rec_aligned_pideeponet.py` - 扩散-反应方程

**注意**：某些示例可能需要额外的数据文件（.npz格式），请确保数据文件在运行目录中。

## 示例代码结构

典型的 DeepONet 代码包含以下部分：

1. **定义 PDE 和边界条件**
   ```python
   def equation(x, y, f):
       dy_xx = dde.grad.hessian(y, x)
       return -dy_xx - f
   
   geom = dde.geometry.Interval(0, 1)
   bc = dde.icbc.DirichletBC(geom, u_boundary, boundary)
   pde = dde.data.PDE(geom, equation, bc, num_domain=100, num_boundary=2)
   ```

2. **定义函数空间和评估点**
   ```python
   space = dde.data.PowerSeries(N=degree + 1)
   evaluation_points = geom.uniform_points(num_eval_points, boundary=True)
   ```

3. **定义 PDE 算子**
   ```python
   pde_op = dde.data.PDEOperatorCartesianProd(
       pde, space, evaluation_points, num_function=100
   )
   ```

4. **创建 DeepONet 网络**
   ```python
   net = dde.nn.DeepONetCartesianProd(
       [num_eval_points, 32, p],  # branch net
       [dim_x, 32, p],             # trunk net
       activation="tanh",
       kernel_initializer="Glorot normal",
   )
   ```

5. **训练模型**
   ```python
   model = dde.Model(pde_op, net)
   model.compile("L-BFGS")
   model.train()
   ```

6. **预测和可视化**
   ```python
   y = model.predict((fx, x))
   plt.plot(x, y)
   ```

## 快速开始示例

运行最简单的示例：

```bash
cd /public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master
export PYTHONPATH=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master:$PYTHONPATH
python examples/operator/poisson_1d_pideeponet.py
```

这个示例会：
1. 学习一维泊松方程的解算子
2. 训练一个物理信息 DeepONet
3. 显示源项 f(x) 和对应的解 u(x) 的可视化结果

## 常见问题

1. **后端选择**：DeepXDE 会自动检测可用的后端。你可以通过设置环境变量来指定：
   ```bash
   export DDE_BACKEND=pytorch  # 或 tensorflow, jax, paddle
   ```

2. **版本问题**：确保你的后端库版本满足要求（如 PyTorch >= 2.0.0）

3. **数据文件缺失**：某些示例需要数据文件，请检查 `examples/operator/` 目录

## 更多资源

- DeepXDE 官方文档：https://deepxde.readthedocs.io
- 算子学习示例：https://deepxde.readthedocs.io/en/latest/demos/operator.html
- GitHub 仓库：https://github.com/lululxvi/deepxde


## 运行命令:
cd /public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master/examples/operator
export PYTHONPATH=/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master:$PYTHONPATH
python diff_rec_aligned_pideeponet.py



训练、测试、验证、评估的区别
1. 训练阶段（Training）
数据：训练集（1000个函数）
方式：无监督学习
真实标签：无（y_train=None）
用途：通过PDE残差学习算子映射
2. 测试阶段（Test，训练过程中）
数据：测试集（100个函数）
时机：训练过程中，每1000次迭代调用一次
真实标签：无（y_test=None）
用途：计算Test loss（PDE残差、BC误差），监控训练，不更新权重
3. 验证（Validation）
DeepXDE中没有单独的验证集
Test set同时用于验证（监控过拟合）
所以Test loss既用于测试，也用于验证
4. 最终评估（Evaluation，训练完成后）
时机：训练完成后，独立评估步骤
数据：新的输入函数（随机生成1个）
真实解来源：使用数值求解器（solve_ADR）计算
用途：验证模型是否学到了正确的算子映射