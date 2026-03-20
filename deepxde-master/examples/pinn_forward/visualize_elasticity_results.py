"""
弹性板问题的专业可视化脚本

这个脚本用于：
1. 读取训练结果数据（.dat文件）
2. 生成专业的应力场和位移场可视化图
3. 创建符合工程规范的结果图

使用方法：
    python visualize_elasticity_results.py
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import deepxde as dde
import os

# ============================================================================
# 配置
# ============================================================================
# 结果目录
result_dir = "/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master/exp/pinn_elasticity_plate"
# 模型文件路径（如果需要重新加载模型）
model_file = None  # 如果有保存的模型，可以指定路径

# 可视化网格分辨率（用于生成平滑的场图）
nx, ny = 100, 100  # 100x100的网格

# ============================================================================
# 1. 读取数据文件
# ============================================================================
print("=" * 60)
print("读取数据文件...")
print("=" * 60)

# 读取损失历史
loss_file = os.path.join(result_dir, "loss.dat")
if os.path.exists(loss_file):
    loss_data = np.loadtxt(loss_file, skiprows=1)
    print(f"✓ 损失历史数据已读取: {loss_file}")
    print(f"  数据形状: {loss_data.shape}")
else:
    print(f"✗ 未找到损失历史文件: {loss_file}")
    loss_data = None

# 读取训练点数据
train_file = os.path.join(result_dir, "train.dat")
if os.path.exists(train_file):
    train_data = np.loadtxt(train_file, skiprows=1)
    print(f"✓ 训练点数据已读取: {train_file}")
    print(f"  数据形状: {train_data.shape}")
else:
    print(f"✗ 未找到训练点文件: {train_file}")
    train_data = None

# 读取测试点数据
test_file = os.path.join(result_dir, "test.dat")
if os.path.exists(test_file):
    test_data = np.loadtxt(test_file, skiprows=1)
    print(f"✓ 测试点数据已读取: {test_file}")
    print(f"  数据形状: {test_data.shape}")
    
    # test.dat格式：x, y, ux_true, uy_true, Sxx_true, Syy_true, Sxy_true, 
    #                ux_pred, uy_pred, Sxx_pred, Syy_pred, Sxy_pred
    # 前2列是坐标，接下来5列是真实值，最后5列是预测值
    X_test = test_data[:, 0:2]  # 坐标 (N, 2)
    y_true = test_data[:, 2:7]  # 真实值 (N, 5): [ux, uy, Sxx, Syy, Sxy]
    y_pred = test_data[:, 7:12]  # 预测值 (N, 5): [ux, uy, Sxx, Syy, Sxy]
    
    print(f"  测试点数量: {X_test.shape[0]}")
    print(f"  物理量数量: {y_true.shape[1]}")
else:
    print(f"✗ 未找到测试点文件: {test_file}")
    X_test = None
    y_true = None
    y_pred = None

print()

# ============================================================================
# 2. 准备可视化网格
# ============================================================================
print("=" * 60)
print("准备可视化网格...")
print("=" * 60)

# 创建规则网格用于插值
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
x_grid = np.linspace(x_min, x_max, nx)
y_grid = np.linspace(y_min, y_max, ny)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

# 如果测试点数据存在，进行插值
if X_test is not None and y_true is not None and y_pred is not None:
    from scipy.interpolate import griddata
    
    # 对每个物理量进行插值
    fields_true = {}
    fields_pred = {}
    field_names = ['ux', 'uy', 'Sxx', 'Syy', 'Sxy']
    
    for i, name in enumerate(field_names):
        # 插值真实值
        field_true = griddata(X_test, y_true[:, i], (X_grid, Y_grid), method='cubic')
        fields_true[name] = field_true
        
        # 插值预测值
        field_pred = griddata(X_test, y_pred[:, i], (X_grid, Y_grid), method='cubic')
        fields_pred[name] = field_pred
        
        print(f"✓ {name} 场已插值到网格")
    
    print(f"  网格大小: {nx} × {ny}")
else:
    print("✗ 无法进行插值：缺少测试点数据")
    fields_true = None
    fields_pred = None

print()

# ============================================================================
# 3. 可视化函数
# ============================================================================

def plot_field_2d(X, Y, field, title, save_path, cmap='RdBu_r', vmin=None, vmax=None):
    """
    绘制2D场的热力图（专业格式）
    
    Args:
        X, Y: 网格坐标
        field: 场值
        title: 图标题
        save_path: 保存路径
        cmap: 颜色映射
        vmin, vmax: 颜色范围
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 设置颜色范围
    if vmin is None:
        vmin = np.nanmin(field)
    if vmax is None:
        vmax = np.nanmax(field)
    
    # 绘制热力图
    im = ax.contourf(X, Y, field, levels=50, cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
    ax.contour(X, Y, field, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    
    # 添加颜色条
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(title, rotation=270, labelpad=20)
    
    # 设置坐标轴
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {save_path}")

def plot_displacement_vector(X, Y, ux, uy, title, save_path, scale=1.0):
    """
    绘制位移矢量场
    
    Args:
        X, Y: 网格坐标
        ux, uy: x和y方向的位移
        title: 图标题
        save_path: 保存路径
        scale: 矢量缩放因子
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 计算位移大小
    u_magnitude = np.sqrt(ux**2 + uy**2)
    
    # 绘制位移大小（背景）
    im = ax.contourf(X, Y, u_magnitude, levels=50, cmap='viridis', alpha=0.7)
    
    # 绘制位移矢量（每隔几个点绘制一个，避免过于密集）
    skip = max(1, nx // 20)  # 每20个点绘制一个矢量
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
              ux[::skip, ::skip], uy[::skip, ::skip],
              u_magnitude[::skip, ::skip], 
              scale=scale, angles='xy', scale_units='xy', 
              cmap='viridis', alpha=0.8)
    
    # 添加颜色条
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Displacement Magnitude', rotation=270, labelpad=20)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {save_path}")

# ============================================================================
# 4. 生成专业可视化图
# ============================================================================
print("=" * 60)
print("生成专业可视化图...")
print("=" * 60)

if fields_true is not None and fields_pred is not None:
    # 创建输出目录
    vis_dir = os.path.join(result_dir, "professional_visualization")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 4.1 位移场可视化
    print("\n[1/3] 位移场可视化...")
    plot_field_2d(X_grid, Y_grid, fields_pred['ux'], 
                   'Displacement ux (x-direction)', 
                   os.path.join(vis_dir, 'displacement_ux.png'),
                   cmap='RdBu_r')
    plot_field_2d(X_grid, Y_grid, fields_pred['uy'], 
                   'Displacement uy (y-direction)', 
                   os.path.join(vis_dir, 'displacement_uy.png'),
                   cmap='RdBu_r')
    plot_displacement_vector(X_grid, Y_grid, fields_pred['ux'], fields_pred['uy'],
                            'Displacement Vector Field',
                            os.path.join(vis_dir, 'displacement_vector.png'))
    
    # 4.2 应力场可视化
    print("\n[2/3] 应力场可视化...")
    plot_field_2d(X_grid, Y_grid, fields_pred['Sxx'], 
                   'Stress Sxx (x-direction normal stress)', 
                   os.path.join(vis_dir, 'stress_Sxx.png'),
                   cmap='RdBu_r')
    plot_field_2d(X_grid, Y_grid, fields_pred['Syy'], 
                   'Stress Syy (y-direction normal stress)', 
                   os.path.join(vis_dir, 'stress_Syy.png'),
                   cmap='RdBu_r')
    plot_field_2d(X_grid, Y_grid, fields_pred['Sxy'], 
                   'Stress Sxy (shear stress)', 
                   os.path.join(vis_dir, 'stress_Sxy.png'),
                   cmap='RdBu_r')
    
    # 4.3 误差分析
    print("\n[3/3] 误差分析...")
    for name in field_names:
        error = np.abs(fields_true[name] - fields_pred[name])
        plot_field_2d(X_grid, Y_grid, error, 
                       f'Absolute Error: {name}', 
                       os.path.join(vis_dir, f'error_{name}.png'),
                       cmap='hot')
    
    # 4.4 对比图（真实值 vs 预测值）
    print("\n[4/4] 生成对比图...")
    for name in field_names:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 真实值
        vmin = min(np.nanmin(fields_true[name]), np.nanmin(fields_pred[name]))
        vmax = max(np.nanmax(fields_true[name]), np.nanmax(fields_pred[name]))
        
        im1 = axes[0].contourf(X_grid, Y_grid, fields_true[name], 
                               levels=50, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[0].set_title(f'True {name}', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].set_aspect('equal')
        plt.colorbar(im1, ax=axes[0])
        
        # 预测值
        im2 = axes[1].contourf(X_grid, Y_grid, fields_pred[name], 
                               levels=50, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[1].set_title(f'Predicted {name}', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        axes[1].set_aspect('equal')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'comparison_{name}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 已保存对比图: comparison_{name}.png")
    
    print(f"\n✓ 所有可视化图已保存到: {vis_dir}")
else:
    print("✗ 无法生成可视化图：缺少必要数据")

print()
print("=" * 60)
print("可视化完成！")
print("=" * 60)
print("\n生成的文件说明：")
print("  位移场：")
print("    - displacement_ux.png: x方向位移场")
print("    - displacement_uy.png: y方向位移场")
print("    - displacement_vector.png: 位移矢量场（箭头表示方向，颜色表示大小）")
print("  应力场：")
print("    - stress_Sxx.png: x方向正应力")
print("    - stress_Syy.png: y方向正应力")
print("    - stress_Sxy.png: 剪应力")
print("  误差分析：")
print("    - error_*.png: 各物理量的绝对误差分布")
print("  对比图：")
print("    - comparison_*.png: 真实值 vs 预测值对比")
print()
