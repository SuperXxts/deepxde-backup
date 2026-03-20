"""
可视化工具函数
用于生成模型预测结果的可视化图
"""
import os
import numpy as np
from .save_results import (
    plot_all_elasticity_fields_with_train,
    plot_deformed_mesh,
    plot_displacement_vector_field
)


def generate_test_points(geom, nx=100, ny=100, x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5):
    """
    生成测试点（用于可视化）
    
    Args:
        geom: DeepXDE几何对象
        nx, ny: 网格分辨率
        x_min, x_max, y_min, y_max: 坐标范围
    
    Returns:
        X_test: 测试点坐标，形状为 (N, 2)
    """
    x_test = np.linspace(x_min, x_max, nx)
    y_test = np.linspace(y_min, y_max, ny)
    X_test_grid, Y_test_grid = np.meshgrid(x_test, y_test)
    X_test = np.hstack([X_test_grid.ravel()[:, None], Y_test_grid.ravel()[:, None]])
    
    # 只保留几何域内的点
    inside_mask = geom.inside(X_test)
    X_test = X_test[inside_mask]
    
    return X_test


def predict_with_model(model, X_test, v_test):
    """
    使用模型进行预测，并处理输出格式
    
    Args:
        model: 训练好的模型
        X_test: 测试点坐标，形状为 (N, 2)
        v_test: Branch Net输入，形状为 (1, 1)
    
    Returns:
        y_pred: 预测值，形状为 (N, 5)
    """
    # 使用模型预测
    y_pred_raw = model.predict((v_test, X_test))
    
    # 处理输出格式
    if isinstance(y_pred_raw, (list, tuple)):
        if len(y_pred_raw) == 1:
            y_pred = y_pred_raw[0]
        else:
            y_pred = np.stack(y_pred_raw, axis=-1)
    else:
        y_pred = y_pred_raw
    
    # 确保输出形状正确
    if len(y_pred.shape) == 2:
        y_pred = y_pred[np.newaxis, :, :]
    elif len(y_pred.shape) == 3:
        pass
    else:
        raise ValueError(f"意外的输出形状: {y_pred.shape}")
    
    # 如果第一个维度是1，可以压缩
    if y_pred.shape[0] == 1:
        y_pred = y_pred[0]  # 形状: (num_points, num_outputs)
    
    return y_pred


def generate_all_visualizations(model, geom, test_delta, save_dir,
                               use_hondros=False, true_solution_path=None,
                               R=0.5, alpha_deg=10.0, lmbd=1.0, mu=0.5,
                               nx=100, ny=100, x_min=-0.5, x_max=0.5, 
                               y_min=-0.5, y_max=0.5):
    """
    生成所有可视化图（统一接口）
    
    Args:
        model: 训练好的模型
        geom: DeepXDE几何对象
        test_delta: 测试δ值（归一化后）
        save_dir: 保存目录
        use_hondros: 是否使用Hondros解析解作为真值
        true_solution_path: 真值数据文件路径
        R: 圆盘半径（归一化后）
        alpha_deg: 加载半角（度）
        lmbd: 拉梅常数（归一化后）
        mu: 剪切模量（归一化后）
        nx, ny: 网格分辨率
        x_min, x_max, y_min, y_max: 坐标范围
    
    Returns:
        results: 包含预测值、真值等信息的字典
    """
    from .true_solution_utils import get_true_solution, calculate_accuracy_metrics
    
    print("\n" + "="*60)
    print("正在生成专业的应力场和位移场可视化图（训练集+测试集）...")
    print("="*60)
    
    print(f"使用测试δ值: {test_delta} (归一化后)")
    
    # 1. 生成测试点
    X_test = generate_test_points(geom, nx=nx, ny=ny, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    print(f"测试点形状: {X_test.shape} (几何域内点)")
    
    # 2. 准备Branch Net输入
    v_test = np.array([[test_delta]])  # 形状: (1, 1)
    
    # 3. 使用模型预测
    y_pred_test = predict_with_model(model, X_test, v_test)
    print(f"测试集预测值形状: {y_pred_test.shape} (应该是 (num_points, 5))")
    
    # 4. 获取真值
    y_true_test, solution_type = get_true_solution(
        X_test, test_delta, 
        use_hondros=use_hondros, 
        true_solution_path=true_solution_path,
        R=R, alpha_deg=alpha_deg, lmbd=lmbd, mu=mu
    )
    
    # 如果没有真值，使用预测值作为真值（仅用于可视化）
    if y_true_test is None:
        y_true_test = y_pred_test.copy()
        print(f"测试集真实值形状: {y_true_test.shape} (注意：这是预测值的副本，用于可视化)")
    else:
        print(f"测试集真实值形状: {y_true_test.shape} (真值类型: {solution_type})")
    
    # 5. 获取训练点（使用测试点作为近似）
    X_train = X_test.copy()
    print(f"训练点形状: {X_train.shape} (使用测试点作为近似)")
    
    # 6. 预测训练集
    y_pred_train = predict_with_model(model, X_train, v_test)
    print(f"训练集预测值形状: {y_pred_train.shape} (应该是 (num_points, 5))")
    
    # 7. 训练集真值（使用预测值作为真值）
    y_true_train = y_pred_train.copy()
    print(f"训练集真实值形状: {y_true_train.shape} (注意：这是预测值的副本，用于可视化)")
    
    # 8. 生成所有场的专业可视化图
    # 归一化系数（用于单位还原）
    L_norm = 0.05  # 长度归一化系数：0.05 m
    S_norm = 4.9e10  # 应力归一化系数：4.9e10 Pa
    
    field_names = ['ux', 'uy', 'Sxx', 'Syy', 'Sxy']
    plot_all_elasticity_fields_with_train(
        X_test, y_true_test, y_pred_test,  # 测试集
        X_train, y_true_train, y_pred_train,  # 训练集（验证阶段）
        save_dir,
        field_names=field_names,
        nx=nx, ny=ny,
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        slice_x=0.0,  # 在x=0处切片（通过圆心）
        L_norm=L_norm,  # 长度归一化系数，用于单位还原
        S_norm=S_norm   # 应力归一化系数，用于单位还原
    )
    
    # 9. 生成变形网格可视化
    print("\n" + "="*60)
    print("正在生成变形网格可视化图（直观显示圆盘变形）...")
    print("="*60)
    
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
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
    )
    
    # 绘制位移矢量场
    plot_displacement_vector_field(
        X_test, ux_test, uy_test, save_dir,
        filename="位移矢量场图_测试集.png",
        nx=20, ny=20,
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        scale=1.0
    )
    
    # 10. 计算精度指标（如果有真值）
    accuracy_metrics = None
    if solution_type != 'prediction' and y_true_test is not None:
        accuracy_metrics = calculate_accuracy_metrics(
            y_true_test, y_pred_test, 
            field_names=field_names, 
            save_dir=save_dir
        )
    
    # 11. 打印生成的文件列表
    print("\n" + "="*60)
    print("所有可视化图生成完成！")
    print(f"结果已保存到: {save_dir}")
    print("="*60)
    print("\n文件存储结构:")
    print("  png/  - 所有图片文件（.png）")
    print("  npz/  - 所有numpy数据文件（.npz, .npy）")
    print("  txt/  - 所有文本文件（.txt, .dat）")
    print("  json/ - 所有JSON配置文件（.json）")
    print("\n生成的文件:")
    print("  png/损失历史详细图.png: 详细损失历史图（PDE Loss、BC Loss、总Loss）")
    print("  png/所有损失项详细图.png: 所有损失项的详细图")
    print("  png/场对比图_测试集_*.png: 测试集场对比图（5个场：ux, uy, Sxx, Syy, Sxy）")
    print("  png/场对比图_训练集_*.png: 训练集场对比图（5个场：ux, uy, Sxx, Syy, Sxy）")
    print("  png/相对误差图_测试集_*.png: 测试集相对误差图（5个场）")
    print("  png/相对误差图_训练集_*.png: 训练集相对误差图（5个场）")
    print("  png/切片对比图_测试集_x0.00.png: 测试集切片对比图（沿x=0，通过圆心）")
    print("  png/切片对比图_训练集_x0.00.png: 训练集切片对比图（沿x=0，通过圆心）")
    print("  png/变形网格对比图_测试集.png: 变形网格可视化")
    print("  png/位移矢量场图_测试集.png: 位移矢量场可视化")
    print("  npz/*.npz: 预测数据（坐标、预测值、真值等）")
    print("  txt/*.txt: 文本格式的预测数据")
    print("  txt/loss_history.dat: 损失历史数据")
    print("  json/*.json: 元数据和精度指标")
    if accuracy_metrics is not None:
        print("  json/accuracy_metrics.json: 精度指标")
    print("="*60)
    
    return {
        'X_test': X_test,
        'y_pred_test': y_pred_test,
        'y_true_test': y_true_test,
        'X_train': X_train,
        'y_pred_train': y_pred_train,
        'y_true_train': y_true_train,
        'accuracy_metrics': accuracy_metrics,
        'solution_type': solution_type
    }
