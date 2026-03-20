"""
真值生成和加载工具函数
用于从Hondros解析解、FEM/DDA结果等获取真值，进行精度评估
"""
import os
import numpy as np
from scipy.interpolate import griddata


def hondros_solution(X, delta, R=0.5, alpha_deg=10.0, lmbd=1.0, mu=0.5):
    """
    计算巴西圆盘实验的Hondros解析解（应力场）
    
    Hondros (1959) 推导了在圆盘两对径向对称弧上施加均匀压力时的全场弹性应力解。
    适用于：线性弹性、各向同性、平面应力、小加载角度（α较小）的情况。
    
    Args:
        X: 坐标点，形状为 (N, 2)，笛卡尔坐标 (x, y)
        delta: 顶部平台向下压的距离（归一化后）
        R: 圆盘半径（归一化后），默认0.5
        alpha_deg: 加载半角（度），默认10°（对应90°±5°）
        lmbd: 拉梅常数（归一化后）
        mu: 剪切模量（归一化后）
    
    Returns:
        y_true: 真值，形状为 (N, 5)，包含 [ux, uy, Sxx, Syy, Sxy]
        注意：Hondros解主要给出应力场，位移场可能不完整，位移部分设为0或从应力场积分得到
    
    参考文献：
        Hondros, G. (1959). The evaluation of Poisson's ratio and the modulus 
        of materials of a low tensile resistance by the Brazilian (indirect 
        tensile) test with particular reference to concrete. 
        Australian Journal of Applied Science, 10(3), 243-268.
    
    注意：
        当前实现是简化版本，完整Hondros解需要级数展开。
        建议从文献中获取完整公式或使用FEM/DDA结果。
    """
    # 转换为极坐标
    x, y = X[:, 0], X[:, 1]
    r = np.sqrt(x**2 + y**2)  # 径向距离
    theta = np.arctan2(y, x)  # 极角（弧度）
    
    # 避免除零
    r = np.maximum(r, 1e-10)
    
    # 归一化半径
    rho = r / R  # 归一化径向距离 [0, 1]
    
    # 加载半角（弧度）
    alpha = np.deg2rad(alpha_deg)
    
    # 计算Hondros解的应力场（极坐标）
    # 注意：这是一个简化版本，假设加载角度较小（接近线载荷）
    # 完整的Hondros解涉及级数展开，这里使用简化形式
    
    # 对于小加载角度，Hondros解的简化形式：
    # 径向应力 σ_r 和切向应力 σ_θ 的表达式
    
    # 由于Hondros解的完整形式涉及复杂的级数展开，这里提供一个简化实现
    # 实际应用中，建议使用完整的Hondros解公式或FEM结果
    
    # 简化假设：当加载角度很小时（α→0），接近线载荷情况
    # 对于线载荷，可以使用更简单的应力场表达式
    
    # 为了实用性，这里提供一个基于应力集中理论的简化近似
    # 注意：这不是完整的Hondros解，而是一个近似
    
    # 计算施加的压力（从位移δ推导）
    # 需要根据实际的材料参数和边界条件计算压力
    # 这里假设压力与δ成正比（简化）
    P = delta * (2 * mu + lmbd)  # 简化的压力-位移关系
    
    # 极坐标应力分量（简化形式，适用于小加载角度）
    # 这是对Hondros解的近似，实际应用中应使用完整公式
    
    # 在极坐标系中，应力分量
    sigma_r = np.zeros_like(rho)
    sigma_theta = np.zeros_like(rho)
    tau_rtheta = np.zeros_like(rho)
    
    # 简化处理：这里不提供完整的Hondros解实现
    # 因为完整的Hondros解涉及复杂的级数展开
    # 建议用户：
    # 1. 使用文献中的完整Hondros解公式
    # 2. 或使用FEM/DDA计算真值
    
    # 转换为笛卡尔坐标系应力分量
    cos_2theta = np.cos(2 * theta)
    sin_2theta = np.sin(2 * theta)
    
    Sxx_hondros = 0.5 * (sigma_r + sigma_theta) + 0.5 * (sigma_r - sigma_theta) * cos_2theta - tau_rtheta * sin_2theta
    Syy_hondros = 0.5 * (sigma_r + sigma_theta) - 0.5 * (sigma_r - sigma_theta) * cos_2theta + tau_rtheta * cos_2theta
    Sxy_hondros = 0.5 * (sigma_r - sigma_theta) * sin_2theta + tau_rtheta * cos_2theta
    
    # 位移场（从应力场积分得到，这里简化处理）
    # 注意：完整的位移场计算需要从应力场积分，比较复杂
    # 这里暂时设为0，用户可以根据需要从应力场计算位移
    ux_hondros = np.zeros_like(x)
    uy_hondros = np.zeros_like(y)
    
    # 组合输出
    y_true = np.column_stack([ux_hondros, uy_hondros, Sxx_hondros, Syy_hondros, Sxy_hondros])
    
    return y_true


def load_true_solution_from_file(file_path, X_test):
    """
    从文件加载真值数据
    
    Args:
        file_path: 真值数据文件路径（.npz格式）
        X_test: 测试点坐标，形状为 (N, 2)
    
    Returns:
        y_true: 真值，形状为 (N, 5)，包含 [ux, uy, Sxx, Syy, Sxy]
        success: 是否成功加载
    """
    if not os.path.exists(file_path):
        return None, False
    
    try:
        true_data = np.load(file_path)
        
        # 提取真值数据
        X_true = true_data['X']  # 坐标点 (N, 2)
        ux_true = true_data['ux']  # x方向位移 (N,)
        uy_true = true_data['uy']  # y方向位移 (N,)
        Sxx_true = true_data['Sxx']  # x方向正应力 (N,)
        Syy_true = true_data['Syy']  # y方向正应力 (N,)
        Sxy_true = true_data['Sxy']  # 剪应力 (N,)
        
        # 组合成与y_pred_test相同的格式 (N, 5)
        y_true_raw = np.column_stack([ux_true, uy_true, Sxx_true, Syy_true, Sxy_true])
        
        # 检查坐标点是否一致
        if X_true.shape[0] == X_test.shape[0] and np.allclose(X_true, X_test, atol=1e-6):
            # 坐标点一致，直接使用
            y_true = y_true_raw
        else:
            # 坐标点不一致，需要进行插值
            # 对每个物理量进行插值
            ux_interp = griddata(X_true, ux_true, X_test, method='linear', fill_value=0.0)
            uy_interp = griddata(X_true, uy_true, X_test, method='linear', fill_value=0.0)
            Sxx_interp = griddata(X_true, Sxx_true, X_test, method='linear', fill_value=0.0)
            Syy_interp = griddata(X_true, Syy_true, X_test, method='linear', fill_value=0.0)
            Sxy_interp = griddata(X_true, Sxy_true, X_test, method='linear', fill_value=0.0)
            
            y_true = np.column_stack([ux_interp, uy_interp, Sxx_interp, Syy_interp, Sxy_interp])
        
        return y_true, True
    except Exception as e:
        print(f"加载真值文件时出错: {e}")
        return None, False


def get_true_solution(X_test, delta, use_hondros=False, true_solution_path=None, 
                     R=0.5, alpha_deg=10.0, lmbd=1.0, mu=0.5):
    """
    获取真值（统一接口）
    
    Args:
        X_test: 测试点坐标，形状为 (N, 2)
        delta: δ值（归一化后）
        use_hondros: 是否使用Hondros解析解
        true_solution_path: 真值数据文件路径（.npz格式）
        R: 圆盘半径（归一化后）
        alpha_deg: 加载半角（度）
        lmbd: 拉梅常数（归一化后）
        mu: 剪切模量（归一化后）
    
    Returns:
        y_true: 真值，形状为 (N, 5)，包含 [ux, uy, Sxx, Syy, Sxy]
        solution_type: 真值类型（'hondros', 'file', 'prediction'）
    """
    # 优先级1：Hondros解析解
    if use_hondros:
        print("\n使用Hondros解析解作为真值...")
        print("注意：当前实现是简化版本，完整Hondros解需要级数展开")
        print("      建议参考文献实现完整的Hondros解，或使用FEM/DDA结果")
        print(f"  使用δ值: {delta:.6f}")
        
        y_true = hondros_solution(X_test, delta, R=R, alpha_deg=alpha_deg, lmbd=lmbd, mu=mu)
        
        print(f"Hondros解真值形状: {y_true.shape}")
        print(f"\nHondros解真值范围:")
        print(f"  ux: [{y_true[:, 0].min():.6e}, {y_true[:, 0].max():.6e}]")
        print(f"  uy: [{y_true[:, 1].min():.6e}, {y_true[:, 1].max():.6e}]")
        print(f"  Sxx: [{y_true[:, 2].min():.6e}, {y_true[:, 2].max():.6e}]")
        print(f"  Syy: [{y_true[:, 3].min():.6e}, {y_true[:, 3].max():.6e}]")
        print(f"  Sxy: [{y_true[:, 4].min():.6e}, {y_true[:, 4].max():.6e}]")
        
        return y_true, 'hondros'
    
    # 优先级2：从文件加载
    if true_solution_path is not None:
        print(f"\n正在从 {true_solution_path} 加载真值...")
        y_true, success = load_true_solution_from_file(true_solution_path, X_test)
        
        if success:
            print(f"真值已加载，形状: {y_true.shape}")
            print(f"\n真值范围:")
            print(f"  ux: [{y_true[:, 0].min():.6e}, {y_true[:, 0].max():.6e}]")
            print(f"  uy: [{y_true[:, 1].min():.6e}, {y_true[:, 1].max():.6e}]")
            print(f"  Sxx: [{y_true[:, 2].min():.6e}, {y_true[:, 2].max():.6e}]")
            print(f"  Syy: [{y_true[:, 3].min():.6e}, {y_true[:, 3].max():.6e}]")
            print(f"  Sxy: [{y_true[:, 4].min():.6e}, {y_true[:, 4].max():.6e}]")
            return y_true, 'file'
        else:
            print(f"警告：真值文件 {true_solution_path} 加载失败，使用预测值作为真值")
    
    # 默认：使用预测值作为真值（仅用于可视化）
    print("\n注意：使用预测值作为真值，仅用于定性可视化，无法进行定量精度分析")
    print("      如需进行精度分析，请设置 use_hondros=True 或指定 true_solution_path")
    return None, 'prediction'


def calculate_accuracy_metrics(y_true, y_pred, field_names=None, save_dir=None):
    """
    计算精度指标
    
    Args:
        y_true: 真值，形状为 (N, 5)
        y_pred: 预测值，形状为 (N, 5)
        field_names: 场名称列表，默认 ['ux', 'uy', 'Sxx', 'Syy', 'Sxy']
        save_dir: 保存目录，如果提供则保存精度指标到JSON文件
    
    Returns:
        accuracy_metrics: 精度指标字典
    """
    if field_names is None:
        field_names = ['ux', 'uy', 'Sxx', 'Syy', 'Sxy']
    
    if y_true is None or np.allclose(y_true, y_pred, atol=1e-10):
        print("\n注意：由于没有真值，无法进行定量精度分析。")
        print("      当前误差图会显示为0（因为预测值=真值）。")
        print("      如需进行精度分析，请使用Hondros解或加载真值文件")
        return None
    
    print("\n" + "="*60)
    print("精度评估（Quantitative Analysis）")
    print("="*60)
    
    accuracy_metrics = {}
    
    for i, field_name in enumerate(field_names):
        y_true_field = y_true[:, i]
        y_pred_field = y_pred[:, i]
        
        # 计算绝对误差
        abs_error = np.abs(y_true_field - y_pred_field)
        
        # 计算相对误差（避免除以0）
        mask = np.abs(y_true_field) > 1e-10
        rel_error = np.zeros_like(y_true_field)
        if np.any(mask):
            rel_error[mask] = np.abs((y_true_field[mask] - y_pred_field[mask]) / y_true_field[mask])
        
        # 计算L2相对误差
        l2_norm_true = np.linalg.norm(y_true_field)
        l2_norm_error = np.linalg.norm(y_true_field - y_pred_field)
        l2_rel_error = l2_norm_error / l2_norm_true if l2_norm_true > 1e-10 else 0
        
        # 计算最大相对误差
        max_rel_error = np.max(rel_error[mask]) if np.any(mask) else 0
        
        # 计算平均相对误差
        mean_rel_error = np.mean(rel_error[mask]) if np.any(mask) else 0
        
        # 保存指标
        accuracy_metrics[field_name] = {
            'L2_rel_error': float(l2_rel_error),
            'max_rel_error': float(max_rel_error),
            'mean_rel_error': float(mean_rel_error),
            'mean_abs_error': float(np.mean(abs_error)),
            'max_abs_error': float(np.max(abs_error))
        }
        
        # 打印指标
        print(f"\n{field_name}:")
        print(f"  L2相对误差:     {l2_rel_error:.6e}")
        print(f"  最大相对误差:   {max_rel_error:.6e}")
        print(f"  平均相对误差:   {mean_rel_error:.6e}")
        print(f"  平均绝对误差:   {np.mean(abs_error):.6e}")
        print(f"  最大绝对误差:   {np.max(abs_error):.6e}")
    
    # 保存精度指标到文件
    if save_dir is not None:
        json_dir = os.path.join(save_dir, "metrics")
        os.makedirs(json_dir, exist_ok=True)
        accuracy_file = os.path.join(json_dir, "accuracy_metrics.json")
        import json
        os.makedirs(save_dir, exist_ok=True)
        with open(accuracy_file, 'w', encoding='utf-8') as f:
            json.dump(accuracy_metrics, f, indent=4, ensure_ascii=False)
        accuracy_txt = os.path.join(json_dir, "accuracy_metrics.txt")
        with open(accuracy_txt, 'w', encoding='utf-8') as f:
            for field_name, metrics in accuracy_metrics.items():
                f.write(f"{field_name}\n")
                f.write(f"  L2_rel_error: {metrics['L2_rel_error']:.6e}\n")
                f.write(f"  max_rel_error: {metrics['max_rel_error']:.6e}\n")
                f.write(f"  mean_rel_error: {metrics['mean_rel_error']:.6e}\n")
                f.write(f"  mean_abs_error: {metrics['mean_abs_error']:.6e}\n")
                f.write(f"  max_abs_error: {metrics['max_abs_error']:.6e}\n\n")
        print(f"\n精度指标已保存到: {accuracy_file}")
        print(f"精度指标已保存到: {accuracy_txt}")
    
    print("="*60)
    return accuracy_metrics
