"""
数据保存工具函数
用于保存预测结果、场数据等，便于后续分析
"""
import os
import numpy as np
import json


def _get_save_path(save_dir, subfolder, filename):
    """
    获取保存路径，自动创建子文件夹
    
    Args:
        save_dir: 主保存目录
        subfolder: 子文件夹名称（如 'npz', 'txt', 'json'）
        filename: 文件名
    
    Returns:
        file_path: 完整的文件路径
    """
    os.makedirs(save_dir, exist_ok=True)
    subfolder_path = os.path.join(save_dir, subfolder)
    os.makedirs(subfolder_path, exist_ok=True)
    return os.path.join(subfolder_path, filename)


def save_prediction_data(X, y_pred, y_true, test_delta, save_dir, 
                         prefix="prediction", field_names=None):
    """
    保存预测结果数据（坐标、预测值、真值等）
    
    Args:
        X: 坐标点，形状为 (N, 2)
        y_pred: 预测值，形状为 (N, 5)
        y_true: 真值，形状为 (N, 5)，如果为None则使用预测值
        test_delta: 测试δ值（归一化后）
        save_dir: 保存目录
        prefix: 文件名前缀
        field_names: 场名称列表，默认 ['ux', 'uy', 'Sxx', 'Syy', 'Sxy']
    
    Returns:
        saved_files: 保存的文件路径列表
    """
    if field_names is None:
        field_names = ['ux', 'uy', 'Sxx', 'Syy', 'Sxy']
    
    os.makedirs(save_dir, exist_ok=True)
    saved_files = []
    
    # 如果没有真值，使用预测值作为真值
    if y_true is None:
        y_true = y_pred.copy()
    
    # 1. 保存为.npz格式（推荐，便于后续加载）
    npz_path = _get_save_path(save_dir, 'npz', f"{prefix}_data.npz")
    # 处理test_delta为None的情况（PINN不需要delta参数）
    save_dict = {
        'X': X,
        'y_pred': y_pred,
        'y_true': y_true,
        'field_names': np.array(field_names)
    }
    if test_delta is not None:
        save_dict['delta'] = np.array([test_delta])
    np.savez(npz_path, **save_dict)
    saved_files.append(npz_path)
    print(f"  ✓ 已保存: {npz_path}")
    
    # 2. 分别保存每个场的数据（便于单独分析）
    for i, field_name in enumerate(field_names):
        field_data = {
            'X': X,
            'delta': test_delta if test_delta is not None else None,
            f'{field_name}_pred': y_pred[:, i],
            f'{field_name}_true': y_true[:, i],
            f'{field_name}_error': y_true[:, i] - y_pred[:, i],
            f'{field_name}_abs_error': np.abs(y_true[:, i] - y_pred[:, i])
        }
        
        # 计算相对误差（避免除以0）
        mask = np.abs(y_true[:, i]) > 1e-10
        rel_error = np.zeros_like(y_true[:, i])
        if np.any(mask):
            rel_error[mask] = np.abs((y_true[:, i][mask] - y_pred[:, i][mask]) / y_true[:, i][mask])
        field_data[f'{field_name}_rel_error'] = rel_error
        
        field_npz_path = _get_save_path(save_dir, 'npz', f"{prefix}_{field_name}.npz")
        np.savez(field_npz_path, **field_data)
        saved_files.append(field_npz_path)
    
    # 3. 保存为文本格式（便于查看，但文件较大）
    txt_path = _get_save_path(save_dir, 'txt', f"{prefix}_data.txt")
    with open(txt_path, 'w') as f:
        f.write("# 预测结果数据\n")
        if test_delta is not None:
            f.write(f"# δ值（归一化后）: {test_delta}\n")
        else:
            f.write(f"# 注意: PINN问题，无δ参数\n")
        f.write("# 列: x, y, ux_pred, uy_pred, Sxx_pred, Syy_pred, Sxy_pred, ")
        f.write("ux_true, uy_true, Sxx_true, Syy_true, Sxy_true\n")
        data_array = np.hstack([X, y_pred, y_true])
        np.savetxt(f, data_array, fmt='%.6e', delimiter='\t')
    saved_files.append(txt_path)
    
    # 4. 保存元数据（JSON格式）
    metadata = {
        'delta': float(test_delta) if test_delta is not None else None,
        'num_points': int(X.shape[0]),
        'field_names': field_names,
        'X_shape': list(X.shape),
        'y_pred_shape': list(y_pred.shape),
        'y_true_shape': list(y_true.shape)
    }
    json_path = _get_save_path(save_dir, 'json', f"{prefix}_metadata.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    saved_files.append(json_path)
    
    return saved_files


def save_loss_data(losshistory, save_dir, prefix="loss"):
    """
    保存损失历史数据（除了loss.dat，还保存详细的JSON格式）
    
    Args:
        losshistory: DeepXDE的LossHistory对象
        save_dir: 保存目录
        prefix: 文件名前缀
    
    Returns:
        saved_files: 保存的文件路径列表
    """
    if losshistory is None:
        return []
    os.makedirs(save_dir, exist_ok=True)
    saved_files = []
    
    # 1. 保存为JSON格式（便于后续分析和可视化）
    # 注意：需要将numpy类型转换为Python原生类型，以便JSON序列化
    def convert_to_python_type(obj):
        """递归转换numpy类型为Python原生类型"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, (list, tuple)):
            return [convert_to_python_type(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_python_type(value) for key, value in obj.items()}
        else:
            return obj
    
    loss_data = {
        'steps': [int(s) for s in losshistory.steps],
        'loss_train': convert_to_python_type(losshistory.loss_train),
        # 注意：losshistory.loss_test实际上是验证loss（虽然DeepXDE API叫test）
        'loss_validation': convert_to_python_type(losshistory.loss_test),
        'metrics_validation': convert_to_python_type(losshistory.metrics_test) if hasattr(losshistory, 'metrics_test') and losshistory.metrics_test is not None else None,
        # 保留原始API命名以便兼容
        'loss_test': convert_to_python_type(losshistory.loss_test),
        'metrics_test': convert_to_python_type(losshistory.metrics_test) if hasattr(losshistory, 'metrics_test') and losshistory.metrics_test is not None else None
    }
    
    json_path = _get_save_path(save_dir, 'json', f"{prefix}_history.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(loss_data, f, indent=2, ensure_ascii=False)
    saved_files.append(json_path)
    
    return saved_files


def save_all_training_data(losshistory, X_test, y_pred_test, y_true_test,
                           X_train, y_pred_train, y_true_train,
                           test_delta, save_dir):
    """
    保存所有训练和预测数据（一站式函数）
    
    Args:
        losshistory: 损失历史对象
        X_test: 测试集坐标
        y_pred_test: 测试集预测值
        y_true_test: 测试集真值
        X_train: 训练集坐标
        y_pred_train: 训练集预测值
        y_true_train: 训练集真值
        test_delta: 测试δ值
        save_dir: 保存目录
    
    Returns:
        saved_files: 所有保存的文件路径
    """
    saved_files = []
    
    print("\n" + "="*60)
    print("正在保存所有训练和预测数据...")
    print("="*60)
    
    if losshistory is not None and hasattr(losshistory, "loss_train") and len(losshistory.loss_train) > 0:
        print("\n[1/3] 保存损失历史数据...")
        loss_files = save_loss_data(losshistory, save_dir, prefix="loss")
        saved_files.extend(loss_files)
    else:
        print("\n[1/3] 跳过损失历史数据保存")
    
    if X_test is not None and y_pred_test is not None:
        print("\n[2/3] 保存测试集预测数据...")
        test_files = save_prediction_data(
            X_test, y_pred_test, y_true_test, test_delta, save_dir,
            prefix="test_set"
        )
        saved_files.extend(test_files)
    else:
        print("\n[2/3] 跳过测试集预测数据保存")
    
    if X_train is not None and y_pred_train is not None:
        print("\n[3/3] 保存训练集预测数据...")
        train_files = save_prediction_data(
            X_train, y_pred_train, y_true_train, test_delta, save_dir,
            prefix="train_set"
        )
        saved_files.extend(train_files)
    else:
        print("\n[3/3] 跳过训练集预测数据保存")
    
    print("\n" + "="*60)
    print(f"所有数据已保存完成！共保存 {len(saved_files)} 个文件")
    print(f"保存目录: {save_dir}")
    print("\n文件存储结构:")
    print("  npz/  - 所有numpy数据文件（.npz）")
    print("  txt/  - 所有文本文件（.txt）")
    print("  json/ - 所有JSON配置文件（.json）")
    print("="*60)
    
    return saved_files
