"""
模型评估与可视化工具函数
封装了模型预测、指标计算、结果可视化和数据保存的完整流程
"""
import os
import numpy as np
import deepxde as dde
from utils.save_results import (
    plot_and_save_loss_history,
    save_loss_history,
    save_loss_history_json,
    save_best_test_loss_json,
    plot_all_loss_components,
    plot_all_elasticity_fields_with_train,
    plot_deformed_mesh,
    plot_displacement_vector_field
)
from utils.data_saving_utils import save_all_training_data
from utils.true_solution_utils import calculate_accuracy_metrics


def run_evaluation(
    model, 
    X_test, 
    X_train, 
    func, 
    save_dir, 
    losshistory, 
    train_state, 
    bcs,
    bc_loss_names=None,
    bc_label="BC Loss",
    data_loss_prefix="obs_",
    plot_visualization=True,
    calculate_metrics=True,
    save_data=True
):
    """
    运行模型评估流程
    
    Args:
        model: DeepXDE模型对象
        X_test: 测试集输入坐标
        X_train: 训练集输入坐标
        func: 精确解函数
        save_dir: 结果保存目录
        losshistory: 损失历史对象
        train_state: 训练状态对象
        bcs: 边界条件列表
        plot_visualization: 是否生成可视化图表
        calculate_metrics: 是否计算精度指标
        save_data: 是否保存预测数据
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 1. 损失历史可视化 (仅在有训练历史时进行)
    # -------------------------------------------------------------------------
    if plot_visualization and losshistory is not None and len(losshistory.loss_train) > 0:
        print("\n" + "="*60)
        print("正在生成损失历史可视化图...")
        print("="*60)
        
        # 自动推断损失分量数量
        num_pde_losses = 5  # 对于弹性板问题，默认5个PDE残差
        num_bc_losses = len(bcs)
        
        # 检查损失历史结构
        first_loss = losshistory.loss_train[0]
        if isinstance(first_loss, (list, np.ndarray)) and len(first_loss) > 0:
            total_loss_components = len(first_loss)
            if total_loss_components != num_pde_losses + num_bc_losses:
                # 如果数量不匹配，使用自动推断
                num_bc_losses = max(0, total_loss_components - num_pde_losses)
        
        # 保存损失历史图
        plot_and_save_loss_history(
            losshistory, save_dir,
            filename="损失历史详细图.png",
            num_pde_losses=num_pde_losses,
            num_bc_losses=num_bc_losses,
            bc_loss_names=bc_loss_names,
            bc_label=bc_label,
            data_loss_prefix=data_loss_prefix
        )
        
        save_loss_history(losshistory, save_dir, filename="loss_history.dat")
        save_loss_history_json(losshistory, save_dir, filename="loss_history.json")
        save_best_test_loss_json(losshistory, save_dir, filename="best_test_loss.json")
        
        # 生成所有损失项详细图
        pde_loss_names = ['momentum_x', 'momentum_y', 'stress_x', 'stress_y', 'stress_xy']
        if bc_loss_names is None and num_bc_losses > 0:
            bc_loss_names = [f'BC_{i+1}' for i in range(num_bc_losses)]
            
        plot_all_loss_components(
            losshistory, save_dir,
            filename="所有损失项详细图.png",
            num_pde_losses=num_pde_losses,
            num_bc_losses=num_bc_losses,
            pde_loss_names=pde_loss_names,
            bc_loss_names=bc_loss_names
        )

    # -------------------------------------------------------------------------
    # 2. 模型预测
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("正在进行模型预测...")
    print("="*60)
    
    # 测试集预测
    print(f"测试点形状: {X_test.shape}")
    y_pred_test = model.predict(X_test)
    print(f"测试集预测值形状: {y_pred_test.shape}")
    y_true_test = func(X_test)
    print(f"测试集真实值形状: {y_true_test.shape}")
    
    # 训练集预测 (用于对比)
    if X_train is not None:
        print(f"训练点形状: {X_train.shape}")
        y_pred_train = model.predict(X_train)
        print(f"训练集预测值形状: {y_pred_train.shape}")
        y_true_train = func(X_train)
        print(f"训练集真实值形状: {y_true_train.shape}")
    else:
        y_pred_train = None
        y_true_train = None

    # -------------------------------------------------------------------------
    # 3. 场可视化
    # -------------------------------------------------------------------------
    if plot_visualization:
        print("\n" + "="*60)
        print("正在生成专业的应力场和位移场可视化图...")
        print("="*60)
        
        field_names = ['ux', 'uy', 'Sxx', 'Syy', 'Sxy']
        x_vis = np.linspace(0.0, 1.0, 100)
        y_vis = np.linspace(0.0, 1.0, 100)
        X_vis_grid, Y_vis_grid = np.meshgrid(x_vis, y_vis)
        X_vis = np.hstack([X_vis_grid.ravel()[:, None], Y_vis_grid.ravel()[:, None]])
        y_pred_vis = model.predict(X_vis)
        y_true_vis = func(X_vis)
        
        # 场对比图、相对误差图、切片图
        if X_train is not None:
            plot_all_elasticity_fields_with_train(
                X_vis, y_true_vis, y_pred_vis,
                X_train, y_true_train, y_pred_train,
                save_dir,
                field_names=field_names,
                nx=100, ny=100,
                x_min=0.0, x_max=1.0,
                y_min=0.0, y_max=1.0,
                slice_x=0.5
            )
        else:
            plot_all_elasticity_fields_with_train(
                X_vis, y_true_vis, y_pred_vis,
                None, None, None,
                save_dir,
                field_names=field_names,
                nx=100, ny=100,
                x_min=0.0, x_max=1.0,
                y_min=0.0, y_max=1.0,
                slice_x=0.5
            )
        
        # 变形网格与矢量场
        print("\n" + "="*60)
        print("正在生成变形网格可视化图...")
        print("="*60)
        
        ux_test = y_pred_test[:, 0]
        uy_test = y_pred_test[:, 1]
        
        max_displacement = max(np.max(np.abs(ux_test)), np.max(np.abs(uy_test)))
        scale_factor = min(5.0, 0.3 / max_displacement) if max_displacement > 0 else 1.0
        print(f"位移放大系数: {scale_factor:.2f} (最大位移: {max_displacement:.6f})")
        
        plot_deformed_mesh(
            X_test, ux_test, uy_test, save_dir,
            filename="变形网格对比图_测试集.png",
            scale_factor=scale_factor,
            nx=20, ny=20,
            x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0
        )
        
        plot_displacement_vector_field(
            X_test, ux_test, uy_test, save_dir,
            filename="位移矢量场图_测试集.png",
            nx=20, ny=20,
            x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0,
            scale=1.0
        )

    # -------------------------------------------------------------------------
    # 4. 精度指标计算
    # -------------------------------------------------------------------------
    accuracy_metrics = None
    if calculate_metrics:
        print("\n" + "="*60)
        print("正在计算精度指标...")
        print("="*60)
        
        accuracy_metrics = calculate_accuracy_metrics(
            y_true_test, y_pred_test, save_dir=save_dir
        )
        
        if accuracy_metrics:
            print("\n测试集精度指标:")
            for field_name, metrics in accuracy_metrics.items():
                print(f"  {field_name}:")
                l2_rel = metrics.get('L2_rel_error', metrics.get('l2_relative_error', None))
                max_abs = metrics.get('max_abs_error', metrics.get('max_absolute_error', None))
                mean_abs = metrics.get('mean_abs_error', metrics.get('mean_absolute_error', None))
                
                if l2_rel is not None:
                    print(f"    - L2相对误差: {l2_rel:.6e}")
                else:
                    print(f"    - L2相对误差: N/A")
                
                if max_abs is not None:
                    print(f"    - 最大绝对误差: {max_abs:.6e}")
                else:
                    print(f"    - 最大绝对误差: N/A")
                    
                if mean_abs is not None:
                    print(f"    - 平均绝对误差: {mean_abs:.6e}")
                else:
                    print(f"    - 平均绝对误差: N/A")

    # -------------------------------------------------------------------------
    # 5. 数据保存
    # -------------------------------------------------------------------------
    if save_data:
        print("\n" + "="*60)
        print("正在保存预测数据...")
        print("="*60)
        
        save_all_training_data(
            losshistory=losshistory,
            X_test=X_test,
            y_pred_test=y_pred_test,
            y_true_test=y_true_test,
            X_train=X_train,
            y_pred_train=y_pred_train,
            y_true_train=y_true_train,
            test_delta=None,
            save_dir=save_dir
        )

    print("\n" + "="*60)
    print("评估流程完成！")
    print(f"结果已保存到: {save_dir}")
    print("="*60)
