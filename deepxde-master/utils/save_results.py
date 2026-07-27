import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.ticker import MaxNLocator
import json
import datetime
from .data_saving_utils import _get_save_path, save_prediction_data

def save_model_config(args, save_dir):
    """
    保存模型配置参数
    """
    config_path = _get_save_path(save_dir, 'json', 'model_config.json')
    try:
        # Convert args to dict if it's a Namespace
        if hasattr(args, '__dict__'):
            config_dict = vars(args)
        else:
            config_dict = args
            
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        print(f"Model config saved to {config_path}")
    except Exception as e:
        print(f"Failed to save model config: {e}")

def save_loss_history(losshistory, save_dir, filename="loss_history.dat"):
    """
    保存损失历史数据
    """
    save_path = _get_save_path(save_dir, 'dat', filename)
    loss_history = np.array(losshistory.loss_train)
    steps = np.array(losshistory.steps)[:, None]
    data = np.hstack((steps, loss_history))
    np.savetxt(save_path, data, header="step, loss_components...")
    print(f"Loss history saved to {save_path}")

def save_loss_history_json(losshistory, save_dir, filename="loss_history.json"):
    """
    保存损失历史数据为JSON格式
    """
    save_path = _get_save_path(save_dir, 'json', filename)
    
    # 处理steps
    steps = losshistory.steps
    if isinstance(steps, np.ndarray):
        steps = steps.tolist()
        
    # 处理loss_train
    loss_train = np.array(losshistory.loss_train)
    loss_train_list = loss_train.tolist()
    
    # 处理loss_test
    loss_test_list = []
    if losshistory.loss_test:
        loss_test = np.array(losshistory.loss_test)
        loss_test_list = loss_test.tolist()
        
    history = {
        "steps": steps,
        "loss_train": loss_train_list,
        "loss_test": loss_test_list
    }
    
    try:
        with open(save_path, 'w') as f:
            json.dump(history, f)
        print(f"Loss history (JSON) saved to {save_path}")
    except Exception as e:
        print(f"Failed to save loss history json: {e}")

def save_best_test_loss_json(losshistory, save_dir, filename="best_test_loss.json"):
    """
    保存最优测试集损失（loss_test 各分量求和的最小值）
    """
    if not losshistory or not losshistory.loss_test:
        return
    
    try:
        loss_test = np.array(losshistory.loss_test, dtype=float)
        if loss_test.size == 0:
            return
        loss_test_sum = np.sum(loss_test, axis=1)
        min_idx = int(np.argmin(loss_test_sum))
        
        steps = losshistory.steps
        step_val = steps[min_idx] if steps and min_idx < len(steps) else None
        
        payload = {
            "best_test_loss": float(loss_test_sum[min_idx]),
            "best_step": int(step_val) if step_val is not None else None
        }
        save_path = _get_save_path(save_dir, 'json', filename)
        with open(save_path, 'w') as f:
            json.dump(payload, f, indent=4)
        print(f"Best test loss saved to {save_path}")
    except Exception as e:
        print(f"Failed to save best test loss json: {e}")

def plot_and_save_loss_history(
    losshistory,
    save_dir,
    filename="loss_history.png",
    num_pde_losses=None,
    num_bc_losses=None,
    pde_label="Physics Loss",
    bc_label="Boundary Loss",
    bc_loss_names=None,
    data_loss_prefix="obs_",
    data_label="Observation Loss",
    show_total=False,
):
    """
    ???????????
    """
    save_path = _get_save_path(save_dir, 'png', filename)
    loss_train = np.array(losshistory.loss_train)
    steps = losshistory.steps

    loss_train_sum = np.sum(loss_train, axis=1)
    loss_test_sum = np.sum(losshistory.loss_test, axis=1) if losshistory.loss_test else []

    plt.figure(figsize=(10, 6))
    if show_total:
        plt.plot(steps, loss_train_sum, label='Train loss', color='k')
        if len(loss_test_sum) > 0:
            plt.plot(steps, loss_test_sum, label='Test loss', color='r')

    if num_pde_losses is not None and num_bc_losses is not None and num_bc_losses > 0:
        loss_pde = np.sum(loss_train[:, :num_pde_losses], axis=1)
        plt.plot(steps, loss_pde, label=pde_label, linestyle='--')

        if bc_loss_names:
            data_indices = [i for i, name in enumerate(bc_loss_names) if isinstance(name, str) and name.startswith(data_loss_prefix)]
            bc_indices = [i for i in range(num_bc_losses) if i not in data_indices]
            if bc_indices:
                loss_bc = np.sum(loss_train[:, num_pde_losses + np.array(bc_indices)], axis=1)
                plt.plot(steps, loss_bc, label=bc_label, linestyle=':')
            if data_indices:
                loss_data = np.sum(loss_train[:, num_pde_losses + np.array(data_indices)], axis=1)
                plt.plot(steps, loss_data, label=data_label, linestyle='-.')
        else:
            loss_bc = np.sum(loss_train[:, num_pde_losses:num_pde_losses + num_bc_losses], axis=1)
            plt.plot(steps, loss_bc, label=bc_label, linestyle=':')
    elif num_pde_losses is not None and (num_bc_losses is None or num_bc_losses == 0):
        loss_pde = np.sum(loss_train[:, :num_pde_losses], axis=1)
        plt.plot(steps, loss_pde, label=pde_label, linestyle='--')

    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Loss History')
    plt.legend(loc="upper right", frameon=True, fancybox=False, shadow=False)
    plt.grid(True)
    plt.yscale("log")
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.minorticks_off()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_all_loss_components(losshistory, save_dir, filename="所有损失项详细图.png", num_pde_losses=None, num_bc_losses=None, pde_loss_names=None, bc_loss_names=None):
    """
    绘制所有损失分量
    """
    save_path = _get_save_path(save_dir, 'png', filename)
    loss_train = np.array(losshistory.loss_train)
    steps = losshistory.steps
    
    plt.figure(figsize=(12, 8))
    
    if pde_loss_names is None and num_pde_losses is not None:
        pde_loss_names = [f'PDE-{i+1}' for i in range(num_pde_losses)]
    if bc_loss_names is None and num_bc_losses is not None:
        bc_loss_names = [f'BC-{i+1}' for i in range(num_bc_losses)]
        
    num_losses = loss_train.shape[1]
    
    def _format_loss_label(label):
        """
        格式化损失标签，将下划线后的部分转换为LaTeX下标
        例如: momentum_x -> $momentum_{x}$
        """
        if "$" in label:
            return label
        
        parts = label.split('_', 1)
        if len(parts) == 2:
            return f"${parts[0]}_{{{parts[1]}}}$"
        return label
    
    for i in range(num_losses):
        label = f'Loss {i+1}'
        if num_pde_losses is not None and i < num_pde_losses:
            if pde_loss_names and i < len(pde_loss_names):
                label = pde_loss_names[i]
        elif num_bc_losses is not None:
             bc_idx = i - (num_pde_losses if num_pde_losses else 0)
             if bc_loss_names and bc_idx < len(bc_loss_names):
                 label = bc_loss_names[bc_idx]

        # 格式化标签
        label = _format_loss_label(label)
        plt.plot(steps, loss_train[:, i], label=label)
        
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Loss Components')
    plt.legend(loc="upper right", frameon=True, fancybox=False, shadow=False)
    plt.grid(True)
    plt.yscale("log")
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.minorticks_off()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def save_model_checkpoint(model, save_dir, filename="model.ckpt"):
    """
    保存模型检查点
    """
    save_path = os.path.join(save_dir, filename)
    model.save(save_path)
    print(f"Model checkpoint saved to {save_path}")

def save_prediction_results(x_test, y_pred, y_true, save_dir, field_names=None):
    """
    保存预测结果
    """
    save_prediction_data(x_test, y_pred, y_true, None, save_dir, field_names=field_names)

def plot_and_save_solution(x, y_pred, save_dir, field_names=None, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0):
    """
    绘制并保存解
    """
    if field_names is None:
        field_names = [f'Field {i}' for i in range(y_pred.shape[1])]
        
    def _format_field_label(field_name):
        field_map = {
            "ux": r"$u_x$",
            "uy": r"$u_y$",
            "Sxx": r"$\sigma_{xx}$",
            "Syy": r"$\sigma_{yy}$",
            "Sxy": r"$\sigma_{xy}$",
        }
        return field_map.get(field_name, field_name)

    for i, name in enumerate(field_names):
        save_path = _get_save_path(save_dir, 'png', f'solution_{name}.png')
        label = _format_field_label(name)
        plt.figure(figsize=(8, 6))
        plt.scatter(x[:, 0], x[:, 1], c=y_pred[:, i], cmap='jet', s=5)
        plt.colorbar(label=label)
        plt.title(f'Predicted {label}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.savefig(save_path)
        plt.close()

def plot_and_save_error_map(x, y_pred, y_true, save_dir, field_names=None, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0):
    """
    绘制并保存误差图
    """
    error = np.abs(y_pred - y_true)
    if field_names is None:
        field_names = [f'Field {i}' for i in range(error.shape[1])]
        
    def _format_field_label(field_name):
        field_map = {
            "ux": r"$u_x$",
            "uy": r"$u_y$",
            "Sxx": r"$\sigma_{xx}$",
            "Syy": r"$\sigma_{yy}$",
            "Sxy": r"$\sigma_{xy}$",
        }
        return field_map.get(field_name, field_name)

    for i, name in enumerate(field_names):
        save_path = _get_save_path(save_dir, 'png', f'error_{name}.png')
        label = _format_field_label(name)
        plt.figure(figsize=(8, 6))
        plt.scatter(x[:, 0], x[:, 1], c=error[:, i], cmap='jet', s=5)
        plt.colorbar(label=f'Absolute Error {label}')
        plt.title(f'Error {label}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.savefig(save_path)
        plt.close()

def plot_comparison(x, y_pred, y_true, save_dir, field_names=None):
    """
    绘制对比图
    """
    plot_and_save_solution(x, y_pred, save_dir, field_names)
    # Ideally should plot true solution too, but keep it simple for now

def save_error_info(y_pred, y_true, save_dir, field_names=None):
    """
    保存误差信息
    """
    error = np.abs(y_pred - y_true)
    mae = np.mean(error, axis=0)
    mse = np.mean(error**2, axis=0)
    
    info = {
        "mae": mae.tolist(),
        "mse": mse.tolist()
    }
    
    save_path = _get_save_path(save_dir, 'json', 'error_info.json')
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=4)

def save_all_results(model, losshistory, train_state, save_dir, **kwargs):
    """
    保存所有结果的封装函数
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_loss_history(losshistory, save_dir)
    plot_and_save_loss_history(losshistory, save_dir)
    # More implementations can be added here

def plot_all_elasticity_fields_with_train(x_test, y_true_test, y_pred_test, x_train, y_true_train, y_pred_train, save_dir, field_names=None, nx=100, ny=100, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, slice_x=0.5):
    """
    绘制弹性力学场（位移和应力）
    """
    if field_names is None:
        field_names = ['ux', 'uy', 'Sxx', 'Syy', 'Sxy']
        
    def _format_field_label(field_name):
        field_map = {
            "ux": r"$u_x$",
            "uy": r"$u_y$",
            "Sxx": r"$\sigma_{xx}$",
            "Syy": r"$\sigma_{yy}$",
            "Sxy": r"$\sigma_{xy}$",
        }
        return field_map.get(field_name, field_name)

    def _dataset_label(dataset_tag):
        tag_map = {"训练集": "Train", "测试集": "Test"}
        return tag_map.get(dataset_tag, dataset_tag)

    def _grid_interpolate(points, values):
        from scipy.interpolate import griddata
        xi = np.linspace(x_min, x_max, nx)
        yi = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(xi, yi)
        Z = griddata(points, values, (X, Y), method='linear')
        if np.isnan(Z).any():
            Z_near = griddata(points, values, (X, Y), method='nearest')
            Z = np.where(np.isnan(Z), Z_near, Z)
        return X, Y, Z

    def _plot_field_comparison(points, y_true, y_pred, field_name, dataset_tag):
        y_true_flat = np.asarray(y_true).reshape(-1)
        y_pred_flat = np.asarray(y_pred).reshape(-1)
        X, Y, Z_true = _grid_interpolate(points, y_true_flat)
        _, _, Z_pred = _grid_interpolate(points, y_pred_flat)
        Z_err = np.abs(Z_true - Z_pred)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        label = _format_field_label(field_name)
        dataset_label = _dataset_label(dataset_tag)
        
        im0 = axes[0].contourf(X, Y, Z_true, levels=200, cmap='jet')
        axes[0].set_title(f'{label} True ({dataset_label})')
        axes[0].set_xlim(x_min, x_max)
        axes[0].set_ylim(y_min, y_max)
        axes[0].set_aspect('equal')
        fig.colorbar(im0, ax=axes[0])
        
        im1 = axes[1].contourf(X, Y, Z_pred, levels=200, cmap='jet')
        axes[1].set_title(f'{label} Pred ({dataset_label})')
        axes[1].set_xlim(x_min, x_max)
        axes[1].set_ylim(y_min, y_max)
        axes[1].set_aspect('equal')
        fig.colorbar(im1, ax=axes[1])
        
        im2 = axes[2].contourf(X, Y, Z_err, levels=200, cmap='magma')
        axes[2].set_title(f'{label} Abs Error ({dataset_label})')
        axes[2].set_xlim(x_min, x_max)
        axes[2].set_ylim(y_min, y_max)
        axes[2].set_aspect('equal')
        fig.colorbar(im2, ax=axes[2])
        
        save_path = _get_save_path(save_dir, 'png', f'场对比图_{dataset_tag}_{field_name}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def _plot_relative_error(points, y_true, y_pred, field_name, dataset_tag):
        y_true_flat = np.asarray(y_true).reshape(-1)
        y_pred_flat = np.asarray(y_pred).reshape(-1)
        X, Y, Z_true = _grid_interpolate(points, y_true_flat)
        _, _, Z_pred = _grid_interpolate(points, y_pred_flat)
        denom = np.where(np.abs(Z_true) > 1e-10, np.abs(Z_true), 1.0)
        z_rel = np.abs(Z_true - Z_pred) / denom
        
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        label = _format_field_label(field_name)
        dataset_label = _dataset_label(dataset_tag)
        
        im = ax.contourf(X, Y, z_rel, levels=200, cmap='viridis')
        ax.set_title(f'{label} Rel Error ({dataset_label})')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        fig.colorbar(im, ax=ax)
        
        save_path = _get_save_path(save_dir, 'png', f'相对误差图_{dataset_tag}_{field_name}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def _plot_slice(points, y_true, y_pred, dataset_tag):
        y_line = np.linspace(y_min, y_max, ny)
        n_fields = len(field_names)
        ncols = 2
        nrows = int(np.ceil(n_fields / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows), sharex=True)
        axes = np.array(axes).reshape(-1)

        xi = np.linspace(x_min, x_max, nx)
        idx = int(np.argmin(np.abs(xi - slice_x)))

        for i, field_name in enumerate(field_names):
            ax = axes[i]
            _, _, z_true = _grid_interpolate(points, y_true[:, i])
            _, _, z_pred = _grid_interpolate(points, y_pred[:, i])
            true_line = z_true[:, idx]
            pred_line = z_pred[:, idx]
            ax.plot(y_line, true_line, label='True')
            ax.plot(y_line, pred_line, label='Pred')
            ax.set_title(_format_field_label(field_name))
            ax.set_xlabel('y')
            ax.legend(loc="upper right", frameon=True, fancybox=False, shadow=False)

        for j in range(n_fields, len(axes)):
            axes[j].axis("off")

        save_path = _get_save_path(save_dir, 'png', f'切片对比图_{dataset_tag}_x{slice_x:.2f}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    if x_test is not None and y_pred_test is not None:
        for i, field_name in enumerate(field_names):
            _plot_field_comparison(x_test, y_true_test[:, i], y_pred_test[:, i], field_name, '测试集')
            _plot_relative_error(x_test, y_true_test[:, i], y_pred_test[:, i], field_name, '测试集')
        _plot_slice(x_test, y_true_test, y_pred_test, '测试集')

    if x_train is not None and y_pred_train is not None and y_true_train is not None:
        for i, field_name in enumerate(field_names):
            _plot_field_comparison(x_train, y_true_train[:, i], y_pred_train[:, i], field_name, '训练集')
            _plot_relative_error(x_train, y_true_train[:, i], y_pred_train[:, i], field_name, '训练集')
        _plot_slice(x_train, y_true_train, y_pred_train, '训练集')

def plot_deformed_mesh(x, ux, uy, save_dir, filename="变形网格对比图.png", scale_factor=1.0, nx=20, ny=20, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0):
    """
    绘制变形网格
    """
    save_path = _get_save_path(save_dir, 'png', filename)
    
    # 将散点插值为网格，以便绘制变形后的网格线
    from scipy.interpolate import griddata
    
    xi = np.linspace(x_min, x_max, nx)
    yi = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(xi, yi)
    
    UX = griddata(x, ux, (X, Y), method='cubic')
    UY = griddata(x, uy, (X, Y), method='cubic')
    
    # 变形后的坐标
    X_def = X + scale_factor * UX
    Y_def = Y + scale_factor * UY
    
    plt.figure(figsize=(10, 8))
    # 绘制原始网格（虚线）
    for i in range(ny):
        plt.plot(X[i, :], Y[i, :], 'k--', alpha=0.2, linewidth=0.5)
    for j in range(nx):
        plt.plot(X[:, j], Y[:, j], 'k--', alpha=0.2, linewidth=0.5)
        
    # 绘制变形后的网格
    for i in range(ny):
        plt.plot(X_def[i, :], Y_def[i, :], 'b-', alpha=0.6, linewidth=1.0)
    for j in range(nx):
        plt.plot(X_def[:, j], Y_def[:, j], 'b-', alpha=0.6, linewidth=1.0)
        
    plt.title(f'Deformed Mesh (Scale Factor: {scale_factor:.2f})', fontweight='bold')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Deformed mesh plot saved to {save_path}")

def plot_displacement_vector_field(x, ux, uy, save_dir, filename="位移矢量场图.png", nx=20, ny=20, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, scale=1.0):
    """
    绘制位移向量场
    """
    save_path = _get_save_path(save_dir, 'png', filename)
    
    from scipy.interpolate import griddata
    xi = np.linspace(x_min, x_max, nx)
    yi = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(xi, yi)
    
    UX = griddata(x, ux, (X, Y), method='cubic')
    UY = griddata(x, uy, (X, Y), method='cubic')
    
    plt.figure(figsize=(10, 8))
    plt.quiver(X, Y, UX, UY, color='r', scale=scale, scale_units='xy', angles='xy')
    plt.title('Displacement Vector Field', fontweight='bold')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Displacement vector field plot saved to {save_path}")

def _append_region_history(records, history_rows, step_offset=0, running_step=None):
    current_step = int(running_step if running_step is not None else step_offset)
    for row in history_rows:
        lambda_regions = row.get("lambda_regions")
        mu_regions = row.get("mu_regions")
        if not isinstance(lambda_regions, list) or not isinstance(mu_regions, list):
            continue
        if len(lambda_regions) == 0 or len(lambda_regions) != len(mu_regions):
            continue

        if row.get("global_step") is not None:
            step = int(row["global_step"])
            current_step = step
        elif row.get("step") is not None:
            step = int(step_offset) + int(row["step"])
            current_step = step
        elif row.get("chunk_iterations") is not None:
            current_step += int(row["chunk_iterations"])
            step = current_step
        else:
            continue

        records.append(
            {
                "step": step,
                "lambda_regions": [float(value) for value in lambda_regions],
                "mu_regions": [float(value) for value in mu_regions],
            }
        )
    return current_step


def _extract_true_region_constants(case_config, num_regions):
    if case_config is None or num_regions <= 0:
        return [], []

    lambda_truths = [float(case_config.lambda_bg)]
    mu_truths = [float(case_config.mu_bg)]

    if num_regions >= 2:
        lambda_truths.append(float(case_config.lambda_bg + case_config.lambda_ctr_1))
        mu_truths.append(float(case_config.mu_bg + case_config.mu_ctr_1))
    if num_regions >= 3:
        lambda_truths.append(float(case_config.lambda_bg + case_config.lambda_ctr_2))
        mu_truths.append(float(case_config.mu_bg + case_config.mu_ctr_2))

    return lambda_truths[:num_regions], mu_truths[:num_regions]


def plot_region_parameter_evolution(
    save_dir,
    case_config,
    total_iterations=None,
    final_region_payload=None,
    filename="region_parameter_evolution.png",
    json_filename="region_parameter_evolution.json",
):
    plan_path = _get_save_path(save_dir, "json", "stage_training_plan.json")
    geometry_path = _get_save_path(save_dir, "json", "geometry_stage_history.json")
    material_path = _get_save_path(save_dir, "json", "material_stage_history.json")
    adaptive_main_path = _get_save_path(save_dir, "json", "adaptive_main_stage_history.json")
    main_schedule_path = _get_save_path(save_dir, "json", "main_stage_schedule.json")
    output_json_path = _get_save_path(save_dir, "json", json_filename)
    output_png_path = _get_save_path(save_dir, "png", filename)

    plan = {}
    if os.path.isfile(plan_path):
        with open(plan_path, "r", encoding="utf-8") as handle:
            plan = json.load(handle)

    warmup_iterations = int(plan.get("warmup_iterations", 0))
    geometry_iterations = int(plan.get("geometry_stage_iterations", 0))
    material_iterations = int(plan.get("material_stage_iterations", 0))
    refinement_iterations = int(plan.get("refinement_stage_iterations", 0))
    if total_iterations is None:
        total_iterations = int(
            warmup_iterations
            + geometry_iterations
            + material_iterations
            + int(plan.get("main_iterations", 0))
            + refinement_iterations
        )

    records = []

    if os.path.isfile(geometry_path):
        with open(geometry_path, "r", encoding="utf-8") as handle:
            geometry_payload = json.load(handle)
        _append_region_history(records, geometry_payload.get("history", []), step_offset=warmup_iterations)

    material_offset = warmup_iterations + geometry_iterations
    if os.path.isfile(material_path):
        with open(material_path, "r", encoding="utf-8") as handle:
            material_payload = json.load(handle)
        _append_region_history(records, material_payload.get("history", []), step_offset=material_offset)

    main_offset = warmup_iterations + geometry_iterations + material_iterations
    if os.path.isfile(adaptive_main_path):
        with open(adaptive_main_path, "r", encoding="utf-8") as handle:
            adaptive_payload = json.load(handle)
        _append_region_history(records, adaptive_payload.get("chunks", []), step_offset=main_offset, running_step=main_offset)
    elif os.path.isfile(main_schedule_path):
        with open(main_schedule_path, "r", encoding="utf-8") as handle:
            main_payload = json.load(handle)
        _append_region_history(records, main_payload.get("history", []), step_offset=main_offset, running_step=main_offset)

    if final_region_payload is not None:
        lambda_regions = final_region_payload.get("lambda_regions")
        mu_regions = final_region_payload.get("mu_regions")
        if isinstance(lambda_regions, list) and isinstance(mu_regions, list) and len(lambda_regions) == len(mu_regions):
            records.append(
                {
                    "step": int(total_iterations),
                    "lambda_regions": [float(value) for value in lambda_regions],
                    "mu_regions": [float(value) for value in mu_regions],
                }
            )

    if not records:
        return None

    unique_records = {}
    for row in records:
        unique_records[int(row["step"])] = row
    records = [unique_records[key] for key in sorted(unique_records)]

    num_regions = len(records[-1]["lambda_regions"])
    lambda_truths, mu_truths = _extract_true_region_constants(case_config, num_regions)
    steps = [int(row["step"]) for row in records]
    lambda_matrix = np.asarray([row["lambda_regions"] for row in records], dtype=float)
    mu_matrix = np.asarray([row["mu_regions"] for row in records], dtype=float)

    payload = {
        "steps": steps,
        "lambda_regions": lambda_matrix.tolist(),
        "mu_regions": mu_matrix.tolist(),
        "lambda_truths": lambda_truths,
        "mu_truths": mu_truths,
    }
    with open(output_json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4", "#ff7f0e", "#2ca02c"])

    for axis, matrix, truths, field_name in zip(
        axes,
        [lambda_matrix, mu_matrix],
        [lambda_truths, mu_truths],
        ["Lambda", "Mu"],
    ):
        for index in range(matrix.shape[1]):
            color = colors[index % len(colors)]
            axis.plot(steps, matrix[:, index], color=color, linewidth=2.0, label=f"pred_r{index}")
            if index < len(truths):
                axis.axhline(truths[index], color=color, linestyle="--", linewidth=1.6, alpha=0.85, label=f"true_r{index}")
        axis.set_title(f"{field_name} Regions vs Step")
        axis.set_xlabel("Step")
        axis.set_ylabel(field_name)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8, frameon=False)

    plt.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    plt.savefig(output_png_path, dpi=300)
    plt.close(fig)
    return payload


def plot_parameter_history(param_data_or_file, true_values, param_names, save_path):
    """
    绘制参数反演历史
    Args:
        param_data_or_file: 参数数据文件路径(str) 或 参数数据数组(numpy array / list)
        true_values: 真实参数值列表
        param_names: 参数名称列表
        save_path: 图片保存路径
    """
    try:
        if isinstance(param_data_or_file, str):
            # 自定义读取逻辑，处理 DeepXDE VariableValue 可能产生的格式（如包含 [] 或 ,）
            data_list = []
            with open(param_data_or_file, 'r') as f:
                for line in f:
                    # 去除首尾空白
                    line = line.strip()
                    if not line: continue
                    # 替换掉可能导致解析错误的字符
                    clean_line = line.replace('[', '').replace(']', '').replace(',', ' ')
                    # 分割并转换为浮点数
                    try:
                        row = [float(x) for x in clean_line.split()]
                        data_list.append(row)
                    except ValueError:
                        continue
            data = np.array(data_list)
        else:
            data = np.array(param_data_or_file)
            
        if data.ndim == 1:
            data = data[:, None]
            
        # Assuming data structure: step, param1, param2, ...
        # If read from VariableValue file (text), it usually doesn't have step unless formatted
        # DeepXDE VariableValue outputs: [var1, var2, ...] in each line. No step index by default?
        # Let's check VariableValue implementation or output. 
        # DeepXDE VariableValue writes: file.write(step + " " + var1 + " " + var2 ...) ? 
        # No, VariableValue source: self.file.write(str(self.model.train_state.step)) ...
        # So first column IS step.
        
        steps = data[:, 0]
        params = data[:, 1:]
        
        plt.figure(figsize=(10, 6))
        def _format_param_name(name):
            name_str = str(name)
            lowered = name_str.strip().lower()
            if lowered in {"lambda", "lmbd", "lmbda"}:
                return r"$\lambda$"
            if lowered == "mu":
                return r"$\mu$"
            return name_str

        for i, name in enumerate(param_names):
            if i < params.shape[1]:
                sym = _format_param_name(name)
                color = f"C{i}"
                plt.plot(steps, params[:, i], label=f'Predicted {sym}', color=color)
                if true_values and i < len(true_values):
                    plt.axhline(y=true_values[i], linestyle='--', color=color, label=f'True {sym}')
                
        plt.xlabel('Steps')
        plt.ylabel('Value')
        plt.title('Parameter Estimation History')
        plt.legend(loc="upper right", frameon=True, fancybox=False, shadow=False)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
    except Exception as e:
        print(f"Failed to plot parameter history: {e}")

def plot_point_distribution(save_dir, geom, num_domain, num_boundary, num_test, observe_x=None, filename="point_distribution.png"):
    """
    可视化训练点（域内点、边界点、观测点）和测试点的分布 - 改进版（顶级期刊风格）
    """
    save_path = _get_save_path(save_dir, 'png', filename)
    
    # 使用上下文管理器临时修改样式，避免影响其他绘图
    with plt.rc_context({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'lines.linewidth': 2,
        'figure.dpi': 300
    }):
        # 生成点
        domain_points = geom.random_points(num_domain)
        boundary_points = geom.random_boundary_points(num_boundary)
        test_points = geom.random_points(num_test)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 定义颜色方案 (Color Palette)
        c_domain = '#1f77b4'  # Muted Blue
        c_boundary = '#2ca02c'  # Cooked Asparagus Green
        c_observe = '#d62728'  # Brick Red
        c_test = '#ff7f0e'    # Safety Orange
        
        # 子图1：训练点
        ax1 = axes[0]
        # 域内点
        ax1.scatter(domain_points[:, 0], domain_points[:, 1], 
                    c=c_domain, s=15, alpha=0.5, label='Domain', edgecolors='none')
        # 边界点 - 改用圆点，不使用叉号，并设置 clip_on=False 避免被坐标轴裁剪
        ax1.scatter(boundary_points[:, 0], boundary_points[:, 1], 
                    c=c_boundary, s=20, alpha=0.9, marker='o', label='Boundary', edgecolors='none', clip_on=False)
        # 观测点
        if observe_x is not None:
            ax1.scatter(observe_x[:, 0], observe_x[:, 1], 
                        c=c_observe, s=50, alpha=1.0, marker='*', label='Observation', edgecolors='k', linewidth=0.5, clip_on=False)
            
        ax1.set_title("Training Data", fontweight='bold', pad=15)
        ax1.set_xlabel("$x$")
        ax1.set_ylabel("$y$")
        
        ax1.legend(loc='upper right', frameon=True, fancybox=False, shadow=False)
        
        ax1.set_aspect('equal')
        # 严格限制在 [0, 1]
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        # 去掉网格线
        ax1.grid(False)
        
        # 子图2：测试点
        ax2 = axes[1]
        alpha_test = 0.6 if num_test < 2000 else 0.3
        s_test = 15 if num_test < 2000 else 5
        ax2.scatter(test_points[:, 0], test_points[:, 1], 
                    c=c_test, s=s_test, alpha=alpha_test, label='Test', edgecolors='none', clip_on=False)
        
        ax2.set_title("Test Data", fontweight='bold', pad=15)
        ax2.set_xlabel("$x$")
        ax2.set_ylabel("$y$")
        ax2.legend(loc='upper right', frameon=True, fancybox=False, shadow=False)
        
        ax2.set_aspect('equal')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.grid(False)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
    print(f"点分布可视化已保存 (Journal Style): {save_path}")
    return save_path
