import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from .data_saving_utils import _get_save_path


def reshape_grid(values, ny, nx):
    return np.asarray(values, dtype=float).reshape(ny, nx)


def _compute_levels(vmin, vmax, count=101):
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = -1.0, 1.0
    if np.isclose(vmin, vmax):
        delta = 1.0 if np.isclose(vmin, 0.0) else max(abs(vmin) * 1e-6, 1e-8)
        vmin -= delta
        vmax += delta
    return float(vmin), float(vmax), np.linspace(float(vmin), float(vmax), int(count))


def _configure_plain_colorbar(colorbar, vmin, vmax):
    vmin = float(vmin)
    vmax = float(vmax)
    span = abs(vmax - vmin)
    scale = max(abs(vmin), abs(vmax), 1e-12)

    if span <= scale * 1e-4:
        ticks = np.linspace(vmin, vmax, 5)
        colorbar.set_ticks(ticks)

    if span <= 0 or not np.isfinite(span):
        decimals = 6
    else:
        decimals = max(3, min(8, int(np.ceil(-np.log10(span))) + 2))

    formatter = FuncFormatter(lambda value, _: f'{value:.{decimals}f}')
    colorbar.formatter = formatter
    colorbar.ax.yaxis.set_major_formatter(formatter)
    colorbar.ax.yaxis.get_offset_text().set_visible(False)
    colorbar.update_ticks()
    colorbar.ax.yaxis.get_offset_text().set_visible(False)


def _add_colorbar(fig, mappable, axis, vmin, vmax):
    colorbar = fig.colorbar(mappable, ax=axis)
    _configure_plain_colorbar(colorbar, vmin, vmax)
    return colorbar


def plot_field_triplet(save_dir, xx, yy, truth_grid, pred_grid, field_name):
    truth_grid = np.asarray(truth_grid, dtype=float)
    pred_grid = np.asarray(pred_grid, dtype=float)
    abs_error = np.abs(pred_grid - truth_grid)

    combined_min = float(np.nanmin([np.nanmin(truth_grid), np.nanmin(pred_grid)]))
    combined_max = float(np.nanmax([np.nanmax(truth_grid), np.nanmax(pred_grid)]))
    field_vmin, field_vmax, field_levels = _compute_levels(combined_min, combined_max)

    err_max = float(np.nanmax(abs_error)) if abs_error.size else 0.0
    _, err_vmax, err_levels = _compute_levels(0.0, err_max)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.5), constrained_layout=True)

    im0 = axes[0].contourf(xx, yy, truth_grid, levels=field_levels, cmap='viridis', vmin=field_vmin, vmax=field_vmax)
    axes[0].set_title(f'True {field_name}')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_aspect('equal')
    _add_colorbar(fig, im0, axes[0], field_vmin, field_vmax)

    im1 = axes[1].contourf(xx, yy, pred_grid, levels=field_levels, cmap='viridis', vmin=field_vmin, vmax=field_vmax)
    axes[1].set_title(f'Predicted {field_name}')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_aspect('equal')
    _add_colorbar(fig, im1, axes[1], field_vmin, field_vmax)

    im2 = axes[2].contourf(xx, yy, abs_error, levels=err_levels, cmap='magma', vmin=0.0, vmax=err_vmax)
    axes[2].set_title(f'Absolute error of {field_name}')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_aspect('equal')
    _add_colorbar(fig, im2, axes[2], 0.0, err_vmax)

    plt.savefig(_get_save_path(save_dir, 'png', f'{field_name}_comparison.png'), dpi=300)
    plt.close(fig)


def plot_material_overlay(save_dir, xx, yy, truth_grid, pred_grid, field_name):
    truth_grid = np.asarray(truth_grid, dtype=float)
    pred_grid = np.asarray(pred_grid, dtype=float)
    combined_min = float(np.nanmin([np.nanmin(truth_grid), np.nanmin(pred_grid)]))
    combined_max = float(np.nanmax([np.nanmax(truth_grid), np.nanmax(pred_grid)]))
    vmin, vmax, levels = _compute_levels(combined_min, combined_max, count=24)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for axis, grid, title in zip(
        axes,
        [truth_grid, pred_grid],
        [f'True {field_name}', f'Predicted {field_name}'],
    ):
        image = axis.contourf(xx, yy, grid, levels=levels, cmap='viridis', vmin=vmin, vmax=vmax)
        axis.contour(xx, yy, grid, levels=levels[::3], colors='white', linewidths=0.5)
        axis.set_title(title)
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.set_aspect('equal')
        _add_colorbar(fig, image, axis, vmin, vmax)
    plt.tight_layout()
    plt.savefig(_get_save_path(save_dir, 'png', f'{field_name}_material_map.png'), dpi=300)
    plt.close(fig)


def plot_observation_fit(save_dir, observation_truth, observation_prediction, filename='observation_fit.png'):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, index, label in zip(axes, [0, 1], ['ux', 'uy']):
        axis.scatter(
            observation_truth[:, index],
            observation_prediction[:, index],
            s=16,
            alpha=0.75,
            color='#355F94',
        )
        limits = [
            float(min(np.min(observation_truth[:, index]), np.min(observation_prediction[:, index]))),
            float(max(np.max(observation_truth[:, index]), np.max(observation_prediction[:, index]))),
        ]
        axis.plot(limits, limits, linestyle='--', color='#d97706', linewidth=1.2)
        axis.set_xlabel(f'Observed {label}')
        axis.set_ylabel(f'Predicted {label}')
        axis.set_title(f'Observation fit for {label}')
        axis.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(_get_save_path(save_dir, 'png', filename), dpi=300)
    plt.close(fig)


def plot_interface_diagnostics(save_dir, xx, yy, interface_indicator_grid, class_probability_grid, split_name):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    for axis, grid, title in zip(
        axes,
        [interface_indicator_grid, class_probability_grid],
        ['Interface indicator', 'Region probability'],
    ):
        vmin, vmax, levels = _compute_levels(np.nanmin(grid), np.nanmax(grid))
        image = axis.contourf(xx, yy, grid, levels=levels, cmap='coolwarm', vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.set_aspect('equal')
        _add_colorbar(fig, image, axis, vmin, vmax)
    plt.tight_layout()
    plt.savefig(_get_save_path(save_dir, 'png', f'{split_name}_interface_diagnostics.png'), dpi=300)
    plt.close(fig)


def plot_validation_history(history, save_dir, filename='validation_history.png'):
    if not history.get('steps'):
        return
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(history['steps'], history['validation_observation_mse'], color='#0f766e', linewidth=2.0)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Validation observation MSE')
    ax.set_title('Validation observation history')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(_get_save_path(save_dir, 'png', filename), dpi=300)
    plt.close(fig)


def plot_case_sampling_layout(
    save_dir,
    xx,
    yy,
    lambda_field,
    mu_field,
    domain_points,
    boundary_points,
    train_observation_points,
    val_observation_points,
    eval_observation_points,
    title_suffix,
    filename='sampling_and_case.png',
):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for axis, field, label in zip(
        axes[:2],
        [lambda_field, mu_field],
        [r'True $\lambda(x,y)$', r'True $\mu(x,y)$'],
    ):
        vmin, vmax, levels = _compute_levels(np.nanmin(field), np.nanmax(field))
        image = axis.contourf(xx, yy, field, levels=levels, cmap='viridis', vmin=vmin, vmax=vmax)
        axis.set_title(label)
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.set_aspect('equal')
        _add_colorbar(fig, image, axis, vmin, vmax)

    axes[2].scatter(domain_points[:, 0], domain_points[:, 1], s=8, alpha=0.25, label='Domain points')
    if len(boundary_points) > 0:
        axes[2].scatter(boundary_points[:, 0], boundary_points[:, 1], s=22, alpha=0.9, color='#111827', marker='x', label='Boundary points')
    if len(train_observation_points) > 0:
        axes[2].scatter(train_observation_points[:, 0], train_observation_points[:, 1], s=18, alpha=0.85, color='#d97706', label='Train observations')
    if len(val_observation_points) > 0:
        axes[2].scatter(val_observation_points[:, 0], val_observation_points[:, 1], s=22, alpha=0.85, color='#0f766e', marker='^', label='Validation observations')
    if len(eval_observation_points) > 0:
        axes[2].scatter(eval_observation_points[:, 0], eval_observation_points[:, 1], s=22, alpha=0.85, color='#7c3aed', marker='s', label='Evaluation observations')
    axes[2].set_title(f'Sampling layout ({title_suffix})')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_xlim(0.0, 1.0)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_aspect('equal')
    axes[2].legend(frameon=False, loc='upper right')
    plt.tight_layout()
    plt.savefig(_get_save_path(save_dir, 'png', filename), dpi=300)
    plt.close(fig)
