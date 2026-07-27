import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from .data_saving_utils import _get_save_path


_UNIT_DOMAIN_TICKS = np.linspace(0.0, 1.0, 5)
_SAMPLING_BULK_LIMITS = (1.3, 3.0)
_SAMPLING_SHEAR_LIMITS = (0.45, 1.35)


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


def _limits_or_none(limits):
    if limits is None:
        return None
    if len(limits) != 2:
        raise ValueError("limits must be a (vmin, vmax) pair.")
    return float(limits[0]), float(limits[1])


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

    def format_tick(value, _):
        rounded = round(float(value), decimals)
        if rounded == 0:
            rounded = 0.0
        return f'{rounded:.{decimals}f}'

    formatter = FuncFormatter(format_tick)
    colorbar.formatter = formatter
    colorbar.ax.yaxis.set_major_formatter(formatter)
    colorbar.ax.yaxis.get_offset_text().set_visible(False)
    colorbar.update_ticks()
    colorbar.ax.yaxis.get_offset_text().set_visible(False)


def _format_three_decimals(value, _):
    rounded = round(float(value), 3)
    if rounded == 0:
        rounded = 0.0
    return f'{rounded:.3f}'


def _configure_sampling_colorbar(colorbar, vmin, vmax):
    colorbar.set_ticks(np.linspace(float(vmin), float(vmax), 5))
    formatter = FuncFormatter(_format_three_decimals)
    colorbar.ax.yaxis.set_major_formatter(formatter)
    colorbar.formatter = formatter
    colorbar.ax.yaxis.get_offset_text().set_visible(False)
    colorbar.update_ticks()
    colorbar.ax.yaxis.get_offset_text().set_visible(False)


def _format_unit_domain_axis(axis):
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(_UNIT_DOMAIN_TICKS)
    axis.set_yticks(_UNIT_DOMAIN_TICKS)
    formatter = FuncFormatter(_format_three_decimals)
    axis.xaxis.set_major_formatter(formatter)
    axis.yaxis.set_major_formatter(formatter)


def _sampling_material_limits(label):
    label_text = str(label).lower()
    if "bulk" in label_text or "modulus k" in label_text or "$k" in label_text:
        return _SAMPLING_BULK_LIMITS
    if "shear" in label_text or "mu" in label_text or r"\mu" in label_text:
        return _SAMPLING_SHEAR_LIMITS
    return None


def _add_colorbar(fig, mappable, axis, vmin, vmax):
    colorbar = fig.colorbar(mappable, ax=axis)
    _configure_plain_colorbar(colorbar, vmin, vmax)
    return colorbar


def plot_field_triplet(
    save_dir,
    xx,
    yy,
    truth_grid,
    pred_grid,
    field_name,
    title_name=None,
    file_stem=None,
    shared_scale=True,
    value_limits=None,
    error_limits=None,
):
    truth_grid = np.asarray(truth_grid, dtype=float)
    pred_grid = np.asarray(pred_grid, dtype=float)
    abs_error = np.abs(pred_grid - truth_grid)
    title_name = title_name or field_name
    file_stem = file_stem or field_name

    truth_min = float(np.nanmin(truth_grid))
    truth_max = float(np.nanmax(truth_grid))
    pred_min = float(np.nanmin(pred_grid))
    pred_max = float(np.nanmax(pred_grid))
    truth_span = abs(truth_max - truth_min)
    pred_span = abs(pred_max - pred_min)

    use_shared_scale = bool(shared_scale)
    if isinstance(shared_scale, str) and shared_scale.lower() == "auto":
        nonzero_spans = [span for span in [truth_span, pred_span] if span > 1e-12]
        if len(nonzero_spans) < 2:
            use_shared_scale = True
        else:
            span_ratio = max(nonzero_spans) / min(nonzero_spans)
            use_shared_scale = span_ratio <= 20.0

    value_limits = _limits_or_none(value_limits)
    error_limits = _limits_or_none(error_limits)

    if value_limits is not None:
        truth_vmin, truth_vmax, truth_levels = _compute_levels(value_limits[0], value_limits[1])
        pred_vmin, pred_vmax, pred_levels = truth_vmin, truth_vmax, truth_levels
    elif use_shared_scale:
        combined_min = float(np.nanmin([truth_min, pred_min]))
        combined_max = float(np.nanmax([truth_max, pred_max]))
        truth_vmin, truth_vmax, truth_levels = _compute_levels(combined_min, combined_max)
        pred_vmin, pred_vmax, pred_levels = truth_vmin, truth_vmax, truth_levels
    else:
        truth_vmin, truth_vmax, truth_levels = _compute_levels(truth_min, truth_max)
        pred_vmin, pred_vmax, pred_levels = _compute_levels(pred_min, pred_max)

    if error_limits is not None:
        err_vmin, err_vmax, err_levels = _compute_levels(error_limits[0], error_limits[1])
    else:
        err_max = float(np.nanmax(abs_error)) if abs_error.size else 0.0
        err_vmin, err_vmax, err_levels = _compute_levels(0.0, err_max)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.5), constrained_layout=True)

    im0 = axes[0].contourf(xx, yy, truth_grid, levels=truth_levels, cmap='viridis', vmin=truth_vmin, vmax=truth_vmax)
    axes[0].set_title(f'True {title_name}')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_aspect('equal')
    _format_unit_domain_axis(axes[0])
    _add_colorbar(fig, im0, axes[0], truth_vmin, truth_vmax)

    im1 = axes[1].contourf(xx, yy, pred_grid, levels=pred_levels, cmap='viridis', vmin=pred_vmin, vmax=pred_vmax)
    axes[1].set_title(f'Predicted {title_name}')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_aspect('equal')
    _format_unit_domain_axis(axes[1])
    _add_colorbar(fig, im1, axes[1], pred_vmin, pred_vmax)

    im2 = axes[2].contourf(xx, yy, abs_error, levels=err_levels, cmap='magma', vmin=err_vmin, vmax=err_vmax)
    axes[2].set_title(f'Absolute error of {title_name}')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_aspect('equal')
    _format_unit_domain_axis(axes[2])
    _add_colorbar(fig, im2, axes[2], err_vmin, err_vmax)

    plt.savefig(_get_save_path(save_dir, 'png', f'{file_stem}_comparison.png'), dpi=300)
    plt.close(fig)


def plot_material_overlay(save_dir, xx, yy, truth_grid, pred_grid, field_name, title_name=None, file_stem=None, value_limits=None):
    truth_grid = np.asarray(truth_grid, dtype=float)
    pred_grid = np.asarray(pred_grid, dtype=float)
    title_name = title_name or field_name
    file_stem = file_stem or field_name
    value_limits = _limits_or_none(value_limits)
    if value_limits is not None:
        combined_min, combined_max = value_limits
    else:
        combined_min = float(np.nanmin([np.nanmin(truth_grid), np.nanmin(pred_grid)]))
        combined_max = float(np.nanmax([np.nanmax(truth_grid), np.nanmax(pred_grid)]))
    vmin, vmax, levels = _compute_levels(combined_min, combined_max, count=24)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for axis, grid, title in zip(
        axes,
        [truth_grid, pred_grid],
        [f'True {title_name}', f'Predicted {title_name}'],
    ):
        image = axis.contourf(xx, yy, grid, levels=levels, cmap='viridis', vmin=vmin, vmax=vmax)
        axis.contour(xx, yy, grid, levels=levels[::3], colors='white', linewidths=0.5)
        axis.set_title(title)
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.set_aspect('equal')
        _format_unit_domain_axis(axis)
        _add_colorbar(fig, image, axis, vmin, vmax)
    plt.tight_layout()
    plt.savefig(_get_save_path(save_dir, 'png', f'{file_stem}_material_map.png'), dpi=300)
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
        _format_unit_domain_axis(axis)
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
    domain_points,
    boundary_points,
    train_observation_points,
    val_observation_points,
    eval_observation_points,
    title_suffix,
    filename='sampling_and_case.png',
    lambda_field=None,
    mu_field=None,
    material_field_a=None,
    material_field_b=None,
    material_label_a=None,
    material_label_b=None,
):
    if material_field_a is None:
        material_field_a = lambda_field
    if material_field_b is None:
        material_field_b = mu_field
    material_label_a = material_label_a or r'True $\lambda(x,y)$'
    material_label_b = material_label_b or r'True $\mu(x,y)$'

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 4.5))
    for axis, field, label in zip(
        axes[:2],
        [material_field_a, material_field_b],
        [material_label_a, material_label_b],
    ):
        material_limits = _sampling_material_limits(label)
        if material_limits is None:
            vmin, vmax, levels = _compute_levels(np.nanmin(field), np.nanmax(field))
        else:
            vmin, vmax, levels = _compute_levels(material_limits[0], material_limits[1])
        image = axis.contourf(xx, yy, field, levels=levels, cmap='viridis', vmin=vmin, vmax=vmax)
        axis.set_title(label)
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.set_aspect('equal')
        colorbar = _add_colorbar(fig, image, axis, vmin, vmax)
        _configure_sampling_colorbar(colorbar, vmin, vmax)
        _format_unit_domain_axis(axis)

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
    _format_unit_domain_axis(axes[2])
    axes[2].set_aspect('equal')
    axes[2].legend(frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    plt.tight_layout(rect=(0.0, 0.0, 0.90, 1.0))
    plt.savefig(_get_save_path(save_dir, 'png', filename), dpi=300)
    plt.close(fig)


def plot_material_probe_evolution(save_dir, history_payload, filename='material_parameter_evolution.png'):
    if not history_payload or not history_payload.get('history'):
        return

    parameterization = str(history_payload.get('parameterization', 'bulkmu')).lower()
    field_specs = [('bulk', 'Bulk modulus K'), ('mu', 'Shear modulus Mu')] if parameterization == 'bulkmu' else [('lambda', 'Lambda'), ('mu', 'Shear modulus Mu')]
    history_rows = history_payload['history']
    snapshots = history_payload.get('snapshots', {})
    probe_names = list(history_payload.get('truth', {}).keys())
    if not probe_names:
        return

    steps = [int(row['step']) for row in history_rows]
    fig, axes = plt.subplots(1, len(field_specs), figsize=(6.4 * len(field_specs), 4.8))
    if len(field_specs) == 1:
        axes = [axes]

    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['#1f77b4', '#ff7f0e', '#2ca02c'])
    snapshot_markers = {
        'Adam_best': 'P',
        'selected_best': '*',
    }
    for axis, (field_key, field_title) in zip(axes, field_specs):
        for index, probe_name in enumerate(probe_names):
            color = colors[index % len(colors)]
            predictions = [float(row.get('probes', {}).get(probe_name, {}).get(field_key, np.nan)) for row in history_rows]
            truth_value = float(history_payload['truth'][probe_name][field_key])
            axis.plot(steps, predictions, color=color, linewidth=2.0, label=f'Pred {probe_name}')
            axis.axhline(truth_value, color=color, linestyle='--', linewidth=1.6, alpha=0.9, label=f'True {probe_name}')
            for snapshot_name, snapshot_row in snapshots.items():
                snapshot_value = float(snapshot_row.get('probes', {}).get(probe_name, {}).get(field_key, np.nan))
                snapshot_step = int(snapshot_row.get('step', steps[-1]))
                axis.scatter(
                    [snapshot_step],
                    [snapshot_value],
                    marker=snapshot_markers.get(str(snapshot_name), 'X'),
                    s=110,
                    color=color,
                    edgecolors='black',
                    linewidths=0.6,
                    zorder=5,
                    label=f'{snapshot_name} {probe_name}',
                )
        axis.set_title(f'{field_title} vs Step')
        axis.set_xlabel('Step')
        axis.set_ylabel(field_title)
        axis.grid(True, alpha=0.25)
        axis.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=3, fontsize=8, frameon=False)

    plt.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    plt.savefig(_get_save_path(save_dir, 'png', filename), dpi=300)
    plt.close(fig)


def plot_k_mu_truth_prediction_comparison(
    save_dir,
    xx,
    yy,
    bulk_truth_grid,
    bulk_pred_grid,
    mu_truth_grid,
    mu_pred_grid,
    filename='K与μ真值预测对比图.png',
    bulk_value_limits=None,
    mu_value_limits=None,
    bulk_error_limits=None,
    mu_error_limits=None,
):
    bulk_truth_grid = np.asarray(bulk_truth_grid, dtype=float)
    bulk_pred_grid = np.asarray(bulk_pred_grid, dtype=float)
    mu_truth_grid = np.asarray(mu_truth_grid, dtype=float)
    mu_pred_grid = np.asarray(mu_pred_grid, dtype=float)
    bulk_error = np.abs(bulk_pred_grid - bulk_truth_grid)
    mu_error = np.abs(mu_pred_grid - mu_truth_grid)

    bulk_value_limits = _limits_or_none(bulk_value_limits)
    mu_value_limits = _limits_or_none(mu_value_limits)
    bulk_error_limits = _limits_or_none(bulk_error_limits)
    mu_error_limits = _limits_or_none(mu_error_limits)

    if bulk_value_limits is None:
        bulk_value_limits = (
            float(np.nanmin([np.nanmin(bulk_truth_grid), np.nanmin(bulk_pred_grid)])),
            float(np.nanmax([np.nanmax(bulk_truth_grid), np.nanmax(bulk_pred_grid)])),
        )
    if mu_value_limits is None:
        mu_value_limits = (
            float(np.nanmin([np.nanmin(mu_truth_grid), np.nanmin(mu_pred_grid)])),
            float(np.nanmax([np.nanmax(mu_truth_grid), np.nanmax(mu_pred_grid)])),
        )
    if bulk_error_limits is None:
        bulk_error_limits = (0.0, float(np.nanmax(bulk_error)))
    if mu_error_limits is None:
        mu_error_limits = (0.0, float(np.nanmax(mu_error)))

    bulk_vmin, bulk_vmax, bulk_levels = _compute_levels(*bulk_value_limits)
    mu_vmin, mu_vmax, mu_levels = _compute_levels(*mu_value_limits)
    bulk_err_vmin, bulk_err_vmax, bulk_err_levels = _compute_levels(*bulk_error_limits)
    mu_err_vmin, mu_err_vmax, mu_err_levels = _compute_levels(*mu_error_limits)

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.5), constrained_layout=True)
    plot_items = [
        (axes[0, 0], bulk_truth_grid, bulk_levels, 'viridis', bulk_vmin, bulk_vmax, 'True Bulk modulus K'),
        (axes[0, 1], bulk_pred_grid, bulk_levels, 'viridis', bulk_vmin, bulk_vmax, 'Predicted Bulk modulus K'),
        (axes[0, 2], bulk_error, bulk_err_levels, 'magma', bulk_err_vmin, bulk_err_vmax, 'Absolute error of Bulk modulus K'),
        (axes[1, 0], mu_truth_grid, mu_levels, 'viridis', mu_vmin, mu_vmax, 'True Shear modulus Mu'),
        (axes[1, 1], mu_pred_grid, mu_levels, 'viridis', mu_vmin, mu_vmax, 'Predicted Shear modulus Mu'),
        (axes[1, 2], mu_error, mu_err_levels, 'magma', mu_err_vmin, mu_err_vmax, 'Absolute error of Shear modulus Mu'),
    ]

    for axis, grid, levels, cmap, vmin, vmax, title in plot_items:
        image = axis.contourf(xx, yy, grid, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.set_aspect('equal')
        _format_unit_domain_axis(axis)
        _add_colorbar(fig, image, axis, vmin, vmax)

    plt.savefig(_get_save_path(save_dir, 'png', filename), dpi=300)
    plt.close(fig)
