import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_summary(exp_dir):
    return load_json(os.path.join(exp_dir, 'json', 'train_summary.json'))


def load_loss_history(exp_dir):
    return load_json(os.path.join(exp_dir, 'json', 'loss_history.json'))


def load_eval_grid(exp_dir):
    return np.load(os.path.join(exp_dir, 'npz', 'evaluation_grid.npz'))


def plot_metric_comparison(case_name, baseline_name, method_name, baseline_summary, method_summary, output_dir):
    metric_names = ['lambda', 'mu', 'ux', 'uy', 'sxx', 'syy', 'sxy']
    baseline_values = [baseline_summary['post_train_metrics']['field_relative_l2'][name] for name in metric_names]
    method_values = [method_summary['post_train_metrics']['field_relative_l2'][name] for name in metric_names]

    x = np.arange(len(metric_names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(x - width / 2, baseline_values, width, label=baseline_name, color='#355F94')
    ax.bar(x + width / 2, method_values, width, label=method_name, color='#D2A071')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel('Relative L2 error')
    ax.set_xlabel('Field')
    ax.set_title(f'{case_name}: method comparison by field')
    ax.legend(frameon=False)
    ax.grid(True, axis='y', alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{case_name}_method_comparison_metrics.png'), dpi=300)
    plt.close(fig)

    extra_names = ['observation_mse', 'pde_residual_mean_abs', 'material_field_mean_abs_vector_error']
    baseline_extra = [baseline_summary['post_train_metrics'][name] for name in extra_names]
    method_extra = [method_summary['post_train_metrics'][name] for name in extra_names]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    x = np.arange(len(extra_names))
    ax.bar(x - width / 2, baseline_extra, width, label=baseline_name, color='#355F94')
    ax.bar(x + width / 2, method_extra, width, label=method_name, color='#D2A071')
    ax.set_xticks(x)
    ax.set_xticklabels(['Obs MSE', 'PDE residual', 'Material MAE'])
    ax.set_ylabel('Value')
    ax.set_title(f'{case_name}: inverse-task summary metrics')
    ax.legend(frameon=False)
    ax.grid(True, axis='y', alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{case_name}_method_comparison_inverse_metrics.png'), dpi=300)
    plt.close(fig)


def plot_loss_comparison(case_name, baseline_name, method_name, baseline_history, method_history, output_dir):
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for history, label, color in [
        (baseline_history, baseline_name, '#355F94'),
        (method_history, method_name, '#D2A071'),
    ]:
        steps = np.asarray(history.get('steps', []), dtype=float)
        loss_train = np.asarray(history.get('loss_train', []), dtype=float)
        if len(steps) == 0 or loss_train.size == 0:
            continue
        ax.plot(steps, np.sum(loss_train, axis=1), label=f'{label} train', color=color, linewidth=2)
    ax.set_yscale('log')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Total train loss')
    ax.set_title(f'{case_name}: train loss comparison')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{case_name}_method_comparison_loss.png'), dpi=300)
    plt.close(fig)


def plot_material_maps(case_name, baseline_name, method_name, baseline_grid, method_grid, output_dir):
    xx = baseline_grid['xx']
    yy = baseline_grid['yy']
    truth = baseline_grid['truth']
    pred_baseline = baseline_grid['prediction']
    pred_method = method_grid['prediction']
    ny, nx = xx.shape

    panels = [
        ('True lambda', truth[:, 5].reshape(ny, nx)),
        (f'{baseline_name} lambda', pred_baseline[:, 5].reshape(ny, nx)),
        (f'{method_name} lambda', pred_method[:, 5].reshape(ny, nx)),
        ('True mu', truth[:, 6].reshape(ny, nx)),
        (f'{baseline_name} mu', pred_baseline[:, 6].reshape(ny, nx)),
        (f'{method_name} mu', pred_method[:, 6].reshape(ny, nx)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))
    for axis, (title, grid) in zip(axes.reshape(-1), panels):
        image = axis.contourf(xx, yy, grid, levels=100, cmap='viridis')
        axis.set_title(title)
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.set_aspect('equal')
        plt.colorbar(image, ax=axis)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{case_name}_method_comparison_material_maps.png'), dpi=300)
    plt.close(fig)


def save_summary_json(case_name, baseline_name, method_name, baseline_summary, method_summary, output_dir):
    payload = {
        'case': case_name,
        'baseline_name': baseline_name,
        'method_name': method_name,
        'baseline_metrics': baseline_summary['post_train_metrics'],
        'method_metrics': method_summary['post_train_metrics'],
        'parameter_count': {
            baseline_name: baseline_summary['parameter_count'],
            method_name: method_summary['parameter_count'],
        },
    }
    with open(os.path.join(output_dir, f'{case_name}_method_comparison_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate direct comparison figures for a baseline and a candidate method')
    parser.add_argument('--case', required=True)
    parser.add_argument('--baseline_name', default='PINN')
    parser.add_argument('--method_name', default='Candidate method')
    parser.add_argument('--baseline_dir', required=True)
    parser.add_argument('--method_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    baseline_summary = load_summary(args.baseline_dir)
    method_summary = load_summary(args.method_dir)
    baseline_history = load_loss_history(args.baseline_dir)
    method_history = load_loss_history(args.method_dir)
    baseline_grid = load_eval_grid(args.baseline_dir)
    method_grid = load_eval_grid(args.method_dir)

    plot_metric_comparison(args.case, args.baseline_name, args.method_name, baseline_summary, method_summary, output_dir)
    plot_loss_comparison(args.case, args.baseline_name, args.method_name, baseline_history, method_history, output_dir)
    plot_material_maps(args.case, args.baseline_name, args.method_name, baseline_grid, method_grid, output_dir)
    save_summary_json(args.case, args.baseline_name, args.method_name, baseline_summary, method_summary, output_dir)

    print(f'Comparison figures saved to: {output_dir}')


if __name__ == '__main__':
    main()
