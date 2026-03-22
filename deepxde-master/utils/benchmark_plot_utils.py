import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

UTILS_DIR = Path(__file__).resolve().parent
DEEPXDE_ROOT = UTILS_DIR.parent
REPO_ROOT = DEEPXDE_ROOT.parent
SPATIAL_DIR = DEEPXDE_ROOT / "my_example" / "spatial_material_inverse"

BLUE = "#355F94"
GOLD = "#D2A071"
INK = "#13233A"
PALE = "#EEF3F8"
PALE2 = "#F7F3EC"
GREEN = "#3C7A5A"
RED = "#A94F4F"


def _load_spatial_shared_symbols():
    if str(SPATIAL_DIR) not in sys.path:
        sys.path.append(str(SPATIAL_DIR))
    from shared import exact_state_numpy, get_case_config, make_grid

    return exact_state_numpy, get_case_config, make_grid


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_summary(exp_dir):
    return load_json(Path(exp_dir) / "json" / "train_summary.json")


def load_loss_history(exp_dir):
    return load_json(Path(exp_dir) / "json" / "loss_history.json")


def load_eval_grid(exp_dir):
    return np.load(Path(exp_dir) / "npz" / "evaluation_grid.npz")


def default_benchmark_images_root():
    return REPO_ROOT / "benchmark" / "images"


def plot_metric_comparison(case_name, baseline_name, method_name, baseline_summary, method_summary, output_dir):
    output_dir = ensure_dir(output_dir)
    metric_names = ["lambda", "mu", "ux", "uy", "sxx", "syy", "sxy"]
    baseline_values = [baseline_summary["post_train_metrics"]["field_relative_l2"][name] for name in metric_names]
    method_values = [method_summary["post_train_metrics"]["field_relative_l2"][name] for name in metric_names]

    x = np.arange(len(metric_names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(x - width / 2, baseline_values, width, label=baseline_name, color=BLUE)
    ax.bar(x + width / 2, method_values, width, label=method_name, color=GOLD)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("Relative L2 error")
    ax.set_xlabel("Field")
    ax.set_title(f"{case_name}: method comparison by field")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / f"{case_name}_method_comparison_metrics.png", dpi=300)
    plt.close(fig)

    extra_names = ["observation_mse", "pde_residual_mean_abs", "material_field_mean_abs_vector_error"]
    baseline_extra = [baseline_summary["post_train_metrics"][name] for name in extra_names]
    method_extra = [method_summary["post_train_metrics"][name] for name in extra_names]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    x = np.arange(len(extra_names))
    ax.bar(x - width / 2, baseline_extra, width, label=baseline_name, color=BLUE)
    ax.bar(x + width / 2, method_extra, width, label=method_name, color=GOLD)
    ax.set_xticks(x)
    ax.set_xticklabels(["Obs MSE", "PDE residual", "Material MAE"])
    ax.set_ylabel("Value")
    ax.set_title(f"{case_name}: inverse-task summary metrics")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / f"{case_name}_method_comparison_inverse_metrics.png", dpi=300)
    plt.close(fig)


def plot_loss_comparison(case_name, baseline_name, method_name, baseline_history, method_history, output_dir):
    output_dir = ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for history, label, color in [
        (baseline_history, baseline_name, BLUE),
        (method_history, method_name, GOLD),
    ]:
        steps = np.asarray(history.get("steps", []), dtype=float)
        loss_train = np.asarray(history.get("loss_train", []), dtype=float)
        if len(steps) == 0 or loss_train.size == 0:
            continue
        ax.plot(steps, np.sum(loss_train, axis=1), label=f"{label} train", color=color, linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Total train loss")
    ax.set_title(f"{case_name}: train loss comparison")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / f"{case_name}_method_comparison_loss.png", dpi=300)
    plt.close(fig)


def plot_material_maps(case_name, baseline_name, method_name, baseline_grid, method_grid, output_dir):
    output_dir = ensure_dir(output_dir)
    xx = baseline_grid["xx"]
    yy = baseline_grid["yy"]
    truth = baseline_grid["truth"]
    pred_baseline = baseline_grid["prediction"]
    pred_method = method_grid["prediction"]
    ny, nx = xx.shape

    panels = [
        ("True lambda", truth[:, 5].reshape(ny, nx)),
        (f"{baseline_name} lambda", pred_baseline[:, 5].reshape(ny, nx)),
        (f"{method_name} lambda", pred_method[:, 5].reshape(ny, nx)),
        ("True mu", truth[:, 6].reshape(ny, nx)),
        (f"{baseline_name} mu", pred_baseline[:, 6].reshape(ny, nx)),
        (f"{method_name} mu", pred_method[:, 6].reshape(ny, nx)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))
    for axis, (title, grid) in zip(axes.reshape(-1), panels):
        image = axis.contourf(xx, yy, grid, levels=100, cmap="viridis")
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_aspect("equal")
        plt.colorbar(image, ax=axis)
    plt.tight_layout()
    plt.savefig(output_dir / f"{case_name}_method_comparison_material_maps.png", dpi=300)
    plt.close(fig)


def save_method_comparison_summary(case_name, baseline_name, method_name, baseline_summary, method_summary, output_dir):
    output_dir = ensure_dir(output_dir)
    payload = {
        "case": case_name,
        "baseline_name": baseline_name,
        "method_name": method_name,
        "baseline_metrics": baseline_summary["post_train_metrics"],
        "method_metrics": method_summary["post_train_metrics"],
        "parameter_count": {
            baseline_name: baseline_summary["parameter_count"],
            method_name: method_summary["parameter_count"],
        },
    }
    with (output_dir / f"{case_name}_method_comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def generate_method_comparison(case_name, baseline_name, method_name, baseline_dir, method_dir, output_dir):
    baseline_summary = load_summary(baseline_dir)
    method_summary = load_summary(method_dir)
    baseline_history = load_loss_history(baseline_dir)
    method_history = load_loss_history(method_dir)
    baseline_grid = load_eval_grid(baseline_dir)
    method_grid = load_eval_grid(method_dir)

    plot_metric_comparison(case_name, baseline_name, method_name, baseline_summary, method_summary, output_dir)
    plot_loss_comparison(case_name, baseline_name, method_name, baseline_history, method_history, output_dir)
    plot_material_maps(case_name, baseline_name, method_name, baseline_grid, method_grid, output_dir)
    save_method_comparison_summary(case_name, baseline_name, method_name, baseline_summary, method_summary, output_dir)


def plot_case_definition(case_name, output_dir):
    output_dir = ensure_dir(output_dir)
    exact_state_numpy, get_case_config, make_grid = _load_spatial_shared_symbols()
    case_config = get_case_config(case_name)
    points, xx, yy = make_grid(200, 200)
    truth = exact_state_numpy(points, case_config)
    lambda_grid = truth[:, 5].reshape(xx.shape)
    mu_grid = truth[:, 6].reshape(xx.shape)
    disp_mag = np.sqrt(truth[:, 0] ** 2 + truth[:, 1] ** 2).reshape(xx.shape)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, grid, title in zip(
        axes,
        [lambda_grid, mu_grid, disp_mag],
        [r"True $\\lambda(x,y)$", r"True $\\mu(x,y)$", "True displacement magnitude"],
    ):
        image = axis.contourf(xx, yy, grid, levels=100, cmap="viridis")
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_aspect("equal")
        plt.colorbar(image, ax=axis)
    plt.tight_layout()
    plt.savefig(output_dir / f"{case_name}_problem_definition.png", dpi=300)
    plt.close(fig)


def plot_model_architecture(output_dir):
    output_dir = ensure_dir(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    diagrams = [
        (
            axes[0],
            "PINN baseline",
            [
                (0.12, 0.50, "Input\n(x, y)"),
                (0.38, 0.50, "Shared MLP"),
                (0.72, 0.50, "Outputs\nux, uy, sxx, syy, sxy,\nlambda, mu"),
            ],
        ),
        (
            axes[1],
            "Direction A concept",
            [
                (0.12, 0.50, "Input\n(x, y)"),
                (0.42, 0.68, "State net\nux, uy, sxx, syy, sxy"),
                (0.42, 0.32, "Interface net\nphi(x, y)"),
                (0.78, 0.50, "Material mixing\nbg/inc params\n+ physics loss"),
            ],
        ),
    ]

    for axis, title, nodes in diagrams:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")
        axis.set_title(title)
        for x, y, text in nodes:
            box = plt.Rectangle((x - 0.12, y - 0.09), 0.24, 0.18, facecolor=BLUE, alpha=0.10, edgecolor=BLUE, linewidth=1.5)
            axis.add_patch(box)
            axis.text(x, y, text, ha="center", va="center", fontsize=10, color="#1f2937")
        if title == "PINN baseline":
            axis.annotate("", xy=(0.27, 0.50), xytext=(0.20, 0.50), arrowprops=dict(arrowstyle="->", lw=1.4, color=BLUE))
            axis.annotate("", xy=(0.60, 0.50), xytext=(0.50, 0.50), arrowprops=dict(arrowstyle="->", lw=1.4, color=BLUE))
        else:
            axis.annotate("", xy=(0.30, 0.63), xytext=(0.20, 0.50), arrowprops=dict(arrowstyle="->", lw=1.4, color=BLUE))
            axis.annotate("", xy=(0.30, 0.37), xytext=(0.20, 0.50), arrowprops=dict(arrowstyle="->", lw=1.4, color=BLUE))
            axis.annotate("", xy=(0.66, 0.55), xytext=(0.54, 0.66), arrowprops=dict(arrowstyle="->", lw=1.4, color=BLUE))
            axis.annotate("", xy=(0.66, 0.45), xytext=(0.54, 0.34), arrowprops=dict(arrowstyle="->", lw=1.4, color=BLUE))

    plt.tight_layout()
    plt.savefig(output_dir / "model_architecture.png", dpi=300)
    plt.close(fig)


def plot_sampling_from_exp(exp_dir, output_dir):
    output_dir = ensure_dir(output_dir)
    observation_path = Path(exp_dir) / "npz" / "observation_data.npz"
    domain_path = Path(exp_dir) / "npz" / "domain_points.npz"
    if not (observation_path.exists() and domain_path.exists()):
        return False

    observation_data = np.load(observation_path)
    domain_data = np.load(domain_path)
    observation_points = observation_data["observation_points"]
    domain_points = domain_data["domain_points"]

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(domain_points[:, 0], domain_points[:, 1], s=8, alpha=0.22, label="Domain points")
    ax.scatter(observation_points[:, 0], observation_points[:, 1], s=16, alpha=0.85, color="#d97706", label="Observation points")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_title("Training and observation points")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_dir / "sampling_layout.png", dpi=300)
    plt.close(fig)
    return True


def plot_result_summary(exp_dir, output_dir):
    output_dir = ensure_dir(output_dir)
    evaluation_grid_path = Path(exp_dir) / "npz" / "evaluation_grid.npz"
    metrics_path = Path(exp_dir) / "metrics" / "evaluation_metrics.json"
    if not (evaluation_grid_path.exists() and metrics_path.exists()):
        return False

    evaluation = np.load(evaluation_grid_path)
    metrics = load_json(metrics_path)
    xx = evaluation["xx"]
    yy = evaluation["yy"]
    truth = evaluation["truth"]
    prediction = evaluation["prediction"]
    ny, nx = xx.shape

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, index, title in zip(
        axes.reshape(-1),
        [5, 6, 0, 1],
        [r"True vs Predicted $\\lambda$", r"True vs Predicted $\\mu$", "Displacement ux", "Displacement uy"],
    ):
        truth_grid = truth[:, index].reshape(ny, nx)
        pred_grid = prediction[:, index].reshape(ny, nx)
        error_grid = np.abs(pred_grid - truth_grid)
        image = axis.contourf(xx, yy, error_grid, levels=100, cmap="magma")
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_aspect("equal")
        plt.colorbar(image, ax=axis)
    plt.tight_layout()
    plt.savefig(output_dir / "result_error_summary.png", dpi=300)
    plt.close(fig)

    field_names = list(metrics["field_relative_l2"].keys())
    values = [metrics["field_relative_l2"][name] for name in field_names]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(field_names, values, color=BLUE)
    ax.set_title("Relative L2 error by field")
    ax.set_ylabel("Relative L2 error")
    ax.set_xlabel("Field")
    ax.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_bar.png", dpi=300)
    plt.close(fig)

    loss_json_path = Path(exp_dir) / "json" / "loss_history.json"
    if loss_json_path.exists():
        history = load_json(loss_json_path)
        steps = history.get("steps", [])
        loss_train = np.asarray(history.get("loss_train", []), dtype=float)
        loss_test = np.asarray(history.get("loss_test", []), dtype=float)
        if len(steps) > 0 and loss_train.size > 0:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(steps, np.sum(loss_train, axis=1), label="Train loss", color=BLUE)
            if loss_test.size > 0:
                ax.plot(steps, np.sum(loss_test, axis=1), label="Test loss", color="#d97706")
            ax.set_yscale("log")
            ax.set_xlabel("Steps")
            ax.set_ylabel("Loss")
            ax.set_title("Training loss summary")
            ax.grid(True, alpha=0.2)
            ax.legend(frameon=False)
            plt.tight_layout()
            plt.savefig(output_dir / "training_loss_summary.png", dpi=300)
            plt.close(fig)
    return True


def generate_paper_images(case_name, exp_dir=None, output_root=None):
    output_root = ensure_dir(output_root or default_benchmark_images_root())
    issue_dir = ensure_dir(output_root / "????")
    model_dir = ensure_dir(output_root / "????")
    sampling_dir = ensure_dir(output_root / "????")
    result_dir = ensure_dir(output_root / "????")

    plot_case_definition(case_name, issue_dir)
    plot_model_architecture(model_dir)
    if exp_dir:
        plot_sampling_from_exp(exp_dir, sampling_dir)
        plot_result_summary(exp_dir, result_dir)
        for filename in ["loss_history.png", "loss_components.png", "observation_fit.png"]:
            source = Path(exp_dir) / "png" / filename
            if source.exists():
                shutil.copy2(source, result_dir / filename)
    return output_root


def _box(ax, x, y, w, h, text, fc=PALE, ec=BLUE, fontsize=11, weight="regular"):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02", linewidth=2.0, edgecolor=ec, facecolor=fc)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=INK, weight=weight)


def _arrow(ax, x1, y1, x2, y2, color=BLUE, lw=2.0, style="-|>"):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14, linewidth=lw, color=color, shrinkA=2, shrinkB=2)
    ax.add_patch(arr)


def generate_pinn_vs_iaminn_v1_architecture(output_path=None):
    output_path = Path(output_path or (default_benchmark_images_root() / "model_architecture" / "pinn_vs_iaminn_v1_architecture.png"))
    ensure_dir(output_path.parent)

    fig = plt.figure(figsize=(16, 9), dpi=220)
    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.25, 0.94, "PINN Baseline", ha="center", va="center", fontsize=22, color=INK, weight="bold")
    ax.text(0.75, 0.94, "IAMINN-v1", ha="center", va="center", fontsize=22, color=INK, weight="bold")

    _box(ax, 0.07, 0.78, 0.14, 0.08, "(x, y)", fc=PALE2, fontsize=14, weight="bold")
    _box(ax, 0.07, 0.63, 0.18, 0.09, "Fourier feature map\n(shared input encoding)", fc=PALE, fontsize=12)
    _box(ax, 0.06, 0.43, 0.22, 0.12, "Single MLP backbone\n128-128-128-128", fc="#E8F0FA", fontsize=13, weight="bold")
    _box(ax, 0.05, 0.22, 0.24, 0.12, "Direct outputs\nux, uy, sxx, syy, sxy, lambda(x,y), mu(x,y)", fc="#DDEAF8", fontsize=12)
    _box(ax, 0.04, 0.05, 0.26, 0.10, "Material field assumption\nfree continuous field regression", fc="#E8F3EC", ec=GREEN, fontsize=12, weight="bold")
    _arrow(ax, 0.14, 0.78, 0.16, 0.72)
    _arrow(ax, 0.16, 0.63, 0.17, 0.55)
    _arrow(ax, 0.17, 0.43, 0.17, 0.34)
    _arrow(ax, 0.17, 0.22, 0.17, 0.15, color=GREEN)

    _box(ax, 0.59, 0.78, 0.14, 0.08, "(x, y)", fc=PALE2, fontsize=14, weight="bold")
    _box(ax, 0.56, 0.63, 0.20, 0.09, "Fourier feature map\n(shared input encoding)", fc=PALE, fontsize=12)
    _box(ax, 0.47, 0.43, 0.18, 0.12, "State net\n128-128-128-128", fc="#E8F0FA", fontsize=13, weight="bold")
    _box(ax, 0.72, 0.43, 0.18, 0.12, "Interface net\n64-64-64", fc="#F4E8DA", ec=GOLD, fontsize=13, weight="bold")
    _box(ax, 0.45, 0.22, 0.22, 0.12, "State outputs\nux, uy, sxx, syy, sxy", fc="#DDEAF8", fontsize=12)
    _box(ax, 0.71, 0.24, 0.20, 0.08, "Region logits", fc="#FAEFD9", ec=GOLD, fontsize=12)
    _box(ax, 0.71, 0.13, 0.20, 0.08, "Softmax over\nbackground + regions", fc="#FAEFD9", ec=GOLD, fontsize=12)
    _box(ax, 0.73, 0.03, 0.17, 0.07, "class probabilities", fc="#FAEFD9", ec=GOLD, fontsize=11)
    _box(ax, 0.49, 0.03, 0.19, 0.10, "Global region params\nraw_lambda_params\nraw_mu_params", fc="#FBE7E7", ec=RED, fontsize=11, weight="bold")
    _box(ax, 0.58, 0.18, 0.17, 0.09, "Weighted mixture\n=> lambda(x,y), mu(x,y)", fc="#E8F3EC", ec=GREEN, fontsize=11, weight="bold")
    _box(ax, 0.46, 0.84, 0.48, 0.04, "Structured assumption: material field = region mixture, not free field regression", fc="#F4F7FB", fontsize=11)
    _arrow(ax, 0.66, 0.78, 0.66, 0.72)
    _arrow(ax, 0.66, 0.63, 0.56, 0.55)
    _arrow(ax, 0.66, 0.63, 0.81, 0.55)
    _arrow(ax, 0.56, 0.43, 0.56, 0.34)
    _arrow(ax, 0.81, 0.43, 0.81, 0.32, color=GOLD)
    _arrow(ax, 0.81, 0.24, 0.81, 0.21, color=GOLD)
    _arrow(ax, 0.81, 0.13, 0.81, 0.10, color=GOLD)
    _arrow(ax, 0.68, 0.08, 0.72, 0.18, color=RED)
    _arrow(ax, 0.73, 0.07, 0.67, 0.18, color=GOLD)
    _arrow(ax, 0.66, 0.18, 0.66, 0.15, color=GREEN)

    _box(ax, 0.23, 0.005, 0.54, 0.06, "Shared training/evaluation protocol: same PDE residuals + same soft boundary points + same train/val/eval observations + same optimizer budget", fc="#F4F7FB", fontsize=11, weight="bold")
    _box(ax, 0.02, 0.87, 0.26, 0.05, "PINN advantage: high flexibility for lambda(x,y), mu(x,y)", fc="#E8F3EC", ec=GREEN, fontsize=10)
    _box(ax, 0.70, 0.87, 0.26, 0.05, "IAMINN-v1 prior: encourages region-like material structure", fc="#FBE7E7", ec=RED, fontsize=10)

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path
