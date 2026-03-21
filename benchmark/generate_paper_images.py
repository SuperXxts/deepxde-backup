import argparse
import json
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np


def resolve_shared_dir():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.join(current_dir, "..", "deepxde-master", "my_example", "spatial_material_inverse"),
        os.path.join(current_dir, "..", "spatial_material_inverse"),
        os.path.join(current_dir, "..", "deepxde_work", "deepxde-master", "my_example", "spatial_material_inverse"),
    ]
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.exists(os.path.join(candidate, "shared.py")):
            return candidate
    raise FileNotFoundError("Cannot locate spatial_material_inverse/shared.py")


SHARED_DIR = resolve_shared_dir()
if SHARED_DIR not in sys.path:
    sys.path.append(SHARED_DIR)

from shared import exact_state_numpy, get_case_config, make_grid


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def default_output_root():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(current_dir, "images")


def plot_case_definition(case_name, output_dir):
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
        [r"True $\lambda(x,y)$", r"True $\mu(x,y)$", "True displacement magnitude"],
    ):
        image = axis.contourf(xx, yy, grid, levels=100, cmap="viridis")
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_aspect("equal")
        plt.colorbar(image, ax=axis)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{case_name}_problem_definition.png"), dpi=300)
    plt.close(fig)


def plot_model_architecture(output_dir):
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
            box = plt.Rectangle((x - 0.12, y - 0.09), 0.24, 0.18, facecolor="#355F94", alpha=0.10, edgecolor="#355F94", linewidth=1.5)
            axis.add_patch(box)
            axis.text(x, y, text, ha="center", va="center", fontsize=10, color="#1f2937")
        if title == "PINN baseline":
            axis.annotate("", xy=(0.27, 0.50), xytext=(0.20, 0.50), arrowprops=dict(arrowstyle="->", lw=1.4, color="#355F94"))
            axis.annotate("", xy=(0.60, 0.50), xytext=(0.50, 0.50), arrowprops=dict(arrowstyle="->", lw=1.4, color="#355F94"))
        else:
            axis.annotate("", xy=(0.30, 0.63), xytext=(0.20, 0.50), arrowprops=dict(arrowstyle="->", lw=1.4, color="#355F94"))
            axis.annotate("", xy=(0.30, 0.37), xytext=(0.20, 0.50), arrowprops=dict(arrowstyle="->", lw=1.4, color="#355F94"))
            axis.annotate("", xy=(0.66, 0.55), xytext=(0.54, 0.66), arrowprops=dict(arrowstyle="->", lw=1.4, color="#355F94"))
            axis.annotate("", xy=(0.66, 0.45), xytext=(0.54, 0.34), arrowprops=dict(arrowstyle="->", lw=1.4, color="#355F94"))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_architecture.png"), dpi=300)
    plt.close(fig)


def plot_sampling_from_exp(exp_dir, output_dir):
    observation_path = os.path.join(exp_dir, "npz", "observation_data.npz")
    domain_path = os.path.join(exp_dir, "npz", "domain_points.npz")
    if not (os.path.exists(observation_path) and os.path.exists(domain_path)):
        return False

    observation_data = np.load(observation_path)
    domain_data = np.load(domain_path)
    observation_points = observation_data["observation_points"]
    domain_points = domain_data["domain_points"]

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(domain_points[:, 0], domain_points[:, 1], s=8, alpha=0.22, label="Domain points")
    ax.scatter(
        observation_points[:, 0],
        observation_points[:, 1],
        s=16,
        alpha=0.85,
        color="#d97706",
        label="Observation points",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_title("Training and observation points")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sampling_layout.png"), dpi=300)
    plt.close(fig)
    return True


def plot_result_summary(exp_dir, output_dir):
    evaluation_grid_path = os.path.join(exp_dir, "npz", "evaluation_grid.npz")
    metrics_path = os.path.join(exp_dir, "metrics", "evaluation_metrics.json")
    if not (os.path.exists(evaluation_grid_path) and os.path.exists(metrics_path)):
        return False

    evaluation = np.load(evaluation_grid_path)
    with open(metrics_path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    xx = evaluation["xx"]
    yy = evaluation["yy"]
    truth = evaluation["truth"]
    prediction = evaluation["prediction"]
    ny, nx = xx.shape

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, index, title in zip(
        axes.reshape(-1),
        [5, 6, 0, 1],
        [r"True vs Predicted $\lambda$", r"True vs Predicted $\mu$", "Displacement ux", "Displacement uy"],
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
    plt.savefig(os.path.join(output_dir, "result_error_summary.png"), dpi=300)
    plt.close(fig)

    field_names = list(metrics["field_relative_l2"].keys())
    values = [metrics["field_relative_l2"][name] for name in field_names]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(field_names, values, color="#355F94")
    ax.set_title("Relative L2 error by field")
    ax.set_ylabel("Relative L2 error")
    ax.set_xlabel("Field")
    ax.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_bar.png"), dpi=300)
    plt.close(fig)

    loss_json_path = os.path.join(exp_dir, "json", "loss_history.json")
    if os.path.exists(loss_json_path):
        with open(loss_json_path, "r", encoding="utf-8") as handle:
            history = json.load(handle)
        steps = history.get("steps", [])
        loss_train = np.asarray(history.get("loss_train", []), dtype=float)
        loss_test = np.asarray(history.get("loss_test", []), dtype=float)
        if len(steps) > 0 and loss_train.size > 0:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(steps, np.sum(loss_train, axis=1), label="Train loss", color="#355F94")
            if loss_test.size > 0:
                ax.plot(steps, np.sum(loss_test, axis=1), label="Test loss", color="#d97706")
            ax.set_yscale("log")
            ax.set_xlabel("Steps")
            ax.set_ylabel("Loss")
            ax.set_title("Training loss summary")
            ax.grid(True, alpha=0.2)
            ax.legend(frameon=False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "training_loss_summary.png"), dpi=300)
            plt.close(fig)
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Generate benchmark and paper-ready images")
    parser.add_argument("--case", choices=["layered", "single_inclusion", "double_inclusion"], default="single_inclusion")
    parser.add_argument("--exp_dir", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=default_output_root())
    return parser.parse_args()


def main():
    args = parse_args()
    issue_dir = ensure_dir(os.path.join(args.output_root, "问题定义"))
    model_dir = ensure_dir(os.path.join(args.output_root, "模型结构"))
    sampling_dir = ensure_dir(os.path.join(args.output_root, "训练采样"))
    result_dir = ensure_dir(os.path.join(args.output_root, "实验结果"))

    plot_case_definition(args.case, issue_dir)
    plot_model_architecture(model_dir)

    if args.exp_dir:
        plot_sampling_from_exp(args.exp_dir, sampling_dir)
        plot_result_summary(args.exp_dir, result_dir)

        for filename in ["loss_history.png", "loss_components.png", "observation_fit.png"]:
            source = os.path.join(args.exp_dir, "png", filename)
            if os.path.exists(source):
                shutil.copy2(source, os.path.join(result_dir, filename))

    print(f"Images saved to: {args.output_root}")


if __name__ == "__main__":
    main()
