# -*- coding: utf-8 -*-
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


ROOT = Path("/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master")
EXP = ROOT / "exp"
OUT = EXP / "_paper_figures_geotech_20260609_uniform_kmu"


GROUPS = [
    {
        "name": "analytic_layered",
        "outputs": ["analytic_layered_pinn_fw_kmu.png", "analytic_layered_tba_kmu.png"],
        "runs": [
            EXP / "01.Base/PINN/PINN + 固定权重 + L3/100k",
            EXP / "01.Base/Three_Branch/Three_Branch + 自适应权重 + L3/100k",
        ],
    },
    {
        "name": "analytic_inclusion",
        "outputs": ["analytic_inclusion_pinn_fw_kmu.png", "analytic_inclusion_tba_kmu.png"],
        "runs": [
            EXP / "01.Base/PINN/PINN + 固定权重 + S2/100k",
            EXP / "01.Base/Three_Branch/Three_Branch + 自适应权重 + S2/100k",
        ],
    },
    {
        "name": "sparse_layered_600",
        "outputs": ["kmu_layered_sparse600_pinn.png", "kmu_layered_sparse600_tba.png"],
        "runs": [
            EXP / "03.Sparsity/PINN_Sparsity/PINN + 固定权重 + L3 + 600观测值/100k",
            EXP / "03.Sparsity/Three_Branch_Sparsity/Three_Branch + 自适应权重 + L3 + 600观测值/100k",
        ],
    },
    {
        "name": "sparse_inclusion_600",
        "outputs": ["kmu_inclusion_sparse600_pinn.png", "kmu_inclusion_sparse600_tba.png"],
        "runs": [
            EXP / "03.Sparsity/PINN_Sparsity/PINN + 固定权重 + S2 + 600观测值/100k",
            EXP / "03.Sparsity/Three_Branch_Sparsity/Three_Branch + 自适应权重 + S2 + 600观测值/100k",
        ],
    },
    {
        "name": "fem_inclusion_medium",
        "outputs": ["fem_medium_inclusion_pinn_kmu.png", "fem_medium_inclusion_tba_kmu.png"],
        "runs": [
            EXP / "04.FEM/100k/PINN/single_inclusion/inclusion_pinn_fixed_s2_fem_medium_100k",
            EXP / "04.FEM/100k/Three_Branch/single_inclusion/inclusion_three_adapt_s2_fem_medium_100k",
        ],
    },
    {
        "name": "weak_interlayer_clean",
        "outputs": ["weak_interlayer_pinn_kmu.png", "weak_interlayer_tba_kmu.png"],
        "runs": [
            EXP / "06.GeotechnicalBenchmark/01.Base/PINN/weak_interlayer/weak_interlayer_pinn_fixed_l3_150k_20260602",
            EXP / "06.GeotechnicalBenchmark/01.Base/Three_Branch/weak_interlayer/weak_interlayer_three_adapt_l3_150k_20260602",
        ],
    },
    {
        "name": "weak_interlayer_noise_3pct",
        "outputs": ["weak_interlayer_noise_pinn_3pct_kmu.png", "weak_interlayer_noise_tba_3pct_kmu.png"],
        "runs": [
            EXP / "08.WeakInterlayerNoiseScale/01.GlobalRMS/PINN/weak_interlayer/weak_interlayer_pinn_globalrms_n03_150k_20260604",
            EXP / "08.WeakInterlayerNoiseScale/01.GlobalRMS/Three_Branch/weak_interlayer/weak_interlayer_three_globalrms_n03_150k_20260604",
        ],
    },
]


def as_grid(values, shape):
    return np.asarray(values, dtype=float).reshape(shape)


def bulk_from_lambda_mu(lambda_grid, mu_grid):
    return np.asarray(lambda_grid, dtype=float) + (2.0 / 3.0) * np.asarray(mu_grid, dtype=float)


def load_run(run_dir):
    path = Path(run_dir) / "npz" / "evaluation_grid.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    xx = np.asarray(data["xx"], dtype=float)
    yy = np.asarray(data["yy"], dtype=float)
    shape = xx.shape
    truth = np.asarray(data["truth"], dtype=float)
    pred = np.asarray(data["prediction"], dtype=float)

    true_lambda = as_grid(truth[:, 5], shape)
    pred_lambda = as_grid(pred[:, 5], shape)
    true_mu = as_grid(truth[:, 6], shape)
    pred_mu = as_grid(pred[:, 6], shape)
    true_bulk = bulk_from_lambda_mu(true_lambda, true_mu)
    pred_bulk = bulk_from_lambda_mu(pred_lambda, pred_mu)
    return {
        "run_dir": Path(run_dir),
        "xx": xx,
        "yy": yy,
        "bulk_true": true_bulk,
        "bulk_pred": pred_bulk,
        "mu_true": true_mu,
        "mu_pred": pred_mu,
    }


def minmax(arrays):
    joined = np.concatenate([np.ravel(np.asarray(a, dtype=float)) for a in arrays])
    return float(np.nanmin(joined)), float(np.nanmax(joined))


def levels(vmin, vmax, count=101):
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = -1.0, 1.0
    if np.isclose(vmin, vmax):
        delta = max(abs(vmin) * 1e-6, 1e-8)
        vmin -= delta
        vmax += delta
    return float(vmin), float(vmax), np.linspace(float(vmin), float(vmax), count)


def tick_formatter(value, _):
    rounded = round(float(value), 3)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:.3f}"


def format_axis(axis):
    ticks = np.linspace(0.0, 1.0, 5)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    formatter = FuncFormatter(tick_formatter)
    axis.xaxis.set_major_formatter(formatter)
    axis.yaxis.set_major_formatter(formatter)
    axis.set_aspect("equal")


def add_colorbar(fig, image, axis, vmin, vmax):
    cb = fig.colorbar(image, ax=axis)
    cb.set_ticks(np.linspace(float(vmin), float(vmax), 5))
    formatter = FuncFormatter(tick_formatter)
    cb.ax.yaxis.set_major_formatter(formatter)
    cb.formatter = formatter
    cb.ax.yaxis.get_offset_text().set_visible(False)
    cb.update_ticks()


def plot_run(run, output_path, limits_for_group):
    bulk_value_limits, mu_value_limits, bulk_error_limits, mu_error_limits = limits_for_group
    bulk_error = np.abs(run["bulk_pred"] - run["bulk_true"])
    mu_error = np.abs(run["mu_pred"] - run["mu_true"])

    bulk_vmin, bulk_vmax, bulk_levels = levels(*bulk_value_limits)
    mu_vmin, mu_vmax, mu_levels = levels(*mu_value_limits)
    bulk_err_vmin, bulk_err_vmax, bulk_err_levels = levels(*bulk_error_limits)
    mu_err_vmin, mu_err_vmax, mu_err_levels = levels(*mu_error_limits)

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.5), constrained_layout=True)
    items = [
        (axes[0, 0], run["bulk_true"], bulk_levels, "viridis", bulk_vmin, bulk_vmax, "True Bulk modulus K"),
        (axes[0, 1], run["bulk_pred"], bulk_levels, "viridis", bulk_vmin, bulk_vmax, "Predicted Bulk modulus K"),
        (axes[0, 2], bulk_error, bulk_err_levels, "magma", bulk_err_vmin, bulk_err_vmax, "Absolute error of Bulk modulus K"),
        (axes[1, 0], run["mu_true"], mu_levels, "viridis", mu_vmin, mu_vmax, "True Shear modulus Mu"),
        (axes[1, 1], run["mu_pred"], mu_levels, "viridis", mu_vmin, mu_vmax, "Predicted Shear modulus Mu"),
        (axes[1, 2], mu_error, mu_err_levels, "magma", mu_err_vmin, mu_err_vmax, "Absolute error of Shear modulus Mu"),
    ]
    for axis, grid, lvls, cmap, vmin, vmax, title in items:
        image = axis.contourf(run["xx"], run["yy"], grid, levels=lvls, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        format_axis(axis)
        add_colorbar(fig, image, axis, vmin, vmax)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for group in GROUPS:
        runs = [load_run(path) for path in group["runs"]]
        bulk_value_limits = minmax([r["bulk_true"] for r in runs] + [r["bulk_pred"] for r in runs])
        mu_value_limits = minmax([r["mu_true"] for r in runs] + [r["mu_pred"] for r in runs])
        bulk_error_limits = (0.0, minmax([np.abs(r["bulk_pred"] - r["bulk_true"]) for r in runs])[1])
        mu_error_limits = (0.0, minmax([np.abs(r["mu_pred"] - r["mu_true"]) for r in runs])[1])
        limits_for_group = (bulk_value_limits, mu_value_limits, bulk_error_limits, mu_error_limits)
        print(
            f"{group['name']}: "
            f"K={bulk_value_limits[0]:.6f},{bulk_value_limits[1]:.6f}; "
            f"mu={mu_value_limits[0]:.6f},{mu_value_limits[1]:.6f}; "
            f"Kerr={bulk_error_limits[0]:.6f},{bulk_error_limits[1]:.6f}; "
            f"muerr={mu_error_limits[0]:.6f},{mu_error_limits[1]:.6f}"
        )
        for run, name in zip(runs, group["outputs"]):
            plot_run(run, OUT / name, limits_for_group)
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
