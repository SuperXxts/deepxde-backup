#!/usr/bin/env python3
"""Create paper-ready comparison figures for the 10 analytical v2 cases."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path("/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/boundary_uncertainty_pinn")
GROUP = PROJECT / "exp" / "01.AnalyticalBenchmark"
ROOT_OUT = GROUP / "07.ComparisonSummary"
IMAGE_DIR = ROOT_OUT / "image"
CSV_DIR = ROOT_OUT / "csv"

OLD_STAMP = "20260629_212009"
NEW_STAMP = "20260701_analytical_diag150k"
SEED = 42

CASES = [
    ("full_boundary_oracle", OLD_STAMP, "Full boundary"),
    ("full_boundary_oracle_reaction", NEW_STAMP, "Full boundary + reaction"),
    ("correct_top_amp_only", OLD_STAMP, "Correct top"),
    ("correct_top_amp_only_reaction", NEW_STAMP, "Correct top + reaction"),
    ("wrong_fixed_A", OLD_STAMP, "Wrong fixed A"),
    ("wrong_fixed_A_reaction", NEW_STAMP, "Wrong fixed A + reaction"),
    ("learnable_A", OLD_STAMP, "Learnable A"),
    ("learnable_A_anchor", NEW_STAMP, "Learnable A + anchors"),
    ("learnable_A_reaction", OLD_STAMP, "Learnable A + reaction"),
    ("learnable_A_anchor_reaction", OLD_STAMP, "Learnable A + anchors + reaction"),
]

KEY_CASES = [
    ("correct_top_amp_only", OLD_STAMP, "Correct top"),
    ("wrong_fixed_A", OLD_STAMP, "Wrong fixed A"),
    ("learnable_A", OLD_STAMP, "Learnable A"),
    ("learnable_A_reaction", OLD_STAMP, "Learnable A + reaction"),
    ("learnable_A_anchor_reaction", OLD_STAMP, "Learnable A + anchors + reaction"),
]

LEARNABLE_CASES = [
    ("learnable_A", OLD_STAMP, "Learnable A"),
    ("learnable_A_anchor", NEW_STAMP, "Learnable A + anchors"),
    ("learnable_A_reaction", OLD_STAMP, "Learnable A + reaction"),
    ("learnable_A_anchor_reaction", OLD_STAMP, "Learnable A + anchors + reaction"),
]


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 6.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.75,
        "axes.labelsize": 6.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 5.8,
        "legend.frameon": False,
    }
)


def run_dir(case: str, stamp: str) -> Path:
    return GROUP / f"v2_mms_single_obs225_seed{SEED}_{case}_{stamp}"


def load_metrics(case: str, stamp: str) -> dict:
    path = run_dir(case, stamp) / "metrics" / "metrics.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_eval(case: str, stamp: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    data = np.load(run_dir(case, stamp) / "npz" / "evaluation_predictions_data.npz", allow_pickle=True)
    names = [str(v) for v in data["field_names"]]
    return data["x"], data["y_true"], data["y_pred"], names


def load_loss(case: str, stamp: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(run_dir(case, stamp) / "npz" / "loss_history.npz", allow_pickle=True)
    steps = np.asarray(data["steps"], dtype=float)
    total = np.sum(np.asarray(data["loss_train"], dtype=float), axis=1)
    return steps, total


def load_amplitude(case: str, stamp: str) -> np.ndarray:
    path = run_dir(case, stamp) / "dat" / "amplitude_history.dat"
    arr = np.loadtxt(path, comments="#")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def save_all(fig: plt.Figure, stem: str, width: float | None = None) -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    if width is not None:
        fig.set_size_inches(width, fig.get_size_inches()[1], forward=True)
    for ext in ("pdf", "png", "jpg"):
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "jpg"}:
            kwargs["dpi"] = 600
        if ext == "jpg":
            kwargs["pil_kwargs"] = {"quality": 95}
        fig.savefig(IMAGE_DIR / f"{stem}.{ext}", **kwargs)
    plt.close(fig)


def style_for(label: str) -> tuple[str, str | None]:
    if "Wrong" in label:
        color = "#C44E52"
    elif "Learnable" in label:
        color = "#55A868"
    elif "Full" in label:
        color = "#4C72B0"
    else:
        color = "#8172B3"
    hatch = "///" if "reaction" in label.lower() else None
    return color, hatch


def make_summary() -> list[dict]:
    rows: list[dict] = []
    for case, stamp, label in CASES:
        m = load_metrics(case, stamp)
        rel = m["relative_l2"]
        rows.append(
            {
                "case": case,
                "stamp": stamp,
                "label": label,
                "final_A": float(m["final_A"]),
                "true_A": float(m["true_A"]),
                "A_error_percent": abs(float(m["final_A"]) - float(m["true_A"])) * 100.0,
                "ux_error_percent": rel["ux"] * 100.0,
                "uy_error_percent": rel["uy"] * 100.0,
                "K_error_percent": rel["K"] * 100.0,
                "mu_error_percent": rel["mu"] * 100.0,
                "E_error_percent": rel["E"] * 100.0,
                "m_error_percent": rel["m"] * 100.0,
                "reaction_error_percent": float(m["reaction_rel_error"]) * 100.0,
                "best_step": int(m["best_step"]),
                "final_train_loss_sum": float(m["final_train_loss_sum"]),
                "elapsed_seconds": float(m["elapsed_seconds"]),
            }
        )
    return rows


def write_tables(rows: list[dict]) -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = CSV_DIR / "summary_10_cases.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = CSV_DIR / "summary_10_cases.md"
    headers = ["Case", "A", "E err (%)", "ux err (%)", "uy err (%)", "reaction err (%)"]
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in rows:
            f.write(
                "| "
                + " | ".join(
                    [
                        r["label"],
                        f"{r['final_A']:.4f}",
                        f"{r['E_error_percent']:.2f}",
                        f"{r['ux_error_percent']:.2f}",
                        f"{r['uy_error_percent']:.2f}",
                        f"{r['reaction_error_percent']:.2f}",
                    ]
                )
                + " |\n"
            )


def figure_result_table(rows: list[dict]) -> None:
    labels = [r["label"] for r in rows]
    columns = ["A", "E err.", "ux err.", "uy err.", "Reaction err."]
    display = np.asarray(
        [
            [
                r["final_A"],
                r["E_error_percent"],
                r["ux_error_percent"],
                r["uy_error_percent"],
                r["reaction_error_percent"],
            ]
            for r in rows
        ],
        dtype=float,
    )
    color_values = np.asarray(
        [
            [
                abs(r["final_A"] - 1.0) * 100.0,
                r["E_error_percent"],
                r["ux_error_percent"],
                r["uy_error_percent"],
                r["reaction_error_percent"],
            ]
            for r in rows
        ],
        dtype=float,
    )
    norm = color_values / np.maximum(np.nanmax(color_values, axis=0, keepdims=True), 1e-12)

    fig, ax = plt.subplots(figsize=(7.6, 4.3), constrained_layout=True)
    im = ax.imshow(norm, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("10-case quantitative summary", loc="left", fontweight="bold")

    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            text = f"{display[i, j]:.4f}" if j == 0 else f"{display[i, j]:.2f}%"
            color = "white" if norm[i, j] > 0.55 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=5.6, color=color)

    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.76)
    cbar.set_label("Column-wise normalized error")
    save_all(fig, "fig0_result_table_image")


def figure_story_summary(rows: list[dict]) -> None:
    by_label = {r["label"]: r for r in rows}
    selected = [
        "Correct top",
        "Wrong fixed A",
        "Learnable A",
        "Learnable A + reaction",
        "Learnable A + anchors + reaction",
    ]
    x = np.arange(len(selected))
    e_err = [by_label[k]["E_error_percent"] for k in selected]
    r_err = [by_label[k]["reaction_error_percent"] for k in selected]

    fig, axes = plt.subplots(2, 1, figsize=(6.9, 4.8), sharex=True, constrained_layout=True)
    colors = ["#8172B3", "#C44E52", "#55A868", "#4C72B0", "#64B5CD"]
    axes[0].bar(x, e_err, color=colors, edgecolor="black", linewidth=0.35)
    axes[1].bar(x, r_err, color=colors, edgecolor="black", linewidth=0.35)
    axes[0].set_ylabel("E error (%)")
    axes[1].set_ylabel("Reaction error (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        ["Correct\nboundary", "Wrong\nboundary", "Learn\nA", "Learn A\n+ reaction", "Learn A\n+ anchors\n+ reaction"]
    )
    axes[0].set_title("Mechanism: boundary error, material pollution, and reaction-aided recovery", loc="left", fontweight="bold")
    for ax, vals in zip(axes, [e_err, r_err]):
        ax.grid(axis="y", color="#DDDDDD", lw=0.4)
        ax.set_ylim(0.0, max(vals) * 1.22)
        for xi, value in zip(x, vals):
            ax.text(xi, value + max(vals) * 0.035, f"{value:.1f}", ha="center", va="bottom", fontsize=5.8)

    axes[0].annotate(
        "wrong boundary\npollutes material",
        xy=(1, e_err[1]),
        xytext=(0.35, max(e_err) * 1.02),
        arrowprops={"arrowstyle": "->", "lw": 0.8},
        fontsize=6,
    )
    axes[0].annotate(
        "learn A + reaction\nrestores stiffness scale",
        xy=(3, e_err[3]),
        xytext=(2.45, max(e_err) * 0.55),
        arrowprops={"arrowstyle": "->", "lw": 0.8},
        fontsize=6,
    )
    save_all(fig, "fig6_story_summary")


def figure_metric_overview(rows: list[dict]) -> None:
    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))
    colors_hatches = [style_for(label) for label in labels]
    colors = [c for c, _ in colors_hatches]
    hatches = [h for _, h in colors_hatches]

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.7), constrained_layout=True)
    panels = [
        ("a", "Boundary scale", [r["final_A"] for r in rows], "Final A"),
        ("b", "Material inversion", [r["E_error_percent"] for r in rows], "Relative L2 error of E (%)"),
        ("c", "Displacement reconstruction", [r["ux_error_percent"] for r in rows], "Relative L2 error of ux (%)"),
        ("d", "Global reaction", [r["reaction_error_percent"] for r in rows], "Relative error (%)"),
    ]
    for ax, (letter, title, vals, ylabel) in zip(axes.ravel(), panels):
        vals = np.asarray(vals, dtype=float)
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.35)
        for bar, hatch in zip(bars, hatches):
            if hatch:
                bar.set_hatch(hatch)
        if title == "Boundary scale":
            ax.axhline(1.0, color="black", lw=0.9, ls="--")
            ax.set_ylim(0.80, 1.03)
        else:
            ax.set_ylim(0.0, float(np.max(vals)) * 1.18)
            for bar, value in zip(bars, vals):
                if value >= 20 or title == "Material inversion":
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        value + float(np.max(vals)) * 0.025,
                        f"{value:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=4.8,
                    )
        ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(axis="y", color="#DDDDDD", lw=0.4)
    save_all(fig, "fig1_metric_overview")


def figure_identifiability_scatter(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.6), constrained_layout=True)
    labels = [r["label"] for r in rows]
    x = np.asarray([r["A_error_percent"] for r in rows])
    y = np.asarray([r["E_error_percent"] for r in rows])
    c = np.asarray([r["reaction_error_percent"] for r in rows])
    sc = ax.scatter(x, y, c=c, s=48, cmap="viridis_r", edgecolor="black", linewidth=0.35)
    for xi, yi, label in zip(x, y, labels):
        ax.text(xi + 0.12, yi + 0.35, label, fontsize=5.8)
    ax.set_xlabel("|A - Atrue| (%)")
    ax.set_ylabel("Relative L2 error of E (%)")
    ax.set_title("Boundary-material identifiability", loc="left", fontweight="bold")
    ax.grid(color="#DDDDDD", lw=0.4)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Reaction error (%)")
    save_all(fig, "fig2_identifiability_scatter")


def grid_field(x: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.unique(x[:, 0])
    ys = np.unique(x[:, 1])
    return xs, ys, values.reshape(len(ys), len(xs))


def figure_material_maps() -> None:
    cases = [
        ("truth", "", "True"),
        ("correct_top_amp_only", OLD_STAMP, "Correct"),
        ("wrong_fixed_A", OLD_STAMP, "Wrong A"),
        ("learnable_A", OLD_STAMP, "Learn A"),
        ("learnable_A_reaction", OLD_STAMP, "Learn A+R"),
        ("learnable_A_anchor_reaction", OLD_STAMP, "Learn A+B+R"),
    ]
    ref_x, ref_true, _, names = load_eval("correct_top_amp_only", OLD_STAMP)
    idx_e = names.index("E")
    xs, ys, true_grid = grid_field(ref_x, ref_true[:, idx_e])
    truth_min = float(np.min(true_grid))
    truth_max = float(np.max(true_grid))

    pred_grids = []
    err_grids = []
    for case, stamp, _label in cases:
        if case == "truth":
            pred_grids.append(true_grid)
            err_grids.append(np.zeros_like(true_grid))
        else:
            x, truth, pred, names_i = load_eval(case, stamp)
            idx = names_i.index("E")
            _, _, pred_grid = grid_field(x, pred[:, idx])
            _, _, err_grid = grid_field(x, np.abs(pred[:, idx] - truth[:, idx]))
            pred_grids.append(pred_grid)
            err_grids.append(err_grid)

    err_max = min(float(max(np.max(e) for e in err_grids[1:])), 0.35)
    fig, axes = plt.subplots(2, len(cases), figsize=(8.6, 3.3), constrained_layout=True)
    for j, (_case, _stamp, label) in enumerate(cases):
        im0 = axes[0, j].imshow(
            pred_grids[j],
            origin="lower",
            extent=[xs.min(), xs.max(), ys.min(), ys.max()],
            vmin=truth_min,
            vmax=truth_max,
            cmap="viridis",
        )
        axes[0, j].set_title(label, fontsize=6.5)
        im1 = axes[1, j].imshow(
            err_grids[j],
            origin="lower",
            extent=[xs.min(), xs.max(), ys.min(), ys.max()],
            vmin=0.0,
            vmax=err_max,
            cmap="magma",
        )
        for ax in (axes[0, j], axes[1, j]):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
    axes[0, 0].set_ylabel("Prediction")
    axes[1, 0].set_ylabel("Abs. error")
    fig.colorbar(im0, ax=axes[0, :], shrink=0.65, label="E")
    fig.colorbar(im1, ax=axes[1, :], shrink=0.65, label="|error|, clipped")
    save_all(fig, "fig3_material_field_maps")


def figure_amplitude_history() -> None:
    fig, ax = plt.subplots(figsize=(4.9, 3.2), constrained_layout=True)
    palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
    for (case, stamp, label), color in zip(LEARNABLE_CASES, palette):
        arr = load_amplitude(case, stamp)
        ax.plot(arr[:, 0], arr[:, 1], label=label, lw=1.2, color=color)
    ax.axhline(1.0, color="black", lw=0.9, ls="--", label="True A")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Boundary scale A")
    ax.set_ylim(0.84, 1.02)
    ax.set_title("Recovery of the boundary scale", loc="left", fontweight="bold")
    ax.legend(ncol=1)
    ax.grid(color="#DDDDDD", lw=0.4)
    save_all(fig, "fig4_amplitude_history")


def figure_loss_curves() -> None:
    fig, ax = plt.subplots(figsize=(4.9, 3.2), constrained_layout=True)
    palette = ["#8172B3", "#C44E52", "#55A868", "#4C72B0", "#64B5CD"]
    for (case, stamp, label), color in zip(KEY_CASES, palette):
        steps, total = load_loss(case, stamp)
        ax.semilogy(steps, total, label=label, lw=1.0, color=color)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Total training loss")
    ax.set_title("Loss convergence of representative cases", loc="left", fontweight="bold")
    ax.legend()
    ax.grid(color="#DDDDDD", lw=0.4)
    save_all(fig, "fig5_loss_curves")


def main() -> None:
    missing = []
    for case, stamp, _label in CASES:
        d = run_dir(case, stamp)
        if not (d / "metrics" / "metrics.json").exists():
            missing.append(str(d))
    if missing:
        raise FileNotFoundError("Missing completed runs:\n" + "\n".join(missing))

    rows = make_summary()
    write_tables(rows)
    figure_result_table(rows)
    figure_metric_overview(rows)
    figure_identifiability_scatter(rows)
    figure_material_maps()
    figure_amplitude_history()
    figure_loss_curves()
    figure_story_summary(rows)
    print(f"FIGURE_IMAGE_DIR={IMAGE_DIR}")
    print(f"FIGURE_CSV_DIR={CSV_DIR}")
    print(f"WROTE_IMAGES={len(list(IMAGE_DIR.glob('*')))}")
    print(f"WROTE_TABLES={len(list(CSV_DIR.glob('*')))}")


if __name__ == "__main__":
    main()
