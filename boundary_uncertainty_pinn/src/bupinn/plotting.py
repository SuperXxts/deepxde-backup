"""Plotting helpers for publication-grade diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


FIELD_LABELS = {
    "ux": r"Horizontal displacement, $u_x$",
    "uy": r"Vertical displacement, $u_y$",
    "sxx": r"Normal stress, $\sigma_{xx}$",
    "syy": r"Normal stress, $\sigma_{yy}$",
    "sxy": r"Shear stress, $\sigma_{xy}$",
    "K": r"Bulk modulus, $K$",
    "mu": r"Shear modulus, $\mu$",
    "E": r"Young's modulus, $E$",
    "nu": r"Poisson's ratio, $\nu$",
}

LOSS_LABELS = {
    "momentum_x": r"Equilibrium residual, $x$",
    "momentum_y": r"Equilibrium residual, $y$",
    "constitutive_sxx": r"Constitutive residual, $\sigma_{xx}$",
    "constitutive_syy": r"Constitutive residual, $\sigma_{yy}$",
    "constitutive_sxy": r"Constitutive residual, $\sigma_{xy}$",
    "bottom_ux": r"Bottom displacement, $u_x$",
    "bottom_uy": r"Bottom displacement, $u_y$",
    "top_ux": r"Top displacement, $u_x$",
    "top_uy": r"Top displacement, $u_y$",
    "obs_u": "Interior displacement observations",
    "obs_material_ux": r"Material-projected observation, $u_x$",
    "obs_material_uy": r"Material-projected observation, $u_y$",
    "obs_boundary_uy": r"Boundary-projected observation, $u_y$",
    "anchor_u": "Boundary displacement anchors",
    "global_reaction_y": "Resultant reaction force",
}


def _field_label(name: str) -> str:
    return FIELD_LABELS.get(name, name)


def _loss_label(name: str) -> str:
    return LOSS_LABELS.get(name, name.replace("_", " "))


def save_figure_all(fig, out: Path, dpi: int = 450) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    if out.suffix.lower() != ".pdf":
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")


def _loss_groups(loss_terms: list[str], loss_train: np.ndarray) -> list[tuple[str, np.ndarray]]:
    groups: list[tuple[str, list[int]]] = [
        ("Total loss", list(range(loss_train.shape[1]))),
        ("Equilibrium residual", [i for i, n in enumerate(loss_terms) if n.startswith("momentum_")]),
        ("Constitutive consistency", [i for i, n in enumerate(loss_terms) if n.startswith("constitutive_")]),
        (
            "Boundary condition",
            [
                i
                for i, n in enumerate(loss_terms)
                if n.startswith("bottom_") or n.startswith("top_") or n.startswith("anchor_")
            ],
        ),
        ("Observation mismatch", [i for i, n in enumerate(loss_terms) if n.startswith("obs_")]),
        ("Reaction constraint", [i for i, n in enumerate(loss_terms) if n.startswith("global_reaction")]),
    ]
    out: list[tuple[str, np.ndarray]] = []
    for label, idx in groups:
        if idx:
            out.append((label, np.sum(loss_train[:, idx], axis=1)))
    return out


def plot_loss_arrays(
    steps: np.ndarray,
    loss_train: np.ndarray,
    loss_test: np.ndarray | None,
    loss_terms: list[str],
    out_history: Path,
    out_components: Path | None = None,
    data_out: Path | None = None,
) -> None:
    steps = np.asarray(steps, dtype=float)
    loss_train = np.asarray(loss_train, dtype=float)
    loss_test = np.asarray(loss_test, dtype=float) if loss_test is not None else np.empty((0, 0))
    if data_out is not None:
        np.savez(data_out, steps=steps, loss_train=loss_train, loss_test=loss_test, loss_terms=np.asarray(loss_terms))
    if loss_train.ndim != 2 or loss_train.shape[0] == 0:
        return

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for label, values in _loss_groups(loss_terms, loss_train):
        ax.semilogy(steps, values, label=label, lw=1.7)
    ax.set_title("Loss history")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Weighted mean-squared loss")
    ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    save_figure_all(fig, out_history)
    plt.close(fig)

    if out_components is not None:
        fig, ax = plt.subplots(figsize=(9.4, 5.2))
        for i in range(loss_train.shape[1]):
            label = _loss_label(loss_terms[i]) if i < len(loss_terms) else f"Loss term {i + 1}"
            ax.semilogy(steps, loss_train[:, i], label=label, lw=1.1)
        ax.set_title("Individual loss components")
        ax.set_xlabel("Training iteration")
        ax.set_ylabel("Weighted mean-squared loss")
        ax.grid(False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
        save_figure_all(fig, out_components)
        plt.close(fig)


def plot_loss_history(losshistory, out: Path, data_out: Path | None = None, loss_terms: list[str] | None = None) -> None:
    steps = np.asarray(losshistory.steps)
    loss_train = np.asarray(losshistory.loss_train, dtype=float)
    loss_test = np.asarray(losshistory.loss_test, dtype=float)
    labels = loss_terms or [f"loss_{i + 1}" for i in range(loss_train.shape[1] if loss_train.ndim == 2 else 0)]
    plot_loss_arrays(steps, loss_train, loss_test, labels, out, out.parent / "损失分量图.png", data_out)


def plot_material_error_history(rows: np.ndarray, out: Path) -> None:
    if rows.size == 0:
        return
    data = np.asarray(rows, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    labels = [r"$K$", r"$\mu$", r"$E$", r"$\nu$"]
    for i, label in enumerate(labels, start=1):
        if data.shape[1] > i:
            ax.plot(data[:, 0], data[:, i], lw=1.5, label=label)
    ax.set_title("Material-parameter error evolution")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(r"Relative $L_2$ error on validation points")
    ax.grid(False)
    ax.legend(loc="best", frameon=True, framealpha=0.92)
    save_figure_all(fig, out)
    plt.close(fig)


def plot_amplitude_history(path: Path, out: Path, true_value: float = 1.0) -> None:
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(data[:, 0], data[:, 1], lw=1.6, label="Inferred boundary amplitude")
    ax.axhline(true_value, color="black", ls="--", lw=1.1, label="Reference amplitude")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(r"Boundary amplitude, $A$")
    ax.set_title("Boundary-parameter evolution")
    ax.grid(False)
    ax.legend(loc="best", frameon=True, framealpha=0.92)
    save_figure_all(fig, out)
    plt.close(fig)


def plot_sampling(obs: np.ndarray, anchors: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.1, 5.0))
    if len(obs):
        ax.scatter(obs[:, 0], obs[:, 1], s=9, label="Interior displacement observations", alpha=0.85)
    if len(anchors):
        ax.scatter(anchors[:, 0], anchors[:, 1], s=42, marker="s", label="Boundary displacement anchors")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title("Training data layout")
    ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=1, frameon=False)
    save_figure_all(fig, out)
    plt.close(fig)


def _grid_field(x: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.unique(x[:, 0])
    ys = np.unique(x[:, 1])
    nx, ny = len(xs), len(ys)
    if nx * ny != len(x):
        raise ValueError("field plotting expects a structured grid")
    return xs, ys, v.reshape(ny, nx)


def _levels(a: np.ndarray, b: np.ndarray | None = None, n: int = 80) -> np.ndarray | int:
    if b is None:
        vmin, vmax = float(np.nanmin(a)), float(np.nanmax(a))
    else:
        vmin = min(float(np.nanmin(a)), float(np.nanmin(b)))
        vmax = max(float(np.nanmax(a)), float(np.nanmax(b)))
    if np.isclose(vmin, vmax):
        return n
    return np.linspace(vmin, vmax, n)


def _contour(ax, xs: np.ndarray, ys: np.ndarray, z: np.ndarray, levels, cmap: str):
    return ax.contourf(xs, ys, z, levels=levels, cmap=cmap, extend="both")


def _format_field_axes(axes: np.ndarray) -> None:
    axes_arr = np.asarray(axes)
    if axes_arr.ndim == 1:
        axes_arr = axes_arr.reshape(1, -1)
    for row in range(axes_arr.shape[0]):
        for col in range(axes_arr.shape[1]):
            ax = axes_arr[row, col]
            ax.set_xlabel(r"$x$")
            if col == 0:
                ax.set_ylabel(r"$y$")
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)
            ax.set_aspect("equal")


def plot_field_triplet(x: np.ndarray, true: np.ndarray, pred: np.ndarray, name: str, out: Path) -> None:
    xs, ys, zt = _grid_field(x, true)
    _, _, zp = _grid_field(x, pred)
    ze = np.abs(zp - zt)
    shared_levels = _levels(zt, zp)
    err_levels = _levels(ze)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), constrained_layout=True)
    im0 = _contour(axes[0], xs, ys, zt, shared_levels, "viridis")
    axes[0].set_title("Reference")
    _contour(axes[1], xs, ys, zp, shared_levels, "viridis")
    axes[1].set_title("Prediction")
    im2 = _contour(axes[2], xs, ys, ze, err_levels, "magma")
    axes[2].set_title("Absolute error")
    _format_field_axes(axes)
    fig.suptitle(_field_label(name))
    fig.colorbar(im0, ax=axes[:2], shrink=0.78)
    fig.colorbar(im2, ax=axes[2], shrink=0.78)
    save_figure_all(fig, out)
    plt.close(fig)


def plot_k_mu_comparison(x: np.ndarray, truth: np.ndarray, pred: np.ndarray, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.4), constrained_layout=True)
    for row, (idx, label) in enumerate([(5, "K"), (6, "mu")]):
        xs, ys, zt = _grid_field(x, truth[:, idx])
        _, _, zp = _grid_field(x, pred[:, idx])
        ze = np.abs(zp - zt)
        shared_levels = _levels(zt, zp)
        err_levels = _levels(ze)
        im0 = _contour(axes[row, 0], xs, ys, zt, shared_levels, "viridis")
        axes[row, 0].set_title(f"{_field_label(label)}: reference")
        _contour(axes[row, 1], xs, ys, zp, shared_levels, "viridis")
        axes[row, 1].set_title(f"{_field_label(label)}: prediction")
        im2 = _contour(axes[row, 2], xs, ys, ze, err_levels, "magma")
        axes[row, 2].set_title(f"{_field_label(label)}: absolute error")
        fig.colorbar(im0, ax=axes[row, :2], shrink=0.76)
        fig.colorbar(im2, ax=axes[row, 2], shrink=0.76)
    _format_field_axes(axes)
    save_figure_all(fig, out)
    plt.close(fig)


def plot_displacement_vector(x: np.ndarray, truth: np.ndarray, pred: np.ndarray, out: Path) -> None:
    xs, ys, _ = _grid_field(x, truth[:, 0])
    x_grid, y_grid = np.meshgrid(xs, ys)
    ux_t = truth[:, 0].reshape(len(ys), len(xs))
    uy_t = truth[:, 1].reshape(len(ys), len(xs))
    ux_p = pred[:, 0].reshape(len(ys), len(xs))
    uy_p = pred[:, 1].reshape(len(ys), len(xs))
    step = max(1, len(xs) // 25)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    axes[0].quiver(x_grid[::step, ::step], y_grid[::step, ::step], ux_t[::step, ::step], uy_t[::step, ::step])
    axes[0].set_title("Reference displacement field")
    axes[1].quiver(x_grid[::step, ::step], y_grid[::step, ::step], ux_p[::step, ::step], uy_p[::step, ::step])
    axes[1].set_title("Predicted displacement field")
    for ax in axes:
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_aspect("equal")
        ax.grid(False)
    save_figure_all(fig, out)
    plt.close(fig)


def plot_line_slice(x: np.ndarray, fields: dict[str, tuple[np.ndarray, np.ndarray]], out: Path, y_value: float = 0.5) -> None:
    y = x[:, 1]
    mask = np.isclose(y, y_value)
    if not np.any(mask):
        idx = np.argsort(np.abs(y - y_value))[: int(np.sqrt(len(x)))]
        mask = np.zeros(len(x), dtype=bool)
        mask[idx] = True
    order = np.argsort(x[mask, 0])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for name, (truth, pred) in fields.items():
        xx = x[mask, 0][order]
        ax.plot(xx, truth[mask][order], lw=1.6, label=f"{_field_label(name)} reference")
        ax.plot(xx, pred[mask][order], "--", lw=1.5, label=f"{_field_label(name)} prediction")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel("Field value")
    ax.set_title(rf"Line-slice comparison at $y={y_value:.2f}$")
    ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    save_figure_all(fig, out)
    plt.close(fig)


def plot_boundary_curve(x: np.ndarray, true: np.ndarray, pred: np.ndarray, name: str, out: Path) -> None:
    order = np.argsort(x[:, 0])
    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    ax.plot(x[order, 0], true[order], lw=1.6, label="Reference")
    ax.plot(x[order, 0], pred[order], "--", lw=1.5, label="Prediction")
    ax.set_xlabel(r"Top-boundary coordinate, $x$")
    ax.set_ylabel("Response")
    ax.set_title(name)
    ax.grid(False)
    ax.legend(loc="best", frameon=True, framealpha=0.92)
    save_figure_all(fig, out)
    plt.close(fig)
