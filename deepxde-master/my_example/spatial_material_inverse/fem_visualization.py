import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIELD_NAMES = ["ux", "uy", "sxx", "syy", "sxy", "lambda", "mu"]
STATE_FIELD_NAMES = FIELD_NAMES[:5]
FIELD_NAME_CN = {
    "ux": "水平位移",
    "uy": "竖向位移",
    "sxx": "横向正应力",
    "syy": "纵向正应力",
    "sxy": "剪应力",
    "lambda": "拉梅参数",
    "mu": "剪切模量",
    "bulk": "体积模量",
}
LOAD_MODE_CN = {
    "biaxial_bulk": "双轴体积",
    "pure_shear": "纯剪切",
    "uniaxial_y": "纵向单轴",
    "uniaxial_x": "横向单轴",
    "normal_to_layer": "垂向穿层",
    "cross_layer_shear": "跨层剪切",
    "bending_y": "纵向弯曲",
    "top_nonuniform_compression": "顶部非均匀压缩",
    "edge_patch_load": "边界局部载荷",
}


def safe_mode_name(load_mode):
    return str(load_mode).replace(" ", "_").replace("-", "_").lower()


def mode_name_cn(load_mode):
    return LOAD_MODE_CN.get(str(load_mode).strip().lower(), str(load_mode))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _get_save_path(save_dir, subfolder, filename):
    subfolder_path = os.path.join(save_dir, subfolder)
    ensure_dir(subfolder_path)
    return os.path.join(subfolder_path, filename)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_grid(points):
    points = np.asarray(points, dtype=float)
    xs = np.unique(np.round(points[:, 0], decimals=12))
    ys = np.unique(np.round(points[:, 1], decimals=12))
    xs.sort()
    ys.sort()
    xx, yy = np.meshgrid(xs, ys)
    return xx, yy, len(ys), len(xs)


def reshape_field(values, ny, nx):
    return np.asarray(values, dtype=float).reshape(int(ny), int(nx))


def compute_bulk(lambda_grid, mu_grid):
    return np.asarray(lambda_grid, dtype=float) + (2.0 / 3.0) * np.asarray(mu_grid, dtype=float)


def save_scalar_map(save_dir, xx, yy, grid, title, file_stem, cmap="viridis"):
    grid = np.asarray(grid, dtype=float)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    image = ax.contourf(xx, yy, grid, levels=81, cmap=cmap)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.ax.tick_params(labelsize=8)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.tight_layout()
    path = _get_save_path(save_dir, "png", f"{file_stem}.png")
    plt.savefig(path, dpi=300)
    plt.close(fig)
    return path


def save_mesh_plot(save_dir, mesh_points, mesh_triangles, title, file_stem):
    mesh_points = np.asarray(mesh_points, dtype=float)
    mesh_triangles = np.asarray(mesh_triangles, dtype=int)
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.triplot(
        mesh_points[:, 0],
        mesh_points[:, 1],
        mesh_triangles,
        linewidth=0.35,
        color="#334155",
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    plt.tight_layout()
    path = _get_save_path(save_dir, "png", f"{file_stem}.png")
    plt.savefig(path, dpi=300)
    plt.close(fig)
    return path


def save_sampling_layout(
    save_dir,
    xx,
    yy,
    domain_points,
    boundary_points,
    train_points,
    validation_points,
    evaluation_points,
    bulk_grid,
    mu_grid,
    filename,
):
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.8))
    for axis, grid, title in zip(
        axes[:2],
        [bulk_grid, mu_grid],
        ["FEM bulk modulus K", "FEM shear modulus Mu"],
    ):
        image = axis.contourf(xx, yy, grid, levels=81, cmap="viridis")
        fig.colorbar(image, ax=axis)
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_aspect("equal")

    axes[2].scatter(domain_points[:, 0], domain_points[:, 1], s=6, alpha=0.2, color="#64748b", label="Grid")
    if len(boundary_points) > 0:
        axes[2].scatter(boundary_points[:, 0], boundary_points[:, 1], s=14, alpha=0.9, color="#0f172a", marker="x", label="Boundary")
    if len(train_points) > 0:
        axes[2].scatter(train_points[:, 0], train_points[:, 1], s=12, alpha=0.8, color="#ea580c", label="Train")
    if len(validation_points) > 0:
        axes[2].scatter(validation_points[:, 0], validation_points[:, 1], s=14, alpha=0.8, color="#0f766e", marker="^", label="Validation")
    if len(evaluation_points) > 0:
        axes[2].scatter(evaluation_points[:, 0], evaluation_points[:, 1], s=14, alpha=0.8, color="#7c3aed", marker="s", label="Evaluation")
    axes[2].set_title("Sampling layout")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_xlim(0.0, 1.0)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_aspect("equal")
    axes[2].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    plt.tight_layout(rect=(0.0, 0.0, 0.9, 1.0))
    path = _get_save_path(save_dir, "png", filename)
    plt.savefig(path, dpi=300)
    plt.close(fig)
    return path


def save_field_triplet(save_dir, xx, yy, truth_grid, pred_grid, title, file_stem):
    truth_grid = np.asarray(truth_grid, dtype=float)
    pred_grid = np.asarray(pred_grid, dtype=float)
    error_grid = np.abs(pred_grid - truth_grid)
    field_vmin = float(np.nanmin([np.nanmin(truth_grid), np.nanmin(pred_grid)]))
    field_vmax = float(np.nanmax([np.nanmax(truth_grid), np.nanmax(pred_grid)]))
    if np.isclose(field_vmin, field_vmax):
        delta = 1.0 if np.isclose(field_vmin, 0.0) else max(abs(field_vmin) * 1e-6, 1e-8)
        field_vmin -= delta
        field_vmax += delta
    field_levels = np.linspace(field_vmin, field_vmax, 81)
    err_vmin = 0.0
    err_vmax = float(np.nanmax(error_grid)) if error_grid.size else 0.0
    if np.isclose(err_vmin, err_vmax):
        err_vmax = 1e-8
    err_levels = np.linspace(err_vmin, err_vmax, 81)
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6), constrained_layout=True)
    truth_image = axes[0].contourf(xx, yy, truth_grid, levels=field_levels, cmap="viridis", vmin=field_vmin, vmax=field_vmax)
    pred_image = axes[1].contourf(xx, yy, pred_grid, levels=field_levels, cmap="viridis", vmin=field_vmin, vmax=field_vmax)
    err_image = axes[2].contourf(xx, yy, error_grid, levels=err_levels, cmap="magma", vmin=err_vmin, vmax=err_vmax)
    for axis, image, axis_title in zip(
        axes,
        [truth_image, pred_image, err_image],
        [f"True {title}", f"Predicted {title}", f"Absolute error {title}"],
    ):
        fig.colorbar(image, ax=axis)
        axis.set_title(axis_title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_aspect("equal")
    path = _get_save_path(save_dir, "png", f"{file_stem}.png")
    plt.savefig(path, dpi=300)
    plt.close(fig)
    return path


def render_teacher_run_visuals(save_dir):
    bundle_path = os.path.join(save_dir, "npz", "teacher_bundle.npz")
    meta_path = os.path.join(save_dir, "json", "teacher_meta.json")
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Missing teacher bundle: {bundle_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing teacher meta: {meta_path}")

    generated = []
    bundle = np.load(bundle_path, allow_pickle=True)
    meta = load_json(meta_path)
    load_modes = [str(item) for item in bundle["load_modes"].tolist()]
    grid_points = np.asarray(bundle["evaluation_grid_points"], dtype=float)
    xx, yy, ny, nx = infer_grid(grid_points)
    boundary_points = np.asarray(bundle["boundary_points"], dtype=float)
    train_points = np.asarray(bundle["train_points"], dtype=float)
    validation_points = np.asarray(bundle["validation_points"], dtype=float)
    evaluation_points = np.asarray(bundle["evaluation_points"], dtype=float)
    full_grid_state = np.asarray(bundle["evaluation_grid_true_state"], dtype=float)

    reference_grid = np.asarray(full_grid_state[0], dtype=float)
    lambda_grid = reshape_field(reference_grid[:, 5], ny, nx)
    mu_grid = reshape_field(reference_grid[:, 6], ny, nx)
    bulk_grid = compute_bulk(lambda_grid, mu_grid)

    generated.append(
        save_sampling_layout(
            save_dir=save_dir,
            xx=xx,
            yy=yy,
            domain_points=grid_points,
            boundary_points=boundary_points,
            train_points=train_points,
            validation_points=validation_points,
            evaluation_points=evaluation_points,
            bulk_grid=bulk_grid,
            mu_grid=mu_grid,
            filename="有限元采样与材料分布图.png",
        )
    )

    generated.append(save_scalar_map(save_dir, xx, yy, bulk_grid, "FEM bulk modulus K", "有限元体积模量图"))
    generated.append(save_scalar_map(save_dir, xx, yy, lambda_grid, "FEM lambda", "有限元拉梅参数图"))
    generated.append(save_scalar_map(save_dir, xx, yy, mu_grid, "FEM shear modulus Mu", "有限元剪切模量图"))

    for load_index, load_mode in enumerate(load_modes):
        mode_tag = safe_mode_name(load_mode)
        mode_cn = mode_name_cn(load_mode)
        grid_path = os.path.join(save_dir, "npz", f"evaluation_grid_l{load_index}_{mode_tag}.npz")
        mesh_path = os.path.join(save_dir, "npz", f"mesh_solution_l{load_index}_{mode_tag}.npz")
        if not os.path.exists(grid_path):
            continue
        payload = np.load(grid_path, allow_pickle=True)
        local_xx = np.asarray(payload["xx"], dtype=float)
        local_yy = np.asarray(payload["yy"], dtype=float)
        local_truth = np.asarray(payload["truth"], dtype=float)
        local_ny, local_nx = local_xx.shape
        for field_index, field_name in enumerate(STATE_FIELD_NAMES):
            field_grid = reshape_field(local_truth[:, field_index], local_ny, local_nx)
            generated.append(
                save_scalar_map(
                    save_dir,
                    local_xx,
                    local_yy,
                    field_grid,
                    f"FEM {field_name} ({load_mode})",
                    f"工况{load_index}_{mode_cn}_{FIELD_NAME_CN.get(field_name, field_name)}图",
                )
            )
        if os.path.exists(mesh_path):
            mesh_payload = np.load(mesh_path, allow_pickle=True)
            generated.append(
                save_mesh_plot(
                    save_dir,
                    mesh_points=np.asarray(mesh_payload["mesh_points"], dtype=float),
                    mesh_triangles=np.asarray(mesh_payload["mesh_triangles"], dtype=int),
                    title=f"Mesh ({load_mode})",
                    file_stem=f"工况{load_index}_{mode_cn}_网格图",
                )
            )

    index_payload = {
        "save_dir": os.path.abspath(save_dir),
        "case": meta.get("case", {}).get("name"),
        "type": "teacher_run_visuals",
        "generated_png_count": len(generated),
        "generated_png": generated,
    }
    index_path = _get_save_path(save_dir, "json", "fem_visual_index.json")
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump(index_payload, handle, indent=2, ensure_ascii=False)
    return generated


def render_teacher_convergence_visuals(summary_path):
    summary_path = os.path.abspath(summary_path)
    summary = load_json(summary_path)
    save_dir = os.path.dirname(summary_path)
    png_dir = ensure_dir(os.path.join(save_dir, "png"))
    generated = []

    pair_names = ["coarse_vs_fine", "medium_vs_fine"]
    metric_names = [
        "global_displacement_relative_l2",
        "global_stress_relative_l2",
        "global_material_relative_l2_lambda_mu",
    ]
    metric_labels = ["Displacement", "Stress", "Material"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for axis, pair_name in zip(axes, pair_names):
        pair = summary.get(pair_name, {})
        values = [float(pair.get(metric_name, np.nan)) for metric_name in metric_names]
        axis.bar(metric_labels, values, color=["#2563eb", "#f59e0b", "#16a34a"])
        axis.set_title(pair_name.replace("_", " "))
        axis.set_ylabel("Relative L2")
        axis.set_yscale("log")
        axis.grid(True, axis="y", alpha=0.2)
    convergence_path = os.path.join(png_dir, "有限元收敛总览图.png")
    plt.savefig(convergence_path, dpi=300)
    plt.close(fig)
    generated.append(convergence_path)

    for pair_name in pair_names:
        pair = summary.get(pair_name, {})
        field_errors = pair.get("global_relative_l2_by_field", {})
        if not field_errors:
            continue
        fig, ax = plt.subplots(figsize=(9.8, 4.8))
        names = list(field_errors.keys())
        values = [float(field_errors[name]) for name in names]
        ax.bar(names, values, color="#475569")
        ax.set_title(pair_name.replace("_", " "))
        ax.set_ylabel("Relative L2")
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.2)
        plt.tight_layout()
        pair_cn = "粗细网格" if pair_name == "coarse_vs_fine" else "中细网格"
        path = os.path.join(png_dir, f"{pair_cn}_字段误差图.png")
        plt.savefig(path, dpi=300)
        plt.close(fig)
        generated.append(path)

    index_path = os.path.join(save_dir, "json", "fem_convergence_visual_index.json")
    ensure_dir(os.path.dirname(index_path))
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "summary_path": summary_path,
                "type": "teacher_convergence_visuals",
                "generated_png_count": len(generated),
                "generated_png": generated,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    return generated


def render_manufactured_validation_visuals(save_dir):
    save_dir = os.path.abspath(save_dir)
    summary_path = os.path.join(save_dir, "manufactured_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Missing manufactured summary: {summary_path}")

    summary = load_json(summary_path)
    generated = []
    png_dir = ensure_dir(os.path.join(save_dir, "png"))

    per_mesh = summary.get("per_mesh", [])
    mesh_sizes = [int(item["mesh_nx"]) for item in per_mesh]
    displacement_errors = [float(item["global_displacement_relative_l2"]) for item in per_mesh]
    stress_errors = [float(item["global_stress_relative_l2"]) for item in per_mesh]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(mesh_sizes, displacement_errors, marker="o", linewidth=2.0, color="#2563eb", label="Displacement")
    ax.plot(mesh_sizes, stress_errors, marker="s", linewidth=2.0, color="#dc2626", label="Stress")
    ax.set_xlabel("Mesh nx = ny")
    ax.set_ylabel("Relative L2")
    ax.set_yscale("log")
    ax.set_title("Manufactured FEM convergence")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    plt.tight_layout()
    convergence_path = os.path.join(png_dir, "制造解收敛图.png")
    plt.savefig(convergence_path, dpi=300)
    plt.close(fig)
    generated.append(convergence_path)

    for item in per_mesh:
        mesh_dir = os.path.join(save_dir, f"mesh_{int(item['mesh_nx'])}x{int(item['mesh_ny'])}")
        compare_path = os.path.join(mesh_dir, "grid_comparison.npz")
        if not os.path.exists(compare_path):
            continue
        payload = np.load(compare_path, allow_pickle=True)
        xx = np.asarray(payload["xx"], dtype=float)
        yy = np.asarray(payload["yy"], dtype=float)
        truth = np.asarray(payload["truth"], dtype=float)
        prediction = np.asarray(payload["prediction"], dtype=float)
        ny, nx = xx.shape
        mesh_name = f"网格{int(item['mesh_nx'])}乘{int(item['mesh_ny'])}"
        for field_index, field_name in enumerate(STATE_FIELD_NAMES):
            truth_grid = reshape_field(truth[:, field_index], ny, nx)
            pred_grid = reshape_field(prediction[:, field_index], ny, nx)
            generated.append(
                save_field_triplet(
                    save_dir=mesh_dir,
                    xx=xx,
                    yy=yy,
                    truth_grid=truth_grid,
                    pred_grid=pred_grid,
                    title=f"{field_name} ({int(item['mesh_nx'])}x{int(item['mesh_ny'])})",
                    file_stem=f"{mesh_name}_{FIELD_NAME_CN.get(field_name, field_name)}对比图",
                )
            )

    index_path = os.path.join(save_dir, "json", "manufactured_visual_index.json")
    ensure_dir(os.path.dirname(index_path))
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "save_dir": save_dir,
                "type": "manufactured_validation_visuals",
                "generated_png_count": len(generated),
                "generated_png": generated,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    return generated


def discover_targets(auto_root):
    teacher_runs = []
    convergence_summaries = []
    manufactured_runs = []
    auto_root = os.path.abspath(auto_root)
    for dirpath, dirnames, filenames in os.walk(auto_root):
        if "teacher_meta.json" in filenames and os.path.basename(dirpath) == "json":
            teacher_runs.append(os.path.dirname(dirpath))
        for filename in filenames:
            if filename.startswith("convergence_summary_") and filename.endswith(".json"):
                convergence_summaries.append(os.path.join(dirpath, filename))
            if filename == "manufactured_summary.json":
                manufactured_runs.append(dirpath)
    return teacher_runs, convergence_summaries, manufactured_runs


def parse_args():
    parser = argparse.ArgumentParser(description="Render visualization assets for FEM runs.")
    parser.add_argument("--teacher_run", action="append", default=[])
    parser.add_argument("--convergence_summary", action="append", default=[])
    parser.add_argument("--manufactured_run", action="append", default=[])
    parser.add_argument("--auto_root", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    teacher_runs = [os.path.abspath(path) for path in args.teacher_run]
    convergence_summaries = [os.path.abspath(path) for path in args.convergence_summary]
    manufactured_runs = [os.path.abspath(path) for path in args.manufactured_run]

    if args.auto_root:
        auto_teacher, auto_convergence, auto_manufactured = discover_targets(args.auto_root)
        teacher_runs.extend(auto_teacher)
        convergence_summaries.extend(auto_convergence)
        manufactured_runs.extend(auto_manufactured)

    teacher_runs = sorted(set(teacher_runs))
    convergence_summaries = sorted(set(convergence_summaries))
    manufactured_runs = sorted(set(manufactured_runs))

    for path in teacher_runs:
        print(f"[render] teacher_run={path}")
        render_teacher_run_visuals(path)
    for path in convergence_summaries:
        print(f"[render] convergence_summary={path}")
        render_teacher_convergence_visuals(path)
    for path in manufactured_runs:
        print(f"[render] manufactured_run={path}")
        render_manufactured_validation_visuals(path)

    print("=" * 80)
    print("FEM visualization rendering finished.")
    print(
        json.dumps(
            {
                "teacher_runs": teacher_runs,
                "convergence_summaries": convergence_summaries,
                "manufactured_runs": manufactured_runs,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
