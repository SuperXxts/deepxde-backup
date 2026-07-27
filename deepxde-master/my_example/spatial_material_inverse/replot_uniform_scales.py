import argparse
import hashlib
import os
import sys
from collections import defaultdict

import numpy as np


def resolve_repo_root():
    current = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        current,
        os.path.abspath(os.path.join(current, "..")),
        os.path.abspath(os.path.join(current, "../..")),
    ]
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "utils", "material_inverse_plot_utils.py")):
            return candidate
    raise FileNotFoundError("Cannot resolve repository root from script location.")


ROOT_DIR = resolve_repo_root()
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.data_saving_utils import _get_save_path
from utils.material_inverse_plot_utils import (  # noqa: E402
    plot_field_triplet,
    plot_k_mu_truth_prediction_comparison,
    plot_material_overlay,
)


FIELD_NAMES = ["ux", "uy", "sxx", "syy", "sxy", "lambda", "mu"]
FIELD_TITLES = {
    "ux": "evaluation_ux",
    "uy": "evaluation_uy",
    "sxx": "evaluation_sxx",
    "syy": "evaluation_syy",
    "sxy": "evaluation_sxy",
    "lambda": "Lambda",
    "mu": "Shear modulus Mu",
    "bulk": "Bulk modulus K",
}
LOAD_MODE_CN = {
    "normal_to_layer": "垂向穿层",
    "bending_y": "y向弯曲",
    "biaxial_bulk": "双轴体积",
    "pure_shear": "纯剪切",
    "uniaxial_y": "y向单轴",
    "uniaxial_x": "x向单轴",
    "cross_layer_shear": "跨层剪切",
}


def infer_grid(points):
    points = np.asarray(points, dtype=float)
    xs = np.unique(np.round(points[:, 0], decimals=12))
    ys = np.unique(np.round(points[:, 1], decimals=12))
    xs.sort()
    ys.sort()
    return np.meshgrid(xs, ys)


def reshape(values, shape):
    return np.asarray(values, dtype=float).reshape(shape)


def bulk_from_lambda_mu(lambda_grid, mu_grid):
    return np.asarray(lambda_grid, dtype=float) + (2.0 / 3.0) * np.asarray(mu_grid, dtype=float)


def load_run(run_dir):
    path = os.path.join(run_dir, "npz", "evaluation_grid.npz")
    if not os.path.exists(path):
        return None
    payload = np.load(path, allow_pickle=True)
    points = np.asarray(payload["points"], dtype=float)
    xx = np.asarray(payload["xx"], dtype=float) if "xx" in payload.files else infer_grid(points)[0]
    yy = np.asarray(payload["yy"], dtype=float) if "yy" in payload.files else infer_grid(points)[1]
    truth = np.asarray(payload["truth"], dtype=float)
    pred = np.asarray(payload["prediction"], dtype=float)
    return {
        "run_dir": run_dir,
        "xx": xx,
        "yy": yy,
        "truth": truth,
        "pred": pred,
        "shape": xx.shape,
    }


def iter_run_dirs(root):
    for dirpath, _, filenames in os.walk(root):
        if "evaluation_grid.npz" in filenames and os.path.basename(dirpath) == "npz":
            run_dir = os.path.dirname(dirpath)
            parts = {part.lower() for part in os.path.normpath(run_dir).split(os.sep)}
            if parts.intersection({"last_model_eval"}):
                continue
            if any("smoke" in part or "debug" in part or part.startswith("_archived") or part.startswith("failed") for part in parts):
                continue
            yield run_dir


def truth_hash(truth):
    rounded = np.ascontiguousarray(np.round(np.asarray(truth, dtype=float), 12))
    return hashlib.sha256(rounded.tobytes()).hexdigest()


def update_minmax(limits, key, values):
    values = np.asarray(values, dtype=float)
    current = limits.get(key)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if current is None:
        limits[key] = [vmin, vmax]
    else:
        current[0] = min(current[0], vmin)
        current[1] = max(current[1], vmax)


def collect_limits(runs):
    value_limits = {}
    error_limits = {}
    load_value_limits = {}
    load_error_limits = {}
    for run in runs:
        shape = run["shape"]
        truth = run["truth"]
        pred = run["pred"]
        for index, field_name in enumerate(FIELD_NAMES):
            true_grid = reshape(truth[:, index], shape)
            pred_grid = reshape(pred[:, index], shape)
            update_minmax(value_limits, field_name, np.concatenate([true_grid.ravel(), pred_grid.ravel()]))
            update_minmax(error_limits, field_name, np.abs(pred_grid - true_grid))

        bulk_truth = bulk_from_lambda_mu(reshape(truth[:, 5], shape), reshape(truth[:, 6], shape))
        bulk_pred = bulk_from_lambda_mu(reshape(pred[:, 5], shape), reshape(pred[:, 6], shape))
        update_minmax(value_limits, "bulk", np.concatenate([bulk_truth.ravel(), bulk_pred.ravel()]))
        update_minmax(error_limits, "bulk", np.abs(bulk_pred - bulk_truth))

        for load_path in sorted(
            os.path.join(run["run_dir"], "npz", name)
            for name in os.listdir(os.path.join(run["run_dir"], "npz"))
            if name.startswith("evaluation_l") and (name.endswith("_grid.npz") or name.endswith("_data.npz"))
        ):
            load_key = os.path.basename(load_path).replace("evaluation_", "")
            load_key = load_key.replace("_grid.npz", "").replace("_data.npz", "")
            data = np.load(load_path, allow_pickle=True)
            load_truth = np.asarray(data["truth"] if "truth" in data.files else data["y_true"], dtype=float)
            load_pred = np.asarray(data["prediction"] if "prediction" in data.files else data["y_pred"], dtype=float)
            for index, field_name in enumerate(FIELD_NAMES[:5]):
                true_grid = reshape(load_truth[:, index], shape)
                pred_grid = reshape(load_pred[:, index], shape)
                key = (load_key, field_name)
                update_minmax(load_value_limits, key, np.concatenate([true_grid.ravel(), pred_grid.ravel()]))
                update_minmax(load_error_limits, key, np.abs(pred_grid - true_grid))
    return value_limits, error_limits, load_value_limits, load_error_limits


def load_mode_title(load_key, field_name):
    parts = load_key.split("_", 1)
    if len(parts) == 2 and parts[0].startswith("l"):
        mode = parts[1]
        index = parts[0][1:]
    else:
        mode = load_key
        index = "?"
    mode_cn = LOAD_MODE_CN.get(mode, mode)
    return f"evaluation_{field_name} ({mode})", f"evaluation_l{index}_{mode}_{field_name}"


def replot_run(run, value_limits, error_limits, load_value_limits, load_error_limits):
    run_dir = run["run_dir"]
    xx = run["xx"]
    yy = run["yy"]
    shape = run["shape"]
    truth = run["truth"]
    pred = run["pred"]
    generated = []

    for index, field_name in enumerate(FIELD_NAMES[:5]):
        generated.append(
            plot_field_triplet(
                run_dir,
                xx,
                yy,
                reshape(truth[:, index], shape),
                reshape(pred[:, index], shape),
                field_name,
                title_name=FIELD_TITLES[field_name],
                file_stem=f"evaluation_{field_name}",
                shared_scale=True,
                value_limits=value_limits[field_name],
                error_limits=[0.0, error_limits[field_name][1]],
            )
        )

    lambda_truth = reshape(truth[:, 5], shape)
    lambda_pred = reshape(pred[:, 5], shape)
    mu_truth = reshape(truth[:, 6], shape)
    mu_pred = reshape(pred[:, 6], shape)
    bulk_truth = bulk_from_lambda_mu(lambda_truth, mu_truth)
    bulk_pred = bulk_from_lambda_mu(lambda_pred, mu_pred)

    for field_name, true_grid, pred_grid in [
        ("bulk", bulk_truth, bulk_pred),
        ("mu", mu_truth, mu_pred),
    ]:
        generated.append(
            plot_field_triplet(
                run_dir,
                xx,
                yy,
                true_grid,
                pred_grid,
                field_name,
                title_name=FIELD_TITLES[field_name],
                file_stem=f"evaluation_{field_name}",
                value_limits=value_limits[field_name],
                error_limits=[0.0, error_limits[field_name][1]],
            )
        )
        generated.append(
            plot_material_overlay(
                run_dir,
                xx,
                yy,
                true_grid,
                pred_grid,
                field_name,
                title_name=FIELD_TITLES[field_name],
                file_stem=f"evaluation_{field_name}",
                value_limits=value_limits[field_name],
            )
        )

    generated.append(
        plot_k_mu_truth_prediction_comparison(
            save_dir=run_dir,
            xx=xx,
            yy=yy,
            bulk_truth_grid=bulk_truth,
            bulk_pred_grid=bulk_pred,
            mu_truth_grid=mu_truth,
            mu_pred_grid=mu_pred,
            filename="evaluation_K与μ真值预测对比图.png",
            bulk_value_limits=value_limits["bulk"],
            mu_value_limits=value_limits["mu"],
            bulk_error_limits=[0.0, error_limits["bulk"][1]],
            mu_error_limits=[0.0, error_limits["mu"][1]],
        )
    )

    npz_dir = os.path.join(run_dir, "npz")
    for name in sorted(os.listdir(npz_dir)):
        if not (name.startswith("evaluation_l") and (name.endswith("_grid.npz") or name.endswith("_data.npz"))):
            continue
        load_key = name.replace("evaluation_", "")
        load_key = load_key.replace("_grid.npz", "").replace("_data.npz", "")
        data = np.load(os.path.join(npz_dir, name), allow_pickle=True)
        load_truth = np.asarray(data["truth"] if "truth" in data.files else data["y_true"], dtype=float)
        load_pred = np.asarray(data["prediction"] if "prediction" in data.files else data["y_pred"], dtype=float)
        for index, field_name in enumerate(FIELD_NAMES[:5]):
            title_name, file_stem = load_mode_title(load_key, field_name)
            key = (load_key, field_name)
            generated.append(
                plot_field_triplet(
                    run_dir,
                    xx,
                    yy,
                    reshape(load_truth[:, index], shape),
                    reshape(load_pred[:, index], shape),
                    field_name,
                    title_name=title_name,
                    file_stem=file_stem,
                    shared_scale=True,
                    value_limits=load_value_limits[key],
                    error_limits=[0.0, load_error_limits[key][1]],
                )
            )

    return generated


def main():
    parser = argparse.ArgumentParser(description="Replot evaluation figures with uniform color scales across comparable runs.")
    parser.add_argument("root", help="Root directory containing completed run directories.")
    args = parser.parse_args()

    run_dirs = sorted(iter_run_dirs(args.root))
    runs = [load_run(path) for path in run_dirs]
    runs = [run for run in runs if run is not None]
    if not runs:
        raise SystemExit(f"No completed runs with evaluation_grid.npz found under {args.root}")

    grouped = defaultdict(list)
    for run in runs:
        # Runs with the same grid and full truth field are visually comparable.
        key = (run["shape"], truth_hash(run["truth"]))
        grouped[key].append(run)

    total = 0
    for group_runs in grouped.values():
        value_limits, error_limits, load_value_limits, load_error_limits = collect_limits(group_runs)
        for run in group_runs:
            replot_run(run, value_limits, error_limits, load_value_limits, load_error_limits)
            total += 1

    print(f"Replotted {total} runs in {len(grouped)} comparable group(s).")


if __name__ == "__main__":
    main()
