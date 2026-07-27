import argparse
import json
import os
import time

import numpy as np

from fem_visualization import render_teacher_convergence_visuals


FIELD_NAMES = ["ux", "uy", "sxx", "syy", "sxy", "lambda", "mu"]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path, payload):
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, dict):
            return {key: to_serializable(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_serializable(item) for item in obj]
        return obj

    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, indent=2, ensure_ascii=False)


def relative_l2(pred, ref):
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    denom = np.linalg.norm(ref.reshape(-1))
    if denom <= 1e-12:
        return float(np.linalg.norm(pred.reshape(-1)))
    return float(np.linalg.norm((pred - ref).reshape(-1)) / denom)


def wait_for_path(path, poll_seconds, timeout_seconds):
    start = time.time()
    while True:
        if os.path.exists(path):
            return
        if time.time() - start > float(timeout_seconds):
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(float(poll_seconds))


def load_teacher_bundle(run_dir):
    path = os.path.join(run_dir, "npz", "teacher_bundle.npz")
    bundle = np.load(path)
    return {
        "path": path,
        "field_names": [str(x) for x in bundle["field_names"].tolist()],
        "load_modes": [str(x) for x in bundle["load_modes"].tolist()],
        "load_scales": [float(x) for x in bundle["load_scales"].tolist()],
        "evaluation_grid_points": np.asarray(bundle["evaluation_grid_points"], dtype=float),
        "evaluation_grid_true_state": np.asarray(bundle["evaluation_grid_true_state"], dtype=float),
    }


def summarize_pair(name, run_a, run_ref):
    state_a = np.asarray(run_a["evaluation_grid_true_state"], dtype=float)
    state_ref = np.asarray(run_ref["evaluation_grid_true_state"], dtype=float)
    if state_a.shape != state_ref.shape:
        raise ValueError(f"Shape mismatch: {state_a.shape} vs {state_ref.shape}")

    per_field = {}
    for field_index, field_name in enumerate(FIELD_NAMES):
        per_field[field_name] = relative_l2(state_a[:, :, field_index], state_ref[:, :, field_index])

    per_load = []
    for load_index, load_mode in enumerate(run_ref["load_modes"]):
        pred = state_a[load_index]
        ref = state_ref[load_index]
        per_load.append(
            {
                "load_index": int(load_index),
                "load_mode": str(load_mode),
                "displacement_relative_l2": relative_l2(pred[:, :2], ref[:, :2]),
                "stress_relative_l2": relative_l2(pred[:, 2:5], ref[:, 2:5]),
                "material_relative_l2_lambda_mu": relative_l2(pred[:, 5:7], ref[:, 5:7]),
            }
        )

    return {
        "name": name,
        "run_dir": run_a["run_dir"],
        "reference_run_dir": run_ref["run_dir"],
        "global_relative_l2_by_field": per_field,
        "global_displacement_relative_l2": relative_l2(state_a[:, :, :2], state_ref[:, :, :2]),
        "global_stress_relative_l2": relative_l2(state_a[:, :, 2:5], state_ref[:, :, 2:5]),
        "global_material_relative_l2_lambda_mu": relative_l2(state_a[:, :, 5:7], state_ref[:, :, 5:7]),
        "per_load": per_load,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize FEM mesh convergence using teacher bundles.")
    parser.add_argument("--case_root", required=True)
    parser.add_argument("--coarse_run", required=True)
    parser.add_argument("--medium_run", required=True)
    parser.add_argument("--fine_run", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--poll_seconds", type=float, default=20.0)
    parser.add_argument("--timeout_seconds", type=float, default=7200.0)
    parser.add_argument("--wait", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    case_root = os.path.abspath(args.case_root)
    coarse_dir = os.path.join(case_root, args.coarse_run)
    medium_dir = os.path.join(case_root, args.medium_run)
    fine_dir = os.path.join(case_root, args.fine_run)

    required_paths = [
        os.path.join(coarse_dir, "npz", "teacher_bundle.npz"),
        os.path.join(medium_dir, "npz", "teacher_bundle.npz"),
        os.path.join(fine_dir, "npz", "teacher_bundle.npz"),
    ]
    if args.wait:
        for path in required_paths:
            wait_for_path(path, args.poll_seconds, args.timeout_seconds)

    coarse = load_teacher_bundle(coarse_dir)
    medium = load_teacher_bundle(medium_dir)
    fine = load_teacher_bundle(fine_dir)
    coarse["run_dir"] = coarse_dir
    medium["run_dir"] = medium_dir
    fine["run_dir"] = fine_dir

    summary = {
        "case_root": case_root,
        "field_names": FIELD_NAMES,
        "reference_run": fine_dir,
        "coarse_vs_fine": summarize_pair("coarse_vs_fine", coarse, fine),
        "medium_vs_fine": summarize_pair("medium_vs_fine", medium, fine),
    }
    save_json(args.output_path, summary)
    try:
        render_teacher_convergence_visuals(args.output_path)
    except Exception as exc:
        print(f"[warn] failed to render FEM convergence visuals: {exc}")
    print("=" * 80)
    print("FEM convergence summary finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
