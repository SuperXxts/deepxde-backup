#!/usr/bin/env python3
"""Diagnose local K-mu excitation supplied by the manufactured load modes.

The diagnostic is deliberately separate from training.  For plane strain,
lambda = K - 2/3 mu, so the coefficient matrix below maps local (K, mu)
perturbations to the three stress components for each prescribed strain
field.  Its rank does not prove global inverse uniqueness, but it directly
tests the affine pure-dilation degeneracy raised by the reviewer.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def _load_project_helpers():
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from my_example.spatial_material_inverse.shared import (  # pylint: disable=import-outside-toplevel
        exact_strain_numpy,
        get_case_config,
    )

    return exact_strain_numpy, get_case_config


def build_points(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(0.0, 1.0, int(nx))
    ys = np.linspace(0.0, 1.0, int(ny))
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack((xx.reshape(-1), yy.reshape(-1))), xx, yy


def stress_sensitivity_matrix(exx: np.ndarray, eyy: np.ndarray, exy: np.ndarray) -> np.ndarray:
    """Return an (n, 3, 2) matrix for [sxx, syy, sxy] vs [K, mu]."""

    trace = exx + eyy
    matrix = np.zeros((trace.size, 3, 2), dtype=float)
    matrix[:, 0, 0] = trace
    matrix[:, 0, 1] = 2.0 * exx - (2.0 / 3.0) * trace
    matrix[:, 1, 0] = trace
    matrix[:, 1, 1] = 2.0 * eyy - (2.0 / 3.0) * trace
    matrix[:, 2, 1] = 2.0 * exy
    return matrix


def summarize_matrix(matrix: np.ndarray, label: str) -> tuple[dict, np.ndarray, np.ndarray]:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    largest = singular_values[:, 0]
    smallest = singular_values[:, 1]
    threshold = np.maximum(largest * 1.0e-10, 1.0e-12)
    rank_two = smallest > threshold
    condition = np.full_like(largest, np.inf)
    valid = smallest > 1.0e-14
    condition[valid] = largest[valid] / smallest[valid]
    finite_condition = condition[np.isfinite(condition)]
    summary = {
        "label": label,
        "rows_per_point": int(matrix.shape[1]),
        "points": int(matrix.shape[0]),
        "rank_two_fraction": float(np.mean(rank_two)),
        "rank_deficient_fraction": float(np.mean(~rank_two)),
        "singular_value_min": float(np.min(smallest)),
        "singular_value_median": float(np.median(smallest)),
        "singular_value_p95": float(np.percentile(smallest, 95.0)),
        "condition_number_median": float(np.median(finite_condition)) if finite_condition.size else None,
        "condition_number_p95": float(np.percentile(finite_condition, 95.0)) if finite_condition.size else None,
        "condition_number_max_finite": float(np.max(finite_condition)) if finite_condition.size else None,
        "infinite_condition_fraction": float(np.mean(~np.isfinite(condition))),
    }
    return summary, singular_values, condition


def affine_counterexample() -> dict:
    """Return the exact rank-one coefficient matrix for exx=eyy=alpha."""

    alpha = 1.0
    trace = 2.0 * alpha
    matrix = np.array(
        [
            [trace, 2.0 * alpha - (2.0 / 3.0) * trace],
            [trace, 2.0 * alpha - (2.0 / 3.0) * trace],
            [0.0, 0.0],
        ],
        dtype=float,
    )
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return {
        "strain": {"exx": alpha, "eyy": alpha, "exy": 0.0},
        "matrix": matrix.tolist(),
        "singular_values": singular_values.tolist(),
        "rank": int(np.linalg.matrix_rank(matrix)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="layered")
    parser.add_argument(
        "--load_modes",
        default="normal_to_layer,bending_y,biaxial_bulk",
        help="Comma-separated modes in the same order used by training.",
    )
    parser.add_argument("--nx", type=int, default=141)
    parser.add_argument("--ny", type=int, default=141)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    exact_strain_numpy, get_case_config = _load_project_helpers()
    case_config = get_case_config(args.case)
    modes = [mode.strip() for mode in str(args.load_modes).split(",") if mode.strip()]
    if not modes:
        raise ValueError("At least one load mode is required.")

    points, xx, yy = build_points(args.nx, args.ny)
    matrices = []
    summaries = []
    pointwise = {"x": points[:, 0], "y": points[:, 1]}
    for mode in modes:
        exx, eyy, exy = exact_strain_numpy(points, case_config, load_mode=mode)
        matrix = stress_sensitivity_matrix(exx.reshape(-1), eyy.reshape(-1), exy.reshape(-1))
        summary, singular_values, condition = summarize_matrix(matrix, mode)
        matrices.append(matrix)
        summaries.append(summary)
        pointwise[f"{mode}_sigma_max"] = singular_values[:, 0]
        pointwise[f"{mode}_sigma_min"] = singular_values[:, 1]
        pointwise[f"{mode}_condition"] = condition

    combined = np.concatenate(matrices, axis=1)
    summary, singular_values, condition = summarize_matrix(combined, "combined:" + "+".join(modes))
    summaries.append(summary)
    pointwise["combined_sigma_max"] = singular_values[:, 0]
    pointwise["combined_sigma_min"] = singular_values[:, 1]
    pointwise["combined_condition"] = condition

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    finite_condition = np.isfinite(pointwise["combined_condition"])
    payload = {
        "case": args.case,
        "load_modes": modes,
        "grid": {"nx": int(args.nx), "ny": int(args.ny), "points": int(len(points))},
        "parameterization": "plane_strain lambda = K - 2/3 mu",
        "interpretation": (
            "The combined local constitutive sensitivity has full column rank where the second singular value is nonzero. "
            "This is a local excitation diagnostic and not a proof of global inverse uniqueness."
        ),
        "affine_counterexample": affine_counterexample(),
        "summaries": summaries,
        "combined_condition_finite_fraction": float(np.mean(finite_condition)),
    }
    with (output_dir / "identifiability_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    fieldnames = list(pointwise.keys())
    with (output_dir / "identifiability_pointwise.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        writer.writerows(zip(*(pointwise[name] for name in fieldnames)))

    # Save a compact grid for plotting without requiring a plotting backend here.
    np.savez(
        output_dir / "identifiability_grid.npz",
        x=xx,
        y=yy,
        combined_condition=pointwise["combined_condition"].reshape(int(args.ny), int(args.nx)),
        combined_sigma_min=pointwise["combined_sigma_min"].reshape(int(args.ny), int(args.nx)),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
