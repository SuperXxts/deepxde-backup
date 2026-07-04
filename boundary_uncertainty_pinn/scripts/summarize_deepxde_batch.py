#!/usr/bin/env python3
"""Summarize one DeepXDE PFNN experiment batch."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CASES = [
    "oracle_A",
    "wrong_fixed_A",
    "learnable_A",
    "learnable_A_anchor",
    "learnable_A_reaction",
    "full_boundary_aware",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def load_rows(root: Path, stamp: str) -> list[dict[str, float | str | int]]:
    rows = []
    for case in CASES:
        run_dir = root / f"mms_smooth_lens_obs225_A085_seed42_{case}_{stamp}"
        metrics_path = run_dir / "metrics" / "metrics.json"
        if not metrics_path.exists():
            rows.append({"case": case, "status": "missing_metrics", "run_dir": str(run_dir)})
            continue
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        rel = metrics.get("relative_l2", {})
        rows.append(
            {
                "case": case,
                "status": "complete",
                "run_dir": str(run_dir),
                "iterations": int(metrics.get("iterations", 0)),
                "elapsed_seconds": float(metrics.get("elapsed_seconds", float("nan"))),
                "best_step": int(metrics.get("best_step", 0)),
                "final_A": float(metrics.get("final_A", float("nan"))),
                "A_abs_error": abs(float(metrics.get("final_A", float("nan"))) - float(metrics.get("true_A", 1.0))),
                "ux_rel_l2": float(rel.get("ux", float("nan"))),
                "uy_rel_l2": float(rel.get("uy", float("nan"))),
                "sxx_rel_l2": float(rel.get("sxx", float("nan"))),
                "syy_rel_l2": float(rel.get("syy", float("nan"))),
                "sxy_rel_l2": float(rel.get("sxy", float("nan"))),
                "K_rel_l2": float(rel.get("K", float("nan"))),
                "mu_rel_l2": float(rel.get("mu", float("nan"))),
                "E_rel_l2": float(rel.get("E", float("nan"))),
                "nu_rel_l2": float(rel.get("nu", float("nan"))),
                "reaction_rel_error": float(metrics.get("reaction_rel_error", float("nan"))),
                "final_train_loss_sum": float(metrics.get("final_train_loss_sum", float("nan"))),
            }
        )
    return rows


def write_csv(rows: list[dict[str, float | str | int]], path: Path) -> None:
    keys = [
        "case",
        "status",
        "iterations",
        "elapsed_seconds",
        "best_step",
        "final_A",
        "A_abs_error",
        "ux_rel_l2",
        "uy_rel_l2",
        "sxx_rel_l2",
        "syy_rel_l2",
        "sxy_rel_l2",
        "K_rel_l2",
        "mu_rel_l2",
        "E_rel_l2",
        "nu_rel_l2",
        "reaction_rel_error",
        "final_train_loss_sum",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def write_markdown(rows: list[dict[str, float | str | int]], path: Path) -> None:
    cols = [
        ("case", "case"),
        ("A", "final_A"),
        ("|A-1|", "A_abs_error"),
        ("K", "K_rel_l2"),
        ("mu", "mu_rel_l2"),
        ("E", "E_rel_l2"),
        ("ux", "ux_rel_l2"),
        ("uy", "uy_rel_l2"),
        ("reaction", "reaction_rel_error"),
        ("loss", "final_train_loss_sum"),
    ]
    lines = []
    lines.append("| " + " | ".join(c[0] for c in cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in rows:
        vals = []
        for _, key in cols:
            value = row.get(key, "")
            if isinstance(value, float):
                vals.append(f"{value:.6g}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(rows: list[dict[str, float | str | int]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    complete = [r for r in rows if r.get("status") == "complete"]
    labels = [str(r["case"]).replace("learnable_", "learn_").replace("full_boundary_aware", "full") for r in complete]

    def bar(keys: list[str], title: str, filename: str) -> None:
        x = range(len(labels))
        width = 0.8 / len(keys)
        fig, ax = plt.subplots(figsize=(11, 4.2), dpi=180)
        for idx, key in enumerate(keys):
            values = [float(r.get(key, float("nan"))) for r in complete]
            offsets = [i - 0.4 + width / 2 + idx * width for i in x]
            ax.bar(offsets, values, width=width, label=key.replace("_rel_l2", "").replace("_abs_error", ""))
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("relative error")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, ncols=min(4, len(keys)))
        fig.tight_layout()
        fig.savefig(out_dir / filename)
        plt.close(fig)

    bar(["K_rel_l2", "mu_rel_l2", "E_rel_l2"], "Material inversion errors", "bar_material_errors.png")
    bar(["ux_rel_l2", "uy_rel_l2"], "Displacement errors", "bar_displacement_errors.png")
    bar(["sxx_rel_l2", "syy_rel_l2", "sxy_rel_l2"], "Stress errors", "bar_stress_errors.png")
    bar(["A_abs_error", "reaction_rel_error"], "Boundary amplitude and reaction errors", "bar_boundary_errors.png")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.root / f"summary_{args.stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.root, args.stamp)
    write_csv(rows, out_dir / "summary_metrics.csv")
    write_markdown(rows, out_dir / "summary_metrics.md")
    write_plots(rows, out_dir)
    print(f"WROTE {out_dir}")


if __name__ == "__main__":
    main()
