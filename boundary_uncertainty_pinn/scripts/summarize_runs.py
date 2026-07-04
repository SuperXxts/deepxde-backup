#!/usr/bin/env python3
"""Summarize run metrics into CSV and Markdown tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-root", type=Path, default=Path("exp"))
    parser.add_argument("--out-csv", type=Path, default=Path("exp/07.ComparisonSummary/run_metrics.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("exp/07.ComparisonSummary/run_metrics.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(args.exp_root.glob("**/metrics.json")):
        if "_archive_torch_prototype" in path.parts:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rel = data.get("relative_l2", {})
        rows.append(
            {
                "run_dir": str(path.parents[1]),
                "case": data.get("case", ""),
                "iterations": data.get("iterations", ""),
                "final_A": data.get("final_A", ""),
                "true_A": data.get("true_A", ""),
                "E_rel_l2": rel.get("E", ""),
                "ux_rel_l2": rel.get("ux", ""),
                "uy_rel_l2": rel.get("uy", ""),
                "sxx_rel_l2": rel.get("sxx", ""),
                "syy_rel_l2": rel.get("syy", ""),
                "sxy_rel_l2": rel.get("sxy", ""),
                "loss": data.get("final_train_loss_sum", ""),
                "elapsed_seconds": data.get("elapsed_seconds", ""),
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_dir",
        "case",
        "iterations",
        "final_A",
        "true_A",
        "E_rel_l2",
        "ux_rel_l2",
        "uy_rel_l2",
        "sxx_rel_l2",
        "syy_rel_l2",
        "sxy_rel_l2",
        "loss",
        "elapsed_seconds",
    ]
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in fields) + " |")
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} rows to {args.out_csv} and {args.out_md}")


if __name__ == "__main__":
    main()
