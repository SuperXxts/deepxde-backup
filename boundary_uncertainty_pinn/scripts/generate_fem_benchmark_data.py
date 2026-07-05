#!/usr/bin/env python3
"""Generate and inspect the FEM benchmark data used by B-group runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--ny", type=int, default=48)
    parser.add_argument("--width", type=float, default=6.0)
    parser.add_argument("--depth", type=float, default=3.0)
    parser.add_argument("--plate-width", type=float, default=1.0)
    parser.add_argument("--settlement", type=float, default=-0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-ref", type=float, default=1.0)
    parser.add_argument("--mu-ref", type=float, default=0.45)
    return parser.parse_args()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from bupinn.fem2d import FEMConfig, save_fem_result, solve_fem

    args = parse_args()
    config = FEMConfig(
        width=args.width,
        depth=args.depth,
        plate_width=args.plate_width,
        nx=args.nx,
        ny=args.ny,
        settlement=args.settlement,
        seed=args.seed,
        k_ref=args.k_ref,
        mu_ref=args.mu_ref,
    )
    result = solve_fem(config)
    save_fem_result(result, args.out_dir)
    print("FEM_DATA_COMPLETE", args.out_dir)
    print("num_nodes", result.nodes.shape[0])
    print("num_elements", result.elements.shape[0])
    print("num_plate_nodes", result.plate_nodes.shape[0])
    print("total_reaction_y", result.total_reaction_y)


if __name__ == "__main__":
    main()
