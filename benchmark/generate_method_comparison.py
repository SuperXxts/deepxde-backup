from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
DEEPXDE_ROOT = REPO_ROOT / "deepxde-master"
if str(DEEPXDE_ROOT) not in sys.path:
    sys.path.append(str(DEEPXDE_ROOT))

from utils.benchmark_plot_utils import generate_method_comparison


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate direct comparison figures for a baseline and a candidate method")
    parser.add_argument("--case", required=True)
    parser.add_argument("--baseline_name", default="PINN")
    parser.add_argument("--method_name", default="Candidate method")
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--method_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    generate_method_comparison(
        case_name=args.case,
        baseline_name=args.baseline_name,
        method_name=args.method_name,
        baseline_dir=args.baseline_dir,
        method_dir=args.method_dir,
        output_dir=args.output_dir,
    )
    print(f"Comparison figures saved to: {args.output_dir}")
