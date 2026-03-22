from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
DEEPXDE_ROOT = REPO_ROOT / "deepxde-master"
if str(DEEPXDE_ROOT) not in sys.path:
    sys.path.append(str(DEEPXDE_ROOT))

from utils.benchmark_plot_utils import default_benchmark_images_root, generate_paper_images


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate benchmark and paper-ready images")
    parser.add_argument("--case", choices=["layered", "single_inclusion", "double_inclusion"], default="single_inclusion")
    parser.add_argument("--exp_dir", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=str(default_benchmark_images_root()))
    args = parser.parse_args()

    output_root = generate_paper_images(case_name=args.case, exp_dir=args.exp_dir, output_root=args.output_root)
    print(f"Images saved to: {output_root}")
