from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
DEEPXDE_ROOT = REPO_ROOT / "deepxde-master"
if str(DEEPXDE_ROOT) not in sys.path:
    sys.path.append(str(DEEPXDE_ROOT))

from utils.benchmark_plot_utils import generate_pinn_vs_iaminn_v1_architecture


if __name__ == "__main__":
    output = generate_pinn_vs_iaminn_v1_architecture()
    print(output)
