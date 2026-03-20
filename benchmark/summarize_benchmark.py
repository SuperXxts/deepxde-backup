import csv
import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "benchmark" / "runs"
SUMMARY_ROOT = REPO_ROOT / "benchmark" / "summary"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_metrics(task_name, run_name, save_dir):
    row = {
        "task_name": task_name,
        "run_name": run_name,
        "save_dir": str(save_dir),
    }

    metrics_file = save_dir / "metrics" / "accuracy_metrics.json"
    best_loss_file = save_dir / "json" / "best_test_loss.json"

    if best_loss_file.exists():
        best_loss = load_json(best_loss_file)
        row["best_test_loss"] = best_loss.get("best_test_loss")
        row["best_step"] = best_loss.get("best_step")

    if metrics_file.exists():
        metrics = load_json(metrics_file)
        for field_name, values in metrics.items():
            for metric_name, metric_value in values.items():
                row[f"{field_name}.{metric_name}"] = metric_value
    return row


def main():
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []

    if not RUNS_ROOT.exists():
        print("No benchmark runs found.")
        return

    for task_dir in sorted([p for p in RUNS_ROOT.iterdir() if p.is_dir()]):
        task_name = task_dir.name
        for run_dir in sorted([p for p in task_dir.iterdir() if p.is_dir()]):
            rows.append(flatten_metrics(task_name, run_dir.name, run_dir))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = SUMMARY_ROOT / f"benchmark_summary_{timestamp}.json"
    csv_path = SUMMARY_ROOT / f"benchmark_summary_{timestamp}.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    all_keys = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved summary json to {json_path}")
    print(f"Saved summary csv to {csv_path}")


if __name__ == "__main__":
    main()
