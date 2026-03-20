import argparse
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "benchmark" / "runs"


def parse_args():
    parser = argparse.ArgumentParser(description="Run linear elasticity benchmark tasks")
    parser.add_argument("--task", required=True, help="Path to a benchmark task json file")
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_cli_args(arg_dict):
    cli_args = []
    for key, value in arg_dict.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cli_args.append(flag)
        else:
            cli_args.extend([flag, str(value)])
    return cli_args


def collect_run_metrics(save_dir: Path):
    metrics_file = save_dir / "metrics" / "accuracy_metrics.json"
    best_loss_file = save_dir / "json" / "best_test_loss.json"
    payload = {
        "save_dir": str(save_dir),
        "metrics_file": str(metrics_file) if metrics_file.exists() else None,
        "best_test_loss_file": str(best_loss_file) if best_loss_file.exists() else None,
        "accuracy_metrics": None,
        "best_test_loss": None,
    }
    if metrics_file.exists():
        payload["accuracy_metrics"] = load_json(metrics_file)
    if best_loss_file.exists():
        payload["best_test_loss"] = load_json(best_loss_file)
    return payload


def run_one(task_name, script_rel, run_name, arg_dict):
    save_dir = RUNS_ROOT / task_name / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    script_path = REPO_ROOT / script_rel
    run_args = deepcopy(arg_dict)
    run_args["save_dir"] = str(save_dir)

    cmd = [sys.executable, str(script_path)] + build_cli_args(run_args)
    print("=" * 80)
    print(f"Running benchmark: {task_name} / {run_name}")
    print("Command:", " ".join(cmd))
    print("=" * 80)

    started_at = datetime.now().isoformat(timespec="seconds")
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    finished_at = datetime.now().isoformat(timespec="seconds")

    result = {
        "task_name": task_name,
        "run_name": run_name,
        "script": script_rel,
        "command": cmd,
        "returncode": proc.returncode,
        "started_at": started_at,
        "finished_at": finished_at,
    }
    result.update(collect_run_metrics(save_dir))
    return result


def main():
    args = parse_args()
    task_path = (REPO_ROOT / args.task).resolve() if not Path(args.task).is_absolute() else Path(args.task)
    task = load_json(task_path)

    task_name = task["name"]
    script_rel = task["script"]
    base_args = task.get("args", {})
    task_type = task.get("type", "single_run")

    results = []
    if task_type == "single_run":
        results.append(run_one(task_name, script_rel, "run_001", base_args))
    elif task_type == "sweep":
        sweep = task["sweep"]
        sweep_arg = sweep["arg"]
        for value in sweep["values"]:
            run_args = deepcopy(base_args)
            run_args[sweep_arg] = value
            run_name = f"{sweep_arg}_{value}"
            results.append(run_one(task_name, script_rel, run_name, run_args))
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    manifest = {
        "task_file": str(task_path),
        "task_name": task_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results": results,
    }
    manifest_path = RUNS_ROOT / task_name / "benchmark_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("Benchmark manifest saved to:", manifest_path)


if __name__ == "__main__":
    main()
