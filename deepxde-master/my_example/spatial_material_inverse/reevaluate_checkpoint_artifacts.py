import argparse
import json
import os

import numpy as np

from shared import (
    build_common_parser,
    build_data,
    build_model,
    configure_torch_runtime,
    evaluate_model,
    find_model_path,
    prepare_run,
    save_json,
)


def load_run_args(run_dir):
    config_path = os.path.join(run_dir, "json", "run_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing run_config.json: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    parser = build_common_parser("Re-evaluate a completed checkpoint without training")
    args = parser.parse_args([])
    for key, value in payload.get("args", {}).items():
        setattr(args, key, value)
    args.save_dir = run_dir
    args.run_eval_after_train = False
    return args


def main():
    parser = argparse.ArgumentParser(description="Regenerate evaluation artifacts from a completed checkpoint.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", choices=["best_model", "last_model"], default="best_model")
    args_cli = parser.parse_args()

    run_dir = os.path.abspath(args_cli.run_dir)
    runtime_thread_info = configure_torch_runtime(thread_limit=1)
    args = load_run_args(run_dir)
    args.save_dir = run_dir
    args.report_checkpoint = args_cli.checkpoint

    case_config = prepare_run(args)
    _, data, metadata = build_data(args, case_config)
    model, _ = build_model(args, data)

    checkpoint_path = find_model_path(run_dir, prefer=args_cli.checkpoint)
    model.restore(checkpoint_path, verbose=0)

    metrics = evaluate_model(
        args=args,
        model=model,
        save_dir=run_dir,
        case_config=case_config,
        observation_points=metadata["observation_points"],
        observation_truth_clean=metadata["observation_clean"],
    )

    summary = {
        "run_dir": run_dir,
        "checkpoint": args_cli.checkpoint,
        "checkpoint_path": checkpoint_path,
        "runtime_thread_info": runtime_thread_info,
        "material_field_mean_abs_vector_error": metrics.get("material_field_mean_abs_vector_error"),
        "material_field_mean_abs_vector_error_bulk_mu": metrics.get("material_field_mean_abs_vector_error_bulk_mu"),
    }
    save_json(os.path.join(run_dir, "json", f"reevaluation_{args_cli.checkpoint}_summary.json"), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
