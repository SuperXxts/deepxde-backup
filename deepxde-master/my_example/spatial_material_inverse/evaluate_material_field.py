import argparse
import json
import os
from pathlib import Path

from shared import (
    build_common_parser,
    build_data,
    build_model,
    evaluate_model,
    find_model_path,
    prepare_run,
)


def parse_args():
    parser = build_common_parser("Evaluate material-field inverse models")
    parser.add_argument("--checkpoint_preference", choices=["best_model", "last_model"], default="best_model")
    return parser.parse_args()


def restore_args_from_run_config(cli_args):
    run_config_path = Path(cli_args.save_dir) / "json" / "run_config.json"
    if not run_config_path.exists():
        return cli_args

    with open(run_config_path, "r", encoding="utf-8") as f:
        run_config = json.load(f)

    restored = argparse.Namespace(**run_config["args"])
    restored.save_dir = cli_args.save_dir
    restored.model_path = cli_args.model_path
    restored.checkpoint_preference = cli_args.checkpoint_preference
    return restored


def main():
    args = restore_args_from_run_config(parse_args())
    case_config = prepare_run(args)
    _, data, metadata = build_data(args, case_config)
    model, _ = build_model(args, data)

    model_path = find_model_path(args.save_dir, args.model_path, prefer=args.checkpoint_preference)
    model.restore(model_path, verbose=1)

    metrics = evaluate_model(
        args=args,
        model=model,
        save_dir=args.save_dir,
        case_config=case_config,
        observation_points=metadata["eval_observation"]["points"],
        observation_truth_clean=metadata["eval_observation"]["clean"],
        observation_split_name="evaluation",
    )

    print("=" * 80)
    print("Evaluation finished.")
    print(f"Model path: {model_path}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
