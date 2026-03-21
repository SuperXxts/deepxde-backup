import json
import os

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


def main():
    args = parse_args()
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
