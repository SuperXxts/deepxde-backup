import json
import os

import numpy as np

from shared import (
    build_common_parser,
    build_data,
    build_model,
    count_trainable_parameters,
    evaluate_model,
    find_model_path,
    make_callbacks,
    pde_loss_names,
    prepare_run,
    save_json,
    save_last_model,
    save_training_artifacts,
)


def parse_args():
    parser = build_common_parser("Train material-field inverse models")
    parser.add_argument("--run_eval_after_train", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    case_config = prepare_run(args)

    geom, data, metadata = build_data(args, case_config)
    model, net = build_model(args, data)

    if hasattr(data, "train_x_all") and data.train_x_all is not None:
        train_x_all = np.asarray(data.train_x_all)
        args._domain_points_for_plot = train_x_all[:, :2]
    else:
        args._domain_points_for_plot = geom.random_points(min(args.num_domain, 4000))

    callbacks = make_callbacks(args, args.save_dir)
    losshistory, train_state = model.train(
        iterations=args.iterations,
        display_every=args.display_every,
        callbacks=callbacks,
    )

    save_last_model(model, args.save_dir)
    save_training_artifacts(args, args.save_dir, case_config, losshistory, metadata, net)

    summary = {
        "save_dir": args.save_dir,
        "parameter_count": count_trainable_parameters(net),
        "method": args.method,
        "case": args.case,
        "iterations": args.iterations,
        "seed": args.seed,
        "num_observe": args.num_observe,
        "noise_level": args.noise_level,
        "pde_loss_names": pde_loss_names(args.reg_weight, args.method),
    }

    if args.run_eval_after_train:
        summary["last_model_path"] = find_model_path(args.save_dir, prefer="last_model")
        summary["best_model_path"] = find_model_path(args.save_dir, prefer="best_model")
        summary["post_train_metrics_last"] = evaluate_model(
            args=args,
            model=model,
            save_dir=args.save_dir,
            case_config=case_config,
            observation_points=metadata["observation_points"],
            observation_truth_clean=metadata["observation_clean"],
            save_artifacts=False,
        )
        model.restore(summary["best_model_path"], verbose=1)
        summary["post_train_metrics_best"] = evaluate_model(
            args=args,
            model=model,
            save_dir=args.save_dir,
            case_config=case_config,
            observation_points=metadata["observation_points"],
            observation_truth_clean=metadata["observation_clean"],
            save_artifacts=True,
        )
        summary["post_train_metrics"] = summary["post_train_metrics_best"]

    save_json(os.path.join(args.save_dir, "json", "train_summary.json"), summary)

    print("=" * 80)
    print("Training finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
