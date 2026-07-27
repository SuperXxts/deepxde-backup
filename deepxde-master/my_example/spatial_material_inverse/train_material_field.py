import json
import os

for env_name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(env_name, "1")

import numpy as np

from shared import (
    build_common_parser,
    build_data,
    build_model,
    configure_torch_runtime,
    count_trainable_parameters,
    evaluate_model,
    find_model_path,
    inspect_runtime_device,
    make_callbacks,
    pde_loss_names,
    prepare_run,
    record_checkpoint_material_probe_snapshot,
    report_checkpoint_preference,
    save_run_config_artifacts,
    save_json,
    save_last_model,
    save_runtime_status,
    save_training_artifacts,
    validation_observation_target_name,
)


def parse_args():
    parser = build_common_parser("Train material-field inverse models")
    parser.add_argument("--run_eval_after_train", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    runtime_thread_info = configure_torch_runtime(thread_limit=1)
    case_config = prepare_run(args)

    geom, data, metadata = build_data(args, case_config)
    model, net = build_model(args, data)
    save_run_config_artifacts(args, args.save_dir, case_config, net=net, metadata=metadata)
    runtime_device_info = inspect_runtime_device(args, net)
    save_json(
        os.path.join(args.save_dir, "json", "运行时环境.json"),
        {
            "thread_runtime": runtime_thread_info,
            "device_runtime": runtime_device_info,
        },
    )

    if hasattr(data, "train_x_all") and data.train_x_all is not None:
        train_x_all = np.asarray(data.train_x_all)
        args._domain_points_for_plot = train_x_all[:, :2]
    else:
        args._domain_points_for_plot = geom.random_points(min(args.num_domain, 4000))

    callbacks = make_callbacks(args, args.save_dir, metadata)
    save_runtime_status(args.save_dir, "before_train", step=int(getattr(model.train_state, "step", 0)))
    losshistory, train_state = model.train(
        iterations=args.iterations,
        display_every=args.display_every,
        callbacks=callbacks,
    )
    save_runtime_status(args.save_dir, "train_completed", step=int(getattr(model.train_state, "step", 0)))

    save_last_model(model, args.save_dir)
    save_training_artifacts(args, args.save_dir, case_config, losshistory, metadata, net)
    best_probe_snapshot = record_checkpoint_material_probe_snapshot(
        model=model,
        save_dir=args.save_dir,
        case_config=case_config,
        args=args,
        checkpoint_preference="best_model",
        restore_after=True,
        verbose=0,
    )
    save_runtime_status(args.save_dir, "artifacts_saved", step=int(getattr(model.train_state, "step", 0)))

    summary = {
        "save_dir": args.save_dir,
        "parameter_count": count_trainable_parameters(net),
        "method": args.method,
        "case": args.case,
        "iterations": args.iterations,
        "seed": args.seed,
        "num_observe": args.num_observe,
        "noise_level": args.noise_level,
        "pde_loss_names": pde_loss_names(args.reg_weight),
        "teacher_source": getattr(args, "teacher_source", "analytic"),
        "teacher_run_dir": getattr(args, "teacher_run_dir", None),
        "validation_observation_target": validation_observation_target_name(args),
        "report_checkpoint": report_checkpoint_preference(args),
        "best_probe_snapshot": best_probe_snapshot,
    }

    if args.run_eval_after_train:
        report_checkpoint_path = find_model_path(
            args.save_dir,
            getattr(args, "model_path", None),
            prefer=report_checkpoint_preference(args),
        )
        model.restore(report_checkpoint_path, verbose=0)
        summary["post_train_metrics"] = evaluate_model(
            args=args,
            model=model,
            save_dir=args.save_dir,
            case_config=case_config,
            observation_points=metadata["observation_points"],
            observation_truth_clean=metadata["observation_clean"],
        )
        summary["reported_checkpoint_path"] = report_checkpoint_path

    save_json(os.path.join(args.save_dir, "json", "train_summary.json"), summary)

    print("=" * 80)
    print("Training finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
