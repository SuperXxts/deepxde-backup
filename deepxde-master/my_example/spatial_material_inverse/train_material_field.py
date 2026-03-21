import json
import os

import numpy as np

from shared import (
    build_common_parser,
    build_data,
    build_model,
    compute_observation_mse,
    count_trainable_parameters,
    evaluate_model,
    find_model_path,
    make_callbacks,
    pde_loss_names,
    prepare_run,
    resolve_loss_weights,
    save_json,
    save_last_model,
    save_loss_history_dat,
    save_loss_history_json,
    save_best_test_loss_json,
    save_training_artifacts,
)


def parse_args():
    parser = build_common_parser("Train material-field inverse models")
    parser.add_argument("--run_eval_after_train", action="store_true")
    parser.add_argument("--staged_training", action="store_true")
    parser.add_argument("--warmup_iterations", type=int, default=1000)
    parser.add_argument("--warmup_lr", type=float, default=1e-3)
    parser.add_argument("--main_lr", type=float, default=5e-4)
    parser.add_argument("--warmup_physics_scale", type=float, default=0.0)
    parser.add_argument("--warmup_reg_scale", type=float, default=0.0)
    parser.add_argument("--freeze_material_warmup", action="store_true")
    return parser.parse_args()


def set_requires_grad(module, flag):
    for parameter in module.parameters():
        parameter.requires_grad = bool(flag)


def set_material_branch_trainable(net, trainable):
    if hasattr(net, "interface_net"):
        set_requires_grad(net.interface_net, trainable)
    if hasattr(net, "raw_lambda_params"):
        net.raw_lambda_params.requires_grad = bool(trainable)
    if hasattr(net, "raw_mu_params"):
        net.raw_mu_params.requires_grad = bool(trainable)


def save_stage_loss_artifacts(losshistory, save_dir, stage_name):
    if losshistory is None:
        return
    save_loss_history_json(losshistory, save_dir, filename=f"{stage_name}_loss_history.json")
    save_best_test_loss_json(losshistory, save_dir, filename=f"{stage_name}_best_test_loss.json")
    save_loss_history_dat(losshistory, save_dir, filename=f"{stage_name}_loss_history.dat")


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

    losshistory = None
    if args.method == "iaminn_v2" and args.staged_training and args.iterations > 1:
        warmup_iterations = min(max(args.warmup_iterations, 0), max(args.iterations - 1, 0))
        main_iterations = args.iterations - warmup_iterations
        if warmup_iterations > 0:
            if args.freeze_material_warmup:
                set_material_branch_trainable(net, False)
            warmup_weights = resolve_loss_weights(
                args,
                physics_scale=args.warmup_physics_scale,
                reg_scale=args.warmup_reg_scale,
            )
            model.compile("adam", lr=args.warmup_lr, loss_weights=warmup_weights)
            warmup_history, _ = model.train(
                iterations=warmup_iterations,
                display_every=max(100, min(args.display_every, warmup_iterations)),
                callbacks=[],
            )
            save_stage_loss_artifacts(warmup_history, args.save_dir, "warmup")
            save_json(
                os.path.join(args.save_dir, "json", "stage_training_plan.json"),
                {
                    "staged_training": True,
                    "warmup_iterations": warmup_iterations,
                    "main_iterations": main_iterations,
                    "warmup_lr": args.warmup_lr,
                    "main_lr": args.main_lr,
                    "warmup_physics_scale": args.warmup_physics_scale,
                    "warmup_reg_scale": args.warmup_reg_scale,
                    "freeze_material_warmup": args.freeze_material_warmup,
                },
            )
            set_material_branch_trainable(net, True)
        model.compile("adam", lr=args.main_lr, loss_weights=resolve_loss_weights(args))
        callbacks = make_callbacks(args, args.save_dir, metadata)
        losshistory, train_state = model.train(
            iterations=max(main_iterations, 1),
            display_every=args.display_every,
            callbacks=callbacks,
        )
    else:
        callbacks = make_callbacks(args, args.save_dir, metadata)
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
        "num_boundary": args.num_boundary,
        "num_observe": args.num_observe,
        "num_val_observe": args.num_val_observe,
        "num_eval_observe": args.num_eval_observe,
        "noise_level": args.noise_level,
        "boundary_weight": args.boundary_weight,
        "observation_split_tag": args.observation_split_tag,
        "pde_loss_names": pde_loss_names(args.reg_weight, args.method),
        "selection_metric": "validation_observation_mse",
        "staged_training": bool(args.staged_training and args.method == "iaminn_v2"),
    }

    if args.run_eval_after_train:
        summary["last_model_path"] = find_model_path(args.save_dir, prefer="last_model")
        summary["best_model_path"] = find_model_path(args.save_dir, prefer="best_model")
        summary["observation_cache_path"] = metadata.get("observation_cache_path")
        summary["observation_split_sizes"] = {
            "boundary": int(len(metadata["boundary_observation"]["points"])),
            "train": int(len(metadata["train_observation"]["points"])),
            "validation": int(len(metadata["val_observation"]["points"])),
            "evaluation": int(len(metadata["eval_observation"]["points"])),
        }
        summary["post_train_metrics_last"] = evaluate_model(
            args=args,
            model=model,
            save_dir=args.save_dir,
            case_config=case_config,
            observation_points=metadata["eval_observation"]["points"],
            observation_truth_clean=metadata["eval_observation"]["clean"],
            observation_split_name="evaluation_last",
            save_artifacts=False,
        )
        summary["post_train_metrics_last_by_split"] = {
            "train": evaluate_model(
                args=args,
                model=model,
                save_dir=args.save_dir,
                case_config=case_config,
                observation_points=metadata["train_observation"]["points"],
                observation_truth_clean=metadata["train_observation"]["clean"],
                observation_split_name="train_last",
                save_artifacts=False,
            ),
            "validation": evaluate_model(
                args=args,
                model=model,
                save_dir=args.save_dir,
                case_config=case_config,
                observation_points=metadata["val_observation"]["points"],
                observation_truth_clean=metadata["val_observation"]["clean"],
                observation_split_name="validation_last",
                save_artifacts=False,
            ),
            "evaluation": summary["post_train_metrics_last"],
        }
        summary["observation_mse_last_by_split"] = {
            "train": compute_observation_mse(model, args, metadata["train_observation"]["points"], metadata["train_observation"]["clean"]),
            "validation": compute_observation_mse(model, args, metadata["val_observation"]["points"], metadata["val_observation"]["clean"]),
            "evaluation": compute_observation_mse(model, args, metadata["eval_observation"]["points"], metadata["eval_observation"]["clean"]),
        }
        summary["validation_selection_mse_last"] = compute_observation_mse(
            model, args, metadata["val_observation"]["points"], metadata["val_observation"]["noisy"]
        )
        model.restore(summary["best_model_path"], verbose=1)
        summary["post_train_metrics_best_by_split"] = {
            "train": evaluate_model(
                args=args,
                model=model,
                save_dir=args.save_dir,
                case_config=case_config,
                observation_points=metadata["train_observation"]["points"],
                observation_truth_clean=metadata["train_observation"]["clean"],
                observation_split_name="train",
                save_artifacts=True,
            ),
            "validation": evaluate_model(
                args=args,
                model=model,
                save_dir=args.save_dir,
                case_config=case_config,
                observation_points=metadata["val_observation"]["points"],
                observation_truth_clean=metadata["val_observation"]["clean"],
                observation_split_name="validation",
                save_artifacts=True,
            ),
            "evaluation": evaluate_model(
                args=args,
                model=model,
                save_dir=args.save_dir,
                case_config=case_config,
                observation_points=metadata["eval_observation"]["points"],
                observation_truth_clean=metadata["eval_observation"]["clean"],
                observation_split_name="evaluation",
                save_artifacts=True,
            ),
        }
        summary["post_train_metrics_best"] = summary["post_train_metrics_best_by_split"]["evaluation"]
        summary["post_train_metrics"] = summary["post_train_metrics_best"]
        summary["observation_mse_best_by_split"] = {
            "train": compute_observation_mse(model, args, metadata["train_observation"]["points"], metadata["train_observation"]["clean"]),
            "validation": compute_observation_mse(model, args, metadata["val_observation"]["points"], metadata["val_observation"]["clean"]),
            "evaluation": compute_observation_mse(model, args, metadata["eval_observation"]["points"], metadata["eval_observation"]["clean"]),
        }
        summary["validation_selection_mse_best"] = compute_observation_mse(
            model, args, metadata["val_observation"]["points"], metadata["val_observation"]["noisy"]
        )
    else:
        summary["post_train_metrics_best"] = None

    save_json(os.path.join(args.save_dir, "json", "train_summary.json"), summary)

    print("=" * 80)
    print("Training finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
