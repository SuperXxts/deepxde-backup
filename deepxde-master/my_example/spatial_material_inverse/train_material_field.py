import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from shared import (
    build_common_parser,
    build_data,
    build_model,
    compute_observation_mse,
    count_trainable_parameters,
    evaluate_model,
    exact_body_force_torch,
    find_model_path,
    is_compact_material_method,
    make_callbacks,
    pde_loss_names,
    prepare_run,
    resolve_loss_weights,
    save_array_txt,
    save_json,
    save_text,
    save_last_model,
    save_loss_history_dat,
    save_loss_history_json,
    save_best_test_loss_json,
    save_training_artifacts,
    plot_all_loss_components,
    plot_and_save_loss_history,
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
    parser.add_argument("--geometry_stage_iterations", type=int, default=0)
    parser.add_argument("--geometry_stage_lr", type=float, default=8e-4)
    parser.add_argument("--geometry_stage_physics_scale", type=float, default=0.02)
    parser.add_argument("--geometry_stage_reg_scale", type=float, default=0.0)
    parser.add_argument("--geometry_stage_data_scale", type=float, default=1.0)
    parser.add_argument("--geometry_stage_boundary_scale", type=float, default=1.0)
    parser.add_argument("--geometry_stage_balance_target", type=float, default=0.5)
    parser.add_argument("--geometry_stage_balance_weight", type=float, default=0.0)
    parser.add_argument("--geometry_stage_binary_weight", type=float, default=0.0)
    parser.add_argument("--geometry_stage_interface_sharpness", type=float, default=-1.0)
    parser.add_argument("--freeze_region_geometry_stage", action="store_true")
    parser.add_argument("--freeze_state_geometry_stage", action="store_true")
    parser.add_argument("--material_stage_iterations", type=int, default=0)
    parser.add_argument("--material_stage_lr", type=float, default=8e-4)
    parser.add_argument("--material_stage_physics_scale", type=float, default=0.05)
    parser.add_argument("--material_stage_reg_scale", type=float, default=1.0)
    parser.add_argument("--material_stage_usage_floor", type=float, default=0.05)
    parser.add_argument("--material_stage_usage_weight", type=float, default=20.0)
    parser.add_argument("--material_stage_binary_weight", type=float, default=0.05)
    parser.add_argument("--material_stage_contrast_weight", type=float, default=0.0)
    parser.add_argument("--material_stage_lambda_gap_target", type=float, default=0.0)
    parser.add_argument("--material_stage_mu_gap_target", type=float, default=0.0)
    parser.add_argument("--geometry_prior_weight", type=float, default=0.0)
    parser.add_argument("--layer_y_prior_target", type=float, default=0.5)
    parser.add_argument("--layer_y_init", type=float, default=-1.0)
    parser.add_argument("--freeze_geometry_material_stage", action="store_true")
    parser.add_argument("--freeze_geometry_main", action="store_true")
    parser.add_argument("--material_stage_interface_sharpness", type=float, default=-1.0)
    parser.add_argument("--main_stage_interface_sharpness", type=float, default=-1.0)
    parser.add_argument("--freeze_state_material_stage", action="store_true")
    parser.add_argument('--refinement_stage_iterations', type=int, default=0)
    parser.add_argument('--refinement_stage_lr', type=float, default=2e-4)
    parser.add_argument('--refinement_stage_physics_scale', type=float, default=1.0)
    parser.add_argument('--refinement_stage_reg_scale', type=float, default=1.0)
    parser.add_argument('--refinement_stage_data_scale', type=float, default=1.0)
    parser.add_argument('--refinement_stage_boundary_scale', type=float, default=1.0)
    parser.add_argument('--freeze_geometry_refinement', action='store_true')
    parser.add_argument('--refinement_stage_interface_sharpness', type=float, default=-1.0)
    parser.add_argument("--adaptive_main_stage", action="store_true")
    parser.add_argument("--adaptive_main_chunks", type=int, default=4)
    parser.add_argument("--adaptive_main_base_physics_scale", type=float, default=1.0)
    parser.add_argument("--adaptive_main_min_physics_scale", type=float, default=0.25)
    parser.add_argument("--adaptive_main_max_physics_scale", type=float, default=4.0)
    parser.add_argument("--adaptive_main_scale_up", type=float, default=1.35)
    parser.add_argument("--adaptive_main_scale_down", type=float, default=0.75)
    parser.add_argument("--adaptive_main_obs_guard", type=float, default=1.15)
    parser.add_argument("--adaptive_main_reg_scale", type=float, default=1.0)
    parser.add_argument("--adaptive_main_data_scale", type=float, default=1.0)
    parser.add_argument("--adaptive_main_boundary_scale", type=float, default=1.0)
    parser.add_argument("--adaptive_main_domain_points", type=int, default=2048)
    parser.add_argument("--main_correction_scale_start", type=float, default=1.0)
    parser.add_argument("--main_correction_scale_end", type=float, default=1.0)
    parser.add_argument("--main_correction_chunks", type=int, default=1)
    parser.add_argument("--main_stage_physics_scale_start", type=float, default=1.0)
    parser.add_argument("--main_stage_physics_scale_end", type=float, default=1.0)
    parser.add_argument("--main_stage_reg_scale_start", type=float, default=1.0)
    parser.add_argument("--main_stage_reg_scale_end", type=float, default=1.0)
    parser.add_argument("--main_stage_data_scale_start", type=float, default=1.0)
    parser.add_argument("--main_stage_data_scale_end", type=float, default=1.0)
    parser.add_argument("--main_stage_boundary_scale_start", type=float, default=1.0)
    parser.add_argument("--main_stage_boundary_scale_end", type=float, default=1.0)
    parser.add_argument("--main_stage_lr_start", type=float, default=-1.0)
    parser.add_argument("--main_stage_lr_end", type=float, default=-1.0)
    parser.add_argument("--geometry_stage_domain_points", type=int, default=1024)
    parser.add_argument("--material_stage_domain_points", type=int, default=1024)
    return parser.parse_args()


def set_requires_grad(module, flag):
    for parameter in module.parameters():
        parameter.requires_grad = bool(flag)


def set_material_branch_trainable(net, trainable):
    for name, parameter in net.named_parameters():
        if name.startswith("state_net."):
            continue
        parameter.requires_grad = bool(trainable)


def set_geometry_branch_trainable(net, trainable):
    geometry_prefixes = ("raw_layer_y", "raw_circle_")
    for name, parameter in net.named_parameters():
        if name.startswith(geometry_prefixes):
            parameter.requires_grad = bool(trainable)


def set_region_parameter_trainable(net, trainable):
    for name, parameter in net.named_parameters():
        if name in {"raw_lambda_params", "raw_mu_params"}:
            parameter.requires_grad = bool(trainable)


def set_residual_branch_trainable(net, trainable):
    residual_prefixes = ("residual_net.", "lambda_residual_net.", "mu_residual_net.")
    for name, parameter in net.named_parameters():
        if name.startswith(residual_prefixes):
            parameter.requires_grad = bool(trainable)


def set_correction_stage_scale(net, scale):
    if hasattr(net, "active_correction_scale"):
        net.active_correction_scale = float(scale)


def mean_squared_error(prediction, target):
    if prediction.numel() == 0:
        return torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    return torch.mean((prediction - target) ** 2)


def bounded_logit(target, lower, upper):
    clipped = min(max(float(target), lower + 1e-6), upper - 1e-6)
    normalized = (clipped - lower) / (upper - lower)
    return float(np.log(normalized / (1.0 - normalized)))


def split_iterations(total_iterations, num_chunks):
    total_iterations = max(int(total_iterations), 0)
    if total_iterations <= 0:
        return []
    num_chunks = max(1, min(int(num_chunks), total_iterations))
    base = total_iterations // num_chunks
    remainder = total_iterations % num_chunks
    return [base + (1 if idx < remainder else 0) for idx in range(num_chunks) if base + (1 if idx < remainder else 0) > 0]


def linear_schedule_value(start, end, alpha):
    return float(start) + (float(end) - float(start)) * float(alpha)


def initialize_geometry_parameters(args, net):
    if getattr(args, "layer_y_init", -1.0) > 0 and hasattr(net, "raw_layer_y"):
        raw_value = bounded_logit(args.layer_y_init, 0.15, 0.85)
        net.raw_layer_y.data.fill_(raw_value)


def save_stage_loss_artifacts(losshistory, save_dir, stage_name, args=None):
    if losshistory is None:
        return
    save_loss_history_json(losshistory, save_dir, filename=f"{stage_name}_loss_history.json")
    save_best_test_loss_json(losshistory, save_dir, filename=f"{stage_name}_best_test_loss.json")
    save_loss_history_dat(losshistory, save_dir, filename=f"{stage_name}_loss_history.dat")
    if args is not None:
        num_pde = len(pde_loss_names(args.reg_weight, args.method))
        plot_and_save_loss_history(
            losshistory,
            save_dir,
            filename=f"{stage_name}_loss_history.png",
            num_pde_losses=num_pde,
            num_bc_losses=4,
            pde_label="Physics Loss",
            bc_label="Boundary + Observation Loss",
            bc_loss_names=["boundary_ux", "boundary_uy", "obs_ux", "obs_uy"],
            data_loss_prefix="obs_",
        )
        plot_all_loss_components(
            losshistory,
            save_dir,
            filename=f"{stage_name}_loss_components.png",
            num_pde_losses=num_pde,
            num_bc_losses=4,
            pde_loss_names=pde_loss_names(args.reg_weight, args.method),
            bc_loss_names=["boundary_ux", "boundary_uy", "obs_ux", "obs_uy"],
        )


def extract_material_stage_points(data, geom, seed, fallback_count, max_points=None):
    train_x_all = getattr(data, "train_x_all", None)
    if train_x_all is not None:
        points = np.asarray(train_x_all, dtype=float)[:, :2]
        interior_mask = (
            (points[:, 0] > 1e-8)
            & (points[:, 0] < 1.0 - 1e-8)
            & (points[:, 1] > 1e-8)
            & (points[:, 1] < 1.0 - 1e-8)
        )
        interior_points = points[interior_mask]
        if len(interior_points) > 0:
            points = interior_points
            if max_points is not None and len(points) > int(max_points):
                rng = np.random.default_rng(seed)
                indices = rng.choice(len(points), size=int(max_points), replace=False)
                points = points[indices]
            return points
    np.random.seed(seed)
    points = geom.random_points(fallback_count)
    if max_points is not None and len(points) > int(max_points):
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(points), size=int(max_points), replace=False)
        points = points[indices]
    return points


def compute_compact_material_stage_terms(net, domain_points, case_config, reg_weight, usage_floor, args):
    device = next(net.parameters()).device
    x = torch.tensor(domain_points, dtype=torch.float32, device=device, requires_grad=True)
    raw = net(x)
    ux = raw[:, 0:1]
    uy = raw[:, 1:2]
    lmbd = raw[:, 2:3]
    mu = raw[:, 3:4]

    ux_grad = torch.autograd.grad(ux, x, grad_outputs=torch.ones_like(ux), create_graph=True, retain_graph=True)[0]
    uy_grad = torch.autograd.grad(uy, x, grad_outputs=torch.ones_like(uy), create_graph=True, retain_graph=True)[0]
    exx = ux_grad[:, 0:1]
    eyy = uy_grad[:, 1:2]
    exy = 0.5 * (ux_grad[:, 1:2] + uy_grad[:, 0:1])

    sxx = lmbd * (exx + eyy) + 2.0 * mu * exx
    syy = lmbd * (exx + eyy) + 2.0 * mu * eyy
    sxy = 2.0 * mu * exy

    sxx_grad = torch.autograd.grad(sxx, x, grad_outputs=torch.ones_like(sxx), create_graph=True, retain_graph=True)[0]
    syy_grad = torch.autograd.grad(syy, x, grad_outputs=torch.ones_like(syy), create_graph=True, retain_graph=True)[0]
    sxy_grad = torch.autograd.grad(sxy, x, grad_outputs=torch.ones_like(sxy), create_graph=True, retain_graph=True)[0]
    fx, fy = exact_body_force_torch(x, case_config)
    momentum_x = sxx_grad[:, 0:1] + sxy_grad[:, 1:2] + fx
    momentum_y = sxy_grad[:, 0:1] + syy_grad[:, 1:2] + fy

    lambda_grad = torch.autograd.grad(lmbd, x, grad_outputs=torch.ones_like(lmbd), create_graph=True, retain_graph=True)[0]
    mu_grad = torch.autograd.grad(mu, x, grad_outputs=torch.ones_like(mu), create_graph=True, retain_graph=True)[0]

    diagnostics = net.predict_material_diagnostics(x)
    class_probs = diagnostics["class_probs"]
    mean_probs = torch.mean(class_probs, dim=0)
    floor = max(float(usage_floor), 1e-8)
    safe_ratio = torch.clamp(mean_probs / floor, min=1e-8, max=1.0)
    active_mask = (mean_probs < floor).to(mean_probs.dtype)
    usage_penalty = torch.mean((-torch.log(safe_ratio)) * active_mask)
    if class_probs.shape[1] > 1:
        binary_penalty = torch.mean(class_probs[:, 1:] * (1.0 - class_probs[:, 1:]))
    else:
        binary_penalty = torch.zeros((), dtype=torch.float32, device=device)

    reg_terms = torch.cat(
        [
            lambda_grad[:, 0:1],
            lambda_grad[:, 1:2],
            mu_grad[:, 0:1],
            mu_grad[:, 1:2],
        ],
        dim=1,
    )
    reg_mse = torch.mean(reg_terms**2)
    physics_mse = torch.mean(momentum_x**2) + torch.mean(momentum_y**2)
    geometry_prior = torch.zeros((), dtype=torch.float32, device=device)
    geometry_values = {}
    correction_penalty = torch.zeros((), dtype=torch.float32, device=device)
    correction_gate_mean = torch.zeros((), dtype=torch.float32, device=device)
    for key, value in diagnostics.items():
        if key in {"lambda", "mu", "interface_indicator", "class_probs", "lambda_regions", "mu_regions"}:
            continue
        if key in {"delta_lambda", "delta_mu", "coarse_lambda", "coarse_mu", "correction_gate", "active_correction_scale"}:
            continue
        geometry_values[key] = value
    if "delta_lambda" in diagnostics and "delta_mu" in diagnostics:
        correction_penalty = torch.mean(diagnostics["delta_lambda"] ** 2 + diagnostics["delta_mu"] ** 2)
    if "correction_gate" in diagnostics:
        correction_gate_mean = torch.mean(diagnostics["correction_gate"])
    if hasattr(net, "case_name") and net.case_name == "layered" and "layer_y" in geometry_values:
        target = torch.tensor(float(args.layer_y_prior_target), dtype=torch.float32, device=device)
        geometry_prior = torch.mean((geometry_values["layer_y"] - target) ** 2)

    return {
        "physics_mse": physics_mse,
        "reg_mse": reg_mse,
        "usage_penalty": usage_penalty,
        "binary_penalty": binary_penalty,
        "geometry_prior": geometry_prior,
        "correction_penalty": correction_penalty,
        "correction_gate_mean": correction_gate_mean,
        "mean_class_probs": mean_probs,
        "lambda_regions": diagnostics["lambda_regions"],
        "mu_regions": diagnostics["mu_regions"],
        "geometry_values": geometry_values,
        "reg_weight": float(reg_weight),
    }


def run_geometry_stage(args, net, geom, data, case_config, metadata, save_dir):
    if not is_compact_material_method(args.method) or args.geometry_stage_iterations <= 0:
        return None

    stage_points = extract_material_stage_points(
        data=data,
        geom=geom,
        seed=args.seed + 651,
        fallback_count=max(args.num_domain, 2000),
        max_points=args.geometry_stage_domain_points,
    )
    os.makedirs(os.path.join(save_dir, "npz"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "txt"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "json"), exist_ok=True)
    np.savez(os.path.join(save_dir, "npz", "geometry_stage_points.npz"), points=stage_points)
    save_array_txt(os.path.join(save_dir, "txt", "geometry_stage_points.txt"), stage_points, "x y")

    device = next(net.parameters()).device
    observation_points = np.asarray(metadata["train_observation"]["points"], dtype=float)
    observation_values = np.asarray(metadata["train_observation"]["noisy"], dtype=float)
    boundary_points = np.asarray(metadata["boundary_observation"]["points"], dtype=float)
    boundary_values = np.asarray(metadata["boundary_observation"]["clean"], dtype=float)

    observation_points_t = torch.tensor(observation_points, dtype=torch.float32, device=device)
    observation_values_t = torch.tensor(observation_values, dtype=torch.float32, device=device)
    boundary_points_t = torch.tensor(boundary_points, dtype=torch.float32, device=device)
    boundary_values_t = torch.tensor(boundary_values, dtype=torch.float32, device=device)

    state_frozen = bool(args.freeze_state_geometry_stage)
    region_frozen = bool(args.freeze_region_geometry_stage)
    if hasattr(net, "state_net"):
        set_requires_grad(net.state_net, not state_frozen)
    set_material_branch_trainable(net, True)
    if hasattr(net, "active_correction_scale"):
        correction_scale_before_stage = float(net.active_correction_scale)
        set_correction_stage_scale(net, 0.0)
    else:
        correction_scale_before_stage = None
    if hasattr(net, "residual_net") or hasattr(net, "lambda_residual_net"):
        set_residual_branch_trainable(net, False)
    if region_frozen:
        set_region_parameter_trainable(net, False)
    set_geometry_branch_trainable(net, True)

    original_sharpness = getattr(net, "interface_sharpness", None)
    if original_sharpness is not None and args.geometry_stage_interface_sharpness > 0:
        net.interface_sharpness = float(args.geometry_stage_interface_sharpness)

    parameters = [parameter for parameter in net.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.geometry_stage_lr)
    display_every = max(50, min(args.display_every, args.geometry_stage_iterations))
    history_rows = []
    best_row = None

    for step in range(1, args.geometry_stage_iterations + 1):
        optimizer.zero_grad()
        terms = compute_compact_material_stage_terms(
            net=net,
            domain_points=stage_points,
            case_config=case_config,
            reg_weight=args.reg_weight,
            usage_floor=args.material_stage_usage_floor,
            args=args,
        )

        observation_prediction = net(observation_points_t)[:, :2]
        boundary_prediction = net(boundary_points_t)[:, :2]
        data_mse = mean_squared_error(observation_prediction, observation_values_t)
        boundary_mse = mean_squared_error(boundary_prediction, boundary_values_t)

        if terms["mean_class_probs"].numel() >= 2:
            balance_penalty = torch.mean(
                (terms["mean_class_probs"][1:] - float(args.geometry_stage_balance_target)) ** 2
            )
        else:
            balance_penalty = torch.zeros((), dtype=torch.float32, device=device)

        total_loss = (
            float(args.geometry_stage_physics_scale) * terms["physics_mse"]
            + float(args.geometry_stage_reg_scale) * float(args.reg_weight) * terms["reg_mse"]
            + float(args.data_weight) * float(args.geometry_stage_data_scale) * data_mse
            + float(args.boundary_weight) * float(args.geometry_stage_boundary_scale) * boundary_mse
            + float(args.geometry_stage_binary_weight) * terms["binary_penalty"]
            + float(args.geometry_stage_balance_weight) * balance_penalty
            + float(args.geometry_prior_weight) * terms["geometry_prior"]
        )
        total_loss.backward()
        optimizer.step()

        row = {
            "step": int(step),
            "total_loss": float(total_loss.detach().cpu().item()),
            "physics_mse": float(terms["physics_mse"].detach().cpu().item()),
            "reg_mse": float(terms["reg_mse"].detach().cpu().item()),
            "data_mse": float(data_mse.detach().cpu().item()),
            "boundary_mse": float(boundary_mse.detach().cpu().item()),
            "binary_penalty": float(terms["binary_penalty"].detach().cpu().item()),
            "balance_penalty": float(balance_penalty.detach().cpu().item()),
            "geometry_prior": float(terms["geometry_prior"].detach().cpu().item()),
            "correction_penalty": float(terms["correction_penalty"].detach().cpu().item()),
            "correction_gate_mean": float(terms["correction_gate_mean"].detach().cpu().item()),
            "mean_class_probs": [float(value) for value in terms["mean_class_probs"].detach().cpu().numpy().tolist()],
            "lambda_regions": [float(value) for value in terms["lambda_regions"].detach().cpu().numpy().tolist()],
            "mu_regions": [float(value) for value in terms["mu_regions"].detach().cpu().numpy().tolist()],
        }
        for key, value in terms["geometry_values"].items():
            row[key] = float(value.detach().cpu().item())
        if best_row is None or row["total_loss"] < best_row["total_loss"]:
            best_row = dict(row)
        if step == 1 or step % display_every == 0 or step == args.geometry_stage_iterations:
            history_rows.append(row)
            print(
                "[geometry_stage] step={} total={:.4e} physics={:.4e} data={:.4e} bc={:.4e} binary={:.4e} balance={:.4e} geom={:.4e} probs={}".format(
                    row["step"],
                    row["total_loss"],
                    row["physics_mse"],
                    row["data_mse"],
                    row["boundary_mse"],
                    row["binary_penalty"],
                    row["balance_penalty"],
                    row["geometry_prior"],
                    [round(value, 5) for value in row["mean_class_probs"]],
                )
            )

    if hasattr(net, "state_net"):
        set_requires_grad(net.state_net, True)
    set_region_parameter_trainable(net, True)
    if correction_scale_before_stage is not None:
        set_correction_stage_scale(net, correction_scale_before_stage)
    if hasattr(net, "residual_net") or hasattr(net, "lambda_residual_net"):
        set_residual_branch_trainable(net, True)
    if original_sharpness is not None:
        net.interface_sharpness = float(original_sharpness)

    history_payload = {
        "stage_name": "geometry_stage",
        "iterations": int(args.geometry_stage_iterations),
        "learning_rate": float(args.geometry_stage_lr),
        "physics_scale": float(args.geometry_stage_physics_scale),
        "reg_scale": float(args.geometry_stage_reg_scale),
        "data_scale": float(args.geometry_stage_data_scale),
        "boundary_scale": float(args.geometry_stage_boundary_scale),
        "balance_target": float(args.geometry_stage_balance_target),
        "balance_weight": float(args.geometry_stage_balance_weight),
        "binary_weight": float(args.geometry_stage_binary_weight),
        "geometry_prior_weight": float(args.geometry_prior_weight),
        "freeze_state": state_frozen,
        "freeze_region": region_frozen,
        "num_points": int(len(stage_points)),
        "history": history_rows,
        "best": best_row,
    }
    save_json(os.path.join(save_dir, "json", "geometry_stage_history.json"), history_payload)
    save_text(os.path.join(save_dir, "txt", "geometry_stage_history.txt"), json.dumps(history_payload, indent=2, ensure_ascii=False))
    return history_payload


def run_material_stage(args, net, geom, data, case_config, save_dir):
    if not is_compact_material_method(args.method) or args.material_stage_iterations <= 0:
        return None

    stage_points = extract_material_stage_points(
        data=data,
        geom=geom,
        seed=args.seed + 701,
        fallback_count=max(args.num_domain, 2000),
        max_points=args.material_stage_domain_points,
    )
    os.makedirs(os.path.join(save_dir, "npz"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "txt"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "json"), exist_ok=True)
    np.savez(os.path.join(save_dir, "npz", "material_stage_points.npz"), points=stage_points)
    save_array_txt(os.path.join(save_dir, "txt", "material_stage_points.txt"), stage_points, "x y")

    state_frozen = bool(args.freeze_state_material_stage)
    if state_frozen and hasattr(net, "state_net"):
        set_requires_grad(net.state_net, False)
    set_material_branch_trainable(net, True)
    if hasattr(net, "active_correction_scale"):
        correction_scale_before_stage = float(net.active_correction_scale)
        set_correction_stage_scale(net, 0.0)
    else:
        correction_scale_before_stage = None
    if hasattr(net, "residual_net") or hasattr(net, "lambda_residual_net"):
        set_residual_branch_trainable(net, False)
    if args.freeze_geometry_material_stage:
        set_geometry_branch_trainable(net, False)
        set_region_parameter_trainable(net, True)
    original_sharpness = getattr(net, "interface_sharpness", None)
    if original_sharpness is not None and args.material_stage_interface_sharpness > 0:
        net.interface_sharpness = float(args.material_stage_interface_sharpness)

    parameters = [parameter for parameter in net.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.material_stage_lr)
    display_every = max(50, min(args.display_every, args.material_stage_iterations))
    history_rows = []
    best_row = None

    for step in range(1, args.material_stage_iterations + 1):
        optimizer.zero_grad()
        terms = compute_compact_material_stage_terms(
            net=net,
            domain_points=stage_points,
            case_config=case_config,
            reg_weight=args.reg_weight,
            usage_floor=args.material_stage_usage_floor,
            args=args,
        )
        total_loss = (
            float(args.material_stage_physics_scale) * terms["physics_mse"]
            + float(args.material_stage_reg_scale) * float(args.reg_weight) * terms["reg_mse"]
            + float(args.material_stage_usage_weight) * terms["usage_penalty"]
            + float(args.material_stage_binary_weight) * terms["binary_penalty"]
            + float(args.geometry_prior_weight) * terms["geometry_prior"]
        )
        contrast_penalty = torch.zeros((), dtype=torch.float32, device=total_loss.device)
        lambda_regions = terms["lambda_regions"]
        mu_regions = terms["mu_regions"]
        if lambda_regions.numel() >= 2:
            lambda_gap = lambda_regions[1] - lambda_regions[0]
            mu_gap = mu_regions[1] - mu_regions[0]
            contrast_penalty = (
                torch.relu(float(args.material_stage_lambda_gap_target) - lambda_gap) ** 2
                + torch.relu(float(args.material_stage_mu_gap_target) - mu_gap) ** 2
            )
        total_loss = total_loss + float(args.material_stage_contrast_weight) * contrast_penalty
        total_loss.backward()
        optimizer.step()

        row = {
            "step": int(step),
            "total_loss": float(total_loss.detach().cpu().item()),
            "physics_mse": float(terms["physics_mse"].detach().cpu().item()),
            "reg_mse": float(terms["reg_mse"].detach().cpu().item()),
            "usage_penalty": float(terms["usage_penalty"].detach().cpu().item()),
            "binary_penalty": float(terms["binary_penalty"].detach().cpu().item()),
            "geometry_prior": float(terms["geometry_prior"].detach().cpu().item()),
            "contrast_penalty": float(contrast_penalty.detach().cpu().item()),
            "correction_penalty": float(terms["correction_penalty"].detach().cpu().item()),
            "correction_gate_mean": float(terms["correction_gate_mean"].detach().cpu().item()),
            "mean_class_probs": [float(value) for value in terms["mean_class_probs"].detach().cpu().numpy().tolist()],
            "lambda_regions": [float(value) for value in terms["lambda_regions"].detach().cpu().numpy().tolist()],
            "mu_regions": [float(value) for value in terms["mu_regions"].detach().cpu().numpy().tolist()],
        }
        for key, value in terms["geometry_values"].items():
            row[key] = float(value.detach().cpu().item())
        if best_row is None or row["total_loss"] < best_row["total_loss"]:
            best_row = dict(row)
        if step == 1 or step % display_every == 0 or step == args.material_stage_iterations:
            history_rows.append(row)
            print(
                "[material_stage] step={} total={:.4e} physics={:.4e} reg={:.4e} usage={:.4e} binary={:.4e} contrast={:.4e} geom={:.4e} probs={}".format(
                    row["step"],
                    row["total_loss"],
                    row["physics_mse"],
                    row["reg_mse"],
                    row["usage_penalty"],
                    row["binary_penalty"],
                    row["contrast_penalty"],
                    row["geometry_prior"],
                    [round(value, 5) for value in row["mean_class_probs"]],
                )
            )

    if state_frozen and hasattr(net, "state_net"):
        set_requires_grad(net.state_net, True)
    set_geometry_branch_trainable(net, True)
    set_region_parameter_trainable(net, True)
    if correction_scale_before_stage is not None:
        set_correction_stage_scale(net, correction_scale_before_stage)
    if hasattr(net, "residual_net") or hasattr(net, "lambda_residual_net"):
        set_residual_branch_trainable(net, True)
    if original_sharpness is not None:
        net.interface_sharpness = float(original_sharpness)

    history_payload = {
        "stage_name": "material_stage",
        "iterations": int(args.material_stage_iterations),
        "learning_rate": float(args.material_stage_lr),
        "physics_scale": float(args.material_stage_physics_scale),
        "reg_scale": float(args.material_stage_reg_scale),
        "usage_floor": float(args.material_stage_usage_floor),
        "usage_weight": float(args.material_stage_usage_weight),
        "binary_weight": float(args.material_stage_binary_weight),
        "contrast_weight": float(args.material_stage_contrast_weight),
        "lambda_gap_target": float(args.material_stage_lambda_gap_target),
        "mu_gap_target": float(args.material_stage_mu_gap_target),
        "geometry_prior_weight": float(args.geometry_prior_weight),
        "freeze_state": state_frozen,
        "num_points": int(len(stage_points)),
        "history": history_rows,
        "best": best_row,
    }
    save_json(os.path.join(save_dir, "json", "material_stage_history.json"), history_payload)
    save_text(os.path.join(save_dir, "txt", "material_stage_history.txt"), json.dumps(history_payload, indent=2, ensure_ascii=False))
    return history_payload


def run_main_stage_with_correction_schedule(args, model, net, metadata, save_dir, main_iterations):
    if main_iterations <= 0:
        return None, None

    chunk_sizes = split_iterations(main_iterations, args.main_correction_chunks)
    if not chunk_sizes:
        return None, None

    schedule_rows = []
    losshistory = None
    train_state = None
    correction_start = float(args.main_correction_scale_start)
    correction_end = float(args.main_correction_scale_end)
    physics_start = float(args.main_stage_physics_scale_start)
    physics_end = float(args.main_stage_physics_scale_end)
    reg_start = float(args.main_stage_reg_scale_start)
    reg_end = float(args.main_stage_reg_scale_end)
    data_start = float(args.main_stage_data_scale_start)
    data_end = float(args.main_stage_data_scale_end)
    boundary_start = float(args.main_stage_boundary_scale_start)
    boundary_end = float(args.main_stage_boundary_scale_end)
    lr_start = float(args.main_stage_lr_start if args.main_stage_lr_start > 0 else args.main_lr)
    lr_end = float(args.main_stage_lr_end if args.main_stage_lr_end > 0 else lr_start)

    for chunk_idx, chunk_iterations in enumerate(chunk_sizes, start=1):
        if len(chunk_sizes) == 1:
            alpha = 1.0
        else:
            alpha = float(chunk_idx - 1) / float(len(chunk_sizes) - 1)
        correction_scale = linear_schedule_value(correction_start, correction_end, alpha)
        physics_scale = linear_schedule_value(physics_start, physics_end, alpha)
        reg_scale = linear_schedule_value(reg_start, reg_end, alpha)
        data_scale = linear_schedule_value(data_start, data_end, alpha)
        boundary_scale = linear_schedule_value(boundary_start, boundary_end, alpha)
        lr = linear_schedule_value(lr_start, lr_end, alpha)
        set_correction_stage_scale(net, correction_scale)
        model.compile(
            "adam",
            lr=lr,
            loss_weights=resolve_loss_weights(
                args,
                physics_scale=physics_scale,
                reg_scale=reg_scale,
                data_scale=data_scale,
                boundary_scale=boundary_scale,
            ),
        )
        callbacks = make_callbacks(args, save_dir, metadata)
        losshistory, train_state = model.train(
            iterations=chunk_iterations,
            display_every=max(100, min(args.display_every, chunk_iterations)),
            callbacks=callbacks,
        )
        save_stage_loss_artifacts(losshistory, save_dir, f"main_chunk_{chunk_idx}", args=args)
        val_mse = compute_observation_mse(
            model, args, metadata["val_observation"]["points"], metadata["val_observation"]["noisy"]
        )
        schedule_rows.append(
            {
                "chunk": int(chunk_idx),
                "chunk_iterations": int(chunk_iterations),
                "lr": float(lr),
                "correction_scale": float(correction_scale),
                "physics_scale": float(physics_scale),
                "reg_scale": float(reg_scale),
                "data_scale": float(data_scale),
                "boundary_scale": float(boundary_scale),
                "validation_observation_mse": float(val_mse),
            }
        )

    payload = {
        "stage_name": "main_stage_schedule",
        "main_iterations": int(main_iterations),
        "main_lr": float(args.main_lr),
        "lr_start": float(lr_start),
        "lr_end": float(lr_end),
        "correction_scale_start": float(correction_start),
        "correction_scale_end": float(correction_end),
        "physics_scale_start": float(physics_start),
        "physics_scale_end": float(physics_end),
        "reg_scale_start": float(reg_start),
        "reg_scale_end": float(reg_end),
        "data_scale_start": float(data_start),
        "data_scale_end": float(data_end),
        "boundary_scale_start": float(boundary_start),
        "boundary_scale_end": float(boundary_end),
        "chunks": int(args.main_correction_chunks),
        "history": schedule_rows,
    }
    save_json(os.path.join(save_dir, "json", "main_stage_schedule.json"), payload)
    save_text(os.path.join(save_dir, "txt", "main_stage_schedule.txt"), json.dumps(payload, indent=2, ensure_ascii=False))
    return losshistory, train_state


def run_adaptive_main_stage(args, model, net, geom, data, case_config, metadata, save_dir, main_iterations):
    if not args.adaptive_main_stage or not is_compact_material_method(args.method) or main_iterations <= 0:
        return None, None

    stage_points = extract_material_stage_points(
        data=data,
        geom=geom,
        seed=args.seed + 801,
        fallback_count=max(args.num_domain, args.adaptive_main_domain_points),
    )
    if len(stage_points) > args.adaptive_main_domain_points:
        stage_points = stage_points[: args.adaptive_main_domain_points]
    np.savez(os.path.join(save_dir, "npz", "adaptive_main_stage_points.npz"), points=stage_points)
    save_array_txt(os.path.join(save_dir, "txt", "adaptive_main_stage_points.txt"), stage_points, "x y")

    chunk_sizes = split_iterations(main_iterations, args.adaptive_main_chunks)
    physics_scale = float(args.adaptive_main_base_physics_scale)
    best_val = float("inf")
    history_rows = []
    losshistory = None
    train_state = None

    for chunk_idx, chunk_iterations in enumerate(chunk_sizes, start=1):
        weights = resolve_loss_weights(
            args,
            physics_scale=physics_scale,
            reg_scale=args.adaptive_main_reg_scale,
            boundary_scale=args.adaptive_main_boundary_scale,
            data_scale=args.adaptive_main_data_scale,
        )
        model.compile("adam", lr=args.main_lr, loss_weights=weights)
        callbacks = make_callbacks(args, save_dir, metadata)
        losshistory, train_state = model.train(
            iterations=chunk_iterations,
            display_every=max(100, min(args.display_every, chunk_iterations)),
            callbacks=callbacks,
        )
        save_stage_loss_artifacts(losshistory, save_dir, f"adaptive_main_chunk_{chunk_idx}", args=args)

        val_mse = compute_observation_mse(
            model, args, metadata["val_observation"]["points"], metadata["val_observation"]["noisy"]
        )
        terms = compute_compact_material_stage_terms(
            net=net,
            domain_points=stage_points,
            case_config=case_config,
            reg_weight=args.reg_weight,
            usage_floor=args.material_stage_usage_floor,
            args=args,
        )
        physics_mse = float(terms["physics_mse"].detach().cpu().item())
        reg_mse = float(terms["reg_mse"].detach().cpu().item())

        if val_mse < best_val:
            best_val = val_mse
            next_physics_scale = min(float(args.adaptive_main_max_physics_scale), physics_scale * float(args.adaptive_main_scale_up))
            update_reason = "improved_validation"
        elif val_mse <= best_val * float(args.adaptive_main_obs_guard):
            next_physics_scale = min(
                float(args.adaptive_main_max_physics_scale),
                physics_scale * (1.0 + 0.5 * (float(args.adaptive_main_scale_up) - 1.0)),
            )
            update_reason = "within_guard"
        else:
            next_physics_scale = max(float(args.adaptive_main_min_physics_scale), physics_scale * float(args.adaptive_main_scale_down))
            update_reason = "validation_regression"

        row = {
            "chunk": int(chunk_idx),
            "chunk_iterations": int(chunk_iterations),
            "physics_scale_before": float(physics_scale),
            "physics_scale_after": float(next_physics_scale),
            "validation_selection_mse": float(val_mse),
            "best_validation_selection_mse": float(best_val),
            "physics_mse": physics_mse,
            "reg_mse": reg_mse,
            "mean_class_probs": [float(v) for v in terms["mean_class_probs"].detach().cpu().numpy().tolist()],
            "lambda_regions": [float(v) for v in terms["lambda_regions"].detach().cpu().numpy().tolist()],
            "mu_regions": [float(v) for v in terms["mu_regions"].detach().cpu().numpy().tolist()],
            "update_reason": update_reason,
        }
        for key, value in terms["geometry_values"].items():
            row[key] = float(value.detach().cpu().item())
        history_rows.append(row)
        print(
            "[adaptive_main] chunk={} iters={} val={:.4e} physics_mse={:.4e} scale={:.4f}->{:.4f} reason={}".format(
                row["chunk"],
                row["chunk_iterations"],
                row["validation_selection_mse"],
                row["physics_mse"],
                row["physics_scale_before"],
                row["physics_scale_after"],
                row["update_reason"],
            )
        )
        physics_scale = next_physics_scale

    payload = {
        "stage_name": "adaptive_main_stage",
        "chunks": history_rows,
        "base_physics_scale": float(args.adaptive_main_base_physics_scale),
        "min_physics_scale": float(args.adaptive_main_min_physics_scale),
        "max_physics_scale": float(args.adaptive_main_max_physics_scale),
        "scale_up": float(args.adaptive_main_scale_up),
        "scale_down": float(args.adaptive_main_scale_down),
        "obs_guard": float(args.adaptive_main_obs_guard),
        "reg_scale": float(args.adaptive_main_reg_scale),
        "data_scale": float(args.adaptive_main_data_scale),
        "boundary_scale": float(args.adaptive_main_boundary_scale),
        "domain_points": int(len(stage_points)),
    }
    save_json(os.path.join(save_dir, "json", "adaptive_main_stage_history.json"), payload)
    save_text(os.path.join(save_dir, "txt", "adaptive_main_stage_history.txt"), json.dumps(payload, indent=2, ensure_ascii=False))
    return losshistory, train_state


def main():
    args = parse_args()
    case_config = prepare_run(args)

    geom, data, metadata = build_data(args, case_config)
    model, net = build_model(args, data)
    initialize_geometry_parameters(args, net)

    if hasattr(data, "train_x_all") and data.train_x_all is not None:
        train_x_all = np.asarray(data.train_x_all)
        args._domain_points_for_plot = train_x_all[:, :2]
    else:
        args._domain_points_for_plot = geom.random_points(min(args.num_domain, 4000))

    losshistory = None
    if is_compact_material_method(args.method) and args.staged_training and args.iterations > 1:
        warmup_iterations = min(max(args.warmup_iterations, 0), max(args.iterations - 1, 0))
        remaining_iterations = args.iterations - warmup_iterations
        geometry_stage_iterations = min(max(args.geometry_stage_iterations, 0), max(remaining_iterations - 1, 0))
        remaining_iterations -= geometry_stage_iterations
        material_stage_iterations = min(max(args.material_stage_iterations, 0), max(remaining_iterations - 1, 0))
        remaining_iterations -= material_stage_iterations
        refinement_stage_iterations = min(max(args.refinement_stage_iterations, 0), max(remaining_iterations - 1, 0))
        main_iterations = args.iterations - warmup_iterations - geometry_stage_iterations - material_stage_iterations - refinement_stage_iterations
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
            save_stage_loss_artifacts(warmup_history, args.save_dir, "warmup", args=args)
            save_json(
                os.path.join(args.save_dir, "json", "stage_training_plan.json"),
                {
                    "staged_training": True,
                    "warmup_iterations": warmup_iterations,
                    "geometry_stage_iterations": geometry_stage_iterations,
                    "material_stage_iterations": material_stage_iterations,
                    "refinement_stage_iterations": refinement_stage_iterations,
                    "main_iterations": main_iterations,
                    "warmup_lr": args.warmup_lr,
                    "main_lr": args.main_lr,
                    "warmup_physics_scale": args.warmup_physics_scale,
                    "warmup_reg_scale": args.warmup_reg_scale,
                    "freeze_material_warmup": args.freeze_material_warmup,
                    "geometry_stage_lr": args.geometry_stage_lr,
                    "geometry_stage_physics_scale": args.geometry_stage_physics_scale,
                    "geometry_stage_reg_scale": args.geometry_stage_reg_scale,
                    "geometry_stage_data_scale": args.geometry_stage_data_scale,
                    "geometry_stage_boundary_scale": args.geometry_stage_boundary_scale,
                    "geometry_stage_balance_target": args.geometry_stage_balance_target,
                    "geometry_stage_balance_weight": args.geometry_stage_balance_weight,
                    "geometry_stage_binary_weight": args.geometry_stage_binary_weight,
                    "geometry_stage_interface_sharpness": args.geometry_stage_interface_sharpness,
                    "freeze_region_geometry_stage": args.freeze_region_geometry_stage,
                    "freeze_state_geometry_stage": args.freeze_state_geometry_stage,
                    "material_stage_lr": args.material_stage_lr,
                    "material_stage_physics_scale": args.material_stage_physics_scale,
                    "material_stage_reg_scale": args.material_stage_reg_scale,
                    "material_stage_usage_floor": args.material_stage_usage_floor,
                    "material_stage_usage_weight": args.material_stage_usage_weight,
                    "material_stage_binary_weight": args.material_stage_binary_weight,
                    "material_stage_contrast_weight": args.material_stage_contrast_weight,
                    "material_stage_lambda_gap_target": args.material_stage_lambda_gap_target,
                    "material_stage_mu_gap_target": args.material_stage_mu_gap_target,
                    "geometry_prior_weight": args.geometry_prior_weight,
                    "layer_y_prior_target": args.layer_y_prior_target,
                    "layer_y_init": args.layer_y_init,
                    "freeze_geometry_material_stage": args.freeze_geometry_material_stage,
                    "freeze_geometry_main": args.freeze_geometry_main,
                    "material_stage_interface_sharpness": args.material_stage_interface_sharpness,
                    "main_stage_interface_sharpness": args.main_stage_interface_sharpness,
                    "freeze_state_material_stage": args.freeze_state_material_stage,
                    "refinement_stage_lr": args.refinement_stage_lr,
                    "refinement_stage_physics_scale": args.refinement_stage_physics_scale,
                    "refinement_stage_reg_scale": args.refinement_stage_reg_scale,
                    "refinement_stage_data_scale": args.refinement_stage_data_scale,
                    "refinement_stage_boundary_scale": args.refinement_stage_boundary_scale,
                    "freeze_geometry_refinement": args.freeze_geometry_refinement,
                    "refinement_stage_interface_sharpness": args.refinement_stage_interface_sharpness,
                    "adaptive_main_stage": args.adaptive_main_stage,
                    "adaptive_main_chunks": args.adaptive_main_chunks,
                    "adaptive_main_base_physics_scale": args.adaptive_main_base_physics_scale,
                    "adaptive_main_min_physics_scale": args.adaptive_main_min_physics_scale,
                    "adaptive_main_max_physics_scale": args.adaptive_main_max_physics_scale,
                    "adaptive_main_scale_up": args.adaptive_main_scale_up,
                    "adaptive_main_scale_down": args.adaptive_main_scale_down,
                    "adaptive_main_obs_guard": args.adaptive_main_obs_guard,
                    "adaptive_main_reg_scale": args.adaptive_main_reg_scale,
                    "adaptive_main_data_scale": args.adaptive_main_data_scale,
                    "adaptive_main_boundary_scale": args.adaptive_main_boundary_scale,
                    "main_correction_scale_start": args.main_correction_scale_start,
                    "main_correction_scale_end": args.main_correction_scale_end,
                    "main_correction_chunks": args.main_correction_chunks,
                    "main_stage_physics_scale_start": args.main_stage_physics_scale_start,
                    "main_stage_physics_scale_end": args.main_stage_physics_scale_end,
                    "main_stage_reg_scale_start": args.main_stage_reg_scale_start,
                    "main_stage_reg_scale_end": args.main_stage_reg_scale_end,
                    "main_stage_data_scale_start": args.main_stage_data_scale_start,
                    "main_stage_data_scale_end": args.main_stage_data_scale_end,
                    "main_stage_boundary_scale_start": args.main_stage_boundary_scale_start,
                    "main_stage_boundary_scale_end": args.main_stage_boundary_scale_end,
                    "main_stage_lr_start": args.main_stage_lr_start,
                    "main_stage_lr_end": args.main_stage_lr_end,
                    "geometry_stage_domain_points": args.geometry_stage_domain_points,
                    "material_stage_domain_points": args.material_stage_domain_points,
                },
            )
            set_material_branch_trainable(net, True)
        geometry_stage_payload = run_geometry_stage(args, net, geom, data, case_config, metadata, args.save_dir)
        if geometry_stage_payload is not None:
            print("[geometry_stage] completed with best total loss {:.4e}".format(geometry_stage_payload["best"]["total_loss"]))
        material_stage_payload = run_material_stage(args, net, geom, data, case_config, args.save_dir)
        if material_stage_payload is not None:
            print("[material_stage] completed with best total loss {:.4e}".format(material_stage_payload["best"]["total_loss"]))
        if args.freeze_geometry_main:
            set_geometry_branch_trainable(net, False)
            set_region_parameter_trainable(net, True)
        original_sharpness = getattr(net, 'interface_sharpness', None)
        if hasattr(net, 'interface_sharpness') and args.main_stage_interface_sharpness > 0:
            net.interface_sharpness = float(args.main_stage_interface_sharpness)
        if args.adaptive_main_stage and main_iterations > 0:
            main_history, train_state = run_adaptive_main_stage(
                args=args,
                model=model,
                net=net,
                geom=geom,
                data=data,
                case_config=case_config,
                metadata=metadata,
                save_dir=args.save_dir,
                main_iterations=main_iterations,
            )
        else:
            main_history, train_state = run_main_stage_with_correction_schedule(
                args=args,
                model=model,
                net=net,
                metadata=metadata,
                save_dir=args.save_dir,
                main_iterations=max(main_iterations, 1),
            )
        losshistory = main_history
        if refinement_stage_iterations > 0:
            save_stage_loss_artifacts(main_history, args.save_dir, 'main', args=args)
            if args.freeze_geometry_refinement:
                set_geometry_branch_trainable(net, False)
                set_region_parameter_trainable(net, True)
            if original_sharpness is not None and args.refinement_stage_interface_sharpness > 0:
                net.interface_sharpness = float(args.refinement_stage_interface_sharpness)
            refinement_weights = resolve_loss_weights(
                args,
                physics_scale=args.refinement_stage_physics_scale,
                reg_scale=args.refinement_stage_reg_scale,
                data_scale=args.refinement_stage_data_scale,
                boundary_scale=args.refinement_stage_boundary_scale,
            )
            model.compile("adam", lr=args.refinement_stage_lr, loss_weights=refinement_weights)
            refinement_callbacks = make_callbacks(args, args.save_dir, metadata)
            refinement_history, train_state = model.train(
                iterations=refinement_stage_iterations,
                display_every=max(100, min(args.display_every, refinement_stage_iterations)),
                callbacks=refinement_callbacks,
            )
            save_stage_loss_artifacts(refinement_history, args.save_dir, 'refinement', args=args)
            losshistory = refinement_history
            if args.freeze_geometry_refinement:
                set_geometry_branch_trainable(net, True)
                set_region_parameter_trainable(net, True)
        if original_sharpness is not None:
            net.interface_sharpness = float(original_sharpness)
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
        "staged_training": bool(args.staged_training and is_compact_material_method(args.method)),
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
