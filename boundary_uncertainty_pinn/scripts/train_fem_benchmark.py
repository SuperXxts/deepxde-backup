#!/usr/bin/env python3
"""DeepXDE runner for the B-group FEM geotechnical benchmark."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("DDE_BACKEND", "pytorch")


CASE_CHOICES = tuple(f"B{i}" for i in range(8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--case", required=True, choices=CASE_CHOICES)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--display-every", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--num-domain", type=int, default=2000)
    parser.add_argument("--num-boundary", type=int, default=600)
    parser.add_argument("--obs-count", type=int, default=120)
    parser.add_argument("--val-count", type=int, default=500)
    parser.add_argument("--test-nx", type=int, default=121)
    parser.add_argument("--test-ny", type=int, default=61)
    parser.add_argument("--anchor-count", type=int, default=4)
    parser.add_argument("--reaction-points", type=int, default=80)
    parser.add_argument("--width", type=int, default=56)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--material-width", type=int, default=56)
    parser.add_argument("--material-depth", type=int, default=4)
    parser.add_argument("--wide-width", type=int, default=96)
    parser.add_argument("--wide-depth", type=int, default=5)
    parser.add_argument("--boundary-modes", type=int, default=1)
    parser.add_argument("--projection-ridge", type=float, default=1.0e-8)
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--fixed-A", type=float, default=0.85)
    parser.add_argument("--init-A", type=float, default=0.85)
    parser.add_argument("--physics-weight", type=float, default=1.0)
    parser.add_argument("--momentum-weight", type=float, default=None)
    parser.add_argument("--constitutive-weight", type=float, default=None)
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--obs-weight", type=float, default=1.0)
    parser.add_argument("--obs-material-weight", type=float, default=20.0)
    parser.add_argument("--obs-boundary-weight", type=float, default=20.0)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--reaction-weight", type=float, default=5.0)
    parser.add_argument("--gradient-diagnostic-period", type=int, default=0)
    parser.add_argument("--eval-checkpoint", choices=("best", "final"), default="best")
    parser.add_argument("--run-note", type=str, default="")
    parser.add_argument("--fem-data-dir", type=Path, default=None)
    parser.add_argument("--fem-nx", type=int, default=96)
    parser.add_argument("--fem-ny", type=int, default=48)
    parser.add_argument("--domain-width", type=float, default=6.0)
    parser.add_argument("--domain-depth", type=float, default=3.0)
    parser.add_argument("--plate-width", type=float, default=1.0)
    parser.add_argument("--settlement", type=float, default=-0.05)
    parser.add_argument("--k-ref", type=float, default=1.0)
    parser.add_argument("--mu-ref", type=float, default=0.45)
    parser.add_argument(
        "--deepxde-root",
        type=Path,
        default=Path("/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/deepxde-master"),
    )
    return parser.parse_args()


def add_paths(args: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    if args.deepxde_root.exists():
        sys.path.insert(0, str(args.deepxde_root))


def main() -> None:
    args = parse_args()
    add_paths(args)

    import deepxde as dde
    import numpy as np
    import torch

    from bupinn.decoupling import ObservationProjector, ReactionIntegral
    from bupinn.fem2d import (
        FEMConfig,
        FEMInterpolator,
        add_displacement_noise,
        fem_observation_design_np,
        load_fem_result,
        make_anchor_points,
        make_observation_points,
        make_reaction_points,
        make_validation_points,
        material_field,
        plate_mode_values_np,
        save_fem_result,
        solve_fem,
        structured_grid_points,
    )
    from bupinn.fem_cases import fem_case_description, fem_case_markdown_table, fem_case_settings
    from bupinn.io import ensure_run_dirs, save_table, write_json
    from bupinn.networks import BoundaryMaterialPFNN
    from bupinn.plotting import (
        plot_amplitude_history,
        plot_boundary_curve,
        plot_displacement_vector,
        plot_field_triplet,
        plot_k_mu_comparison,
        plot_line_slice,
        plot_loss_arrays,
        plot_loss_history,
        plot_material_error_history,
        plot_sampling,
    )

    settings = fem_case_settings(args.case)
    dirs = ensure_run_dirs(args.run_dir)

    np.random.seed(args.seed)
    dde.config.set_random_seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    fem_config = FEMConfig(
        width=args.domain_width,
        depth=args.domain_depth,
        plate_width=args.plate_width,
        nx=args.fem_nx,
        ny=args.fem_ny,
        settlement=args.settlement,
        seed=args.seed,
        k_ref=args.k_ref,
        mu_ref=args.mu_ref,
    )
    fem_dir = args.fem_data_dir or (args.run_dir.parent / "_fem_data")
    fem_npz = fem_dir / "fem_solution.npz"
    if fem_npz.exists():
        fem_result = load_fem_result(fem_npz)
    else:
        fem_result = solve_fem(fem_config)
        save_fem_result(fem_result, fem_dir)
    interp = FEMInterpolator(fem_result)

    geom = dde.geometry.Rectangle([0.0, 0.0], [fem_config.width, fem_config.depth])
    k0 = float(args.k_ref)
    mu0 = float(args.mu_ref)
    true_A = float(fem_config.settlement)
    fixed_A = float(args.fixed_A) * true_A
    init_A = float(args.init_A) * true_A

    def lambda_from_k_mu(k, mu):
        return k - 2.0 * mu / 3.0

    def stress_from_k_mu(k, mu, exx, eyy, exy):
        lam = lambda_from_k_mu(k, mu)
        sxx = (2.0 * mu + lam) * exx + lam * eyy
        syy = lam * exx + (2.0 * mu + lam) * eyy
        sxy = 2.0 * mu * exy
        return sxx, syy, sxy

    def material_from_output(y):
        zk = torch.clamp(y[:, 5:6], min=-3.0, max=3.0)
        zmu = torch.clamp(y[:, 6:7], min=-3.0, max=3.0)
        return k0 * torch.exp(zk), mu0 * torch.exp(zmu)

    def assemble_fields(raw: np.ndarray) -> np.ndarray:
        pred_k = k0 * np.exp(np.clip(raw[:, 5:6], -3.0, 3.0))
        pred_mu = mu0 * np.exp(np.clip(raw[:, 6:7], -3.0, 3.0))
        pred_e = 9.0 * pred_k * pred_mu / (3.0 * pred_k + pred_mu)
        pred_nu = (3.0 * pred_k - 2.0 * pred_mu) / (2.0 * (3.0 * pred_k + pred_mu))
        return np.column_stack([raw[:, 0:5], pred_k, pred_mu, pred_e, pred_nu])

    def predict_constitutive_stress_np(x_np: np.ndarray) -> np.ndarray:
        device = next(net.parameters()).device
        dtype = next(net.parameters()).dtype
        was_training = bool(getattr(net, "training", False))
        net.eval()
        x_tensor = torch.as_tensor(x_np, dtype=dtype, device=device).requires_grad_(True)
        with torch.enable_grad():
            raw = net(x_tensor)

            def jac(component: int, coord: int):
                values = raw[:, component : component + 1]
                grads = torch.autograd.grad(
                    values,
                    x_tensor,
                    grad_outputs=torch.ones_like(values),
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=False,
                )[0]
                return grads[:, coord : coord + 1]

            ux_x = jac(0, 0)
            uy_y = jac(1, 1)
            ux_y = jac(0, 1)
            uy_x = jac(1, 0)
            k_pred, mu_pred = material_from_output(raw)
            stress = torch.cat(stress_from_k_mu(k_pred, mu_pred, ux_x, uy_y, 0.5 * (ux_y + uy_x)), dim=1)
        if was_training:
            net.train()
        return stress.detach().cpu().numpy()

    def relative_l2(y_true: np.ndarray, y_pred: np.ndarray, names: list[str]) -> dict[str, float]:
        out = {}
        for i, name in enumerate(names):
            denom = np.linalg.norm(y_true[:, i]) + 1.0e-12
            out[name] = float(np.linalg.norm(y_pred[:, i] - y_true[:, i]) / denom)
        return out

    def pde(x, y):
        ux_x = dde.grad.jacobian(y, x, i=0, j=0)
        uy_y = dde.grad.jacobian(y, x, i=1, j=1)
        ux_y = dde.grad.jacobian(y, x, i=0, j=1)
        uy_x = dde.grad.jacobian(y, x, i=1, j=0)
        exx = ux_x
        eyy = uy_y
        exy = 0.5 * (ux_y + uy_x)
        k_pred, mu_pred = material_from_output(y)
        sxx_c, syy_c, sxy_c = stress_from_k_mu(k_pred, mu_pred, exx, eyy, exy)
        sxx_x = dde.grad.jacobian(sxx_c, x, i=0, j=0)
        syy_y = dde.grad.jacobian(syy_c, x, i=0, j=1)
        sxy_x = dde.grad.jacobian(sxy_c, x, i=0, j=0)
        sxy_y = dde.grad.jacobian(sxy_c, x, i=0, j=1)
        return [
            sxx_x + sxy_y,
            sxy_x + syy_y,
            sxx_c - y[:, 2:3],
            syy_c - y[:, 3:4],
            sxy_c - y[:, 4:5],
        ]

    def boundary_bottom(x, on_boundary):
        return on_boundary and dde.utils.isclose(x[1], 0.0)

    def boundary_left(x, on_boundary):
        return on_boundary and dde.utils.isclose(x[0], 0.0)

    def boundary_right(x, on_boundary):
        return on_boundary and dde.utils.isclose(x[0], fem_config.width)

    def boundary_plate(x, on_boundary):
        return (
            on_boundary
            and dde.utils.isclose(x[1], fem_config.depth)
            and x[0] >= fem_config.plate_left - 1.0e-10
            and x[0] <= fem_config.plate_right + 1.0e-10
        )

    def boundary_top_free(x, on_boundary):
        return (
            on_boundary
            and dde.utils.isclose(x[1], fem_config.depth)
            and (x[0] < fem_config.plate_left - 1.0e-10 or x[0] > fem_config.plate_right + 1.0e-10)
        )

    def side_sxy_bc(inputs, outputs, _X):
        return outputs[:, 4:5]

    def top_syy_bc(inputs, outputs, _X):
        return outputs[:, 3:4]

    def top_sxy_bc(inputs, outputs, _X):
        return outputs[:, 4:5]

    external_vars = []
    amplitude_var = None
    if settings["learnable_A"]:
        amplitude_var = dde.Variable(init_A)
        external_vars.append(amplitude_var)

    net = None

    def plate_shape_torch(inputs):
        return torch.ones_like(inputs[:, 0:1])

    def top_target(inputs):
        top_mode = str(settings["top"])
        if top_mode == "true":
            return true_A * plate_shape_torch(inputs)
        if top_mode == "wrong":
            return fixed_A * plate_shape_torch(inputs)
        if top_mode == "learnable_A":
            return amplitude_var * plate_shape_torch(inputs)
        if top_mode == "network_top":
            return net.top_boundary_value(inputs)
        raise ValueError(f"Unknown top mode: {top_mode}")

    def plate_uy_bc(inputs, outputs, _X):
        return outputs[:, 1:2] - top_target(inputs)

    def plate_ux_bc(inputs, outputs, _X):
        return outputs[:, 0:1]

    bcs = [
        dde.icbc.DirichletBC(geom, lambda x: 0.0, boundary_bottom, component=0),
        dde.icbc.DirichletBC(geom, lambda x: 0.0, boundary_bottom, component=1),
        dde.icbc.DirichletBC(geom, lambda x: 0.0, boundary_left, component=0),
        dde.icbc.DirichletBC(geom, lambda x: 0.0, boundary_right, component=0),
        dde.icbc.OperatorBC(geom, side_sxy_bc, boundary_left),
        dde.icbc.OperatorBC(geom, side_sxy_bc, boundary_right),
        dde.icbc.OperatorBC(geom, top_syy_bc, boundary_top_free),
        dde.icbc.OperatorBC(geom, top_sxy_bc, boundary_top_free),
        dde.icbc.OperatorBC(geom, plate_ux_bc, boundary_plate),
        dde.icbc.OperatorBC(geom, plate_uy_bc, boundary_plate),
    ]
    loss_terms = [
        "momentum_x",
        "momentum_y",
        "constitutive_sxx",
        "constitutive_syy",
        "constitutive_sxy",
        "bottom_ux",
        "bottom_uy",
        "left_ux",
        "right_ux",
        "left_sxy",
        "right_sxy",
        "top_free_syy",
        "top_free_sxy",
        "plate_ux",
        "plate_uy",
    ]

    obs_x = make_observation_points(fem_config, args.obs_count, args.seed)
    obs_u = interp.evaluate(obs_x)[:, 0:2]
    obs_u = add_displacement_noise(obs_u, args.noise_level, args.seed)
    design = fem_observation_design_np(obs_x, int(args.boundary_modes), fem_config)

    if settings["decoupled_obs"]:
        obs_projector = ObservationProjector(obs_x, obs_u, design, args.projection_ridge)
        zeros = np.zeros((len(obs_x), 1), dtype=np.float32)

        def obs_material_ux(inputs, outputs, _X):
            return obs_projector.projected_component(
                inputs, outputs, _X, 0, "material", update_path="material", occurrence=0
            )

        def obs_material_uy(inputs, outputs, _X):
            return obs_projector.projected_component(
                inputs, outputs, _X, 1, "material", update_path="material", occurrence=1
            )

        def obs_boundary_uy(inputs, outputs, _X):
            return obs_projector.projected_component(
                inputs, outputs, _X, 1, "boundary", update_path="boundary", occurrence=2
            )

        bcs.extend(
            [
                dde.icbc.PointSetOperatorBC(obs_x, zeros, obs_material_ux),
                dde.icbc.PointSetOperatorBC(obs_x, zeros, obs_material_uy),
                dde.icbc.PointSetOperatorBC(obs_x, zeros, obs_boundary_uy),
            ]
        )
        loss_terms.extend(["obs_material_ux", "obs_material_uy", "obs_boundary_uy"])
    else:
        bcs.append(dde.icbc.PointSetBC(obs_x, obs_u, component=[0, 1]))
        loss_terms.append("obs_u")

    anchor_x = make_anchor_points(fem_config, args.anchor_count if settings["anchor"] else 0)
    anchor_u = interp.evaluate(anchor_x)[:, 0:2] if len(anchor_x) else np.empty((0, 2), dtype=np.float32)
    if len(anchor_x):
        bcs.append(dde.icbc.PointSetBC(anchor_x, anchor_u, component=[0, 1]))
        loss_terms.append("anchor_u")

    reaction_x, reaction_weights = make_reaction_points(fem_config, args.reaction_points)
    true_reaction_y = float(fem_result.total_reaction_y)

    class ConstitutiveReactionIntegral(ReactionIntegral):
        def __call__(self, inputs, outputs, X):
            weights = self._weights(outputs.device, outputs.dtype)
            row_index_np = self._row_index(X)
            row_index = torch.as_tensor(row_index_np, dtype=torch.long, device=outputs.device)
            ux_x = dde.grad.jacobian(outputs, inputs, i=0, j=0)
            uy_y = dde.grad.jacobian(outputs, inputs, i=1, j=1)
            ux_y = dde.grad.jacobian(outputs, inputs, i=0, j=1)
            uy_x = dde.grad.jacobian(outputs, inputs, i=1, j=0)
            k_pred, mu_pred = material_from_output(outputs)
            _, syy_c, _ = stress_from_k_mu(k_pred, mu_pred, ux_x, uy_y, 0.5 * (ux_y + uy_x))
            resultant = torch.sum(syy_c[row_index] * weights)
            result = torch.zeros((outputs.shape[0], 1), dtype=outputs.dtype, device=outputs.device)
            result[row_index, 0:1] = resultant
            return result

    if settings["reaction"]:
        target_reaction = np.full((len(reaction_x), 1), true_reaction_y, dtype=np.float32)
        reaction_operator = ConstitutiveReactionIntegral(reaction_weights, boundary_y=fem_config.depth)
        bcs.append(dde.icbc.PointSetOperatorBC(reaction_x, target_reaction, reaction_operator))
        loss_terms.append("global_reaction_y")

    data = dde.data.PDE(
        geom,
        pde,
        bcs,
        num_domain=args.num_domain,
        num_boundary=args.num_boundary,
        train_distribution="Hammersley",
        num_test=None,
    )

    if settings["net"] == "pfnn":
        width = args.wide_width if settings["wide"] else args.width
        depth = args.wide_depth if settings["wide"] else args.depth
        layer_sizes = [2] + [[int(width)] * 7 for _ in range(int(depth))] + [7]
        net = dde.nn.PFNN(layer_sizes, "tanh", "Glorot normal")
        network_config = {"type": "DeepXDE PFNN", "layer_sizes": layer_sizes}
    else:
        init_beta = [init_A] + [0.0] * (max(1, int(args.boundary_modes)) - 1)
        net = BoundaryMaterialPFNN(
            dde,
            state_width=args.width,
            state_depth=args.depth,
            material_width=args.material_width,
            material_depth=args.material_depth,
            num_boundary_modes=int(args.boundary_modes),
            init_beta=init_beta[: int(args.boundary_modes)],
            trainable_boundary=bool(settings["boundary_layer"]),
            use_boundary_layer=bool(settings["boundary_layer"]),
            boundary_mode_type="plate",
            domain_x_min=0.0,
            domain_x_max=fem_config.width,
            domain_y_bottom=0.0,
            domain_y_top=fem_config.depth,
            plate_center=fem_config.plate_center,
            plate_width=fem_config.plate_width,
            boundary_mode_scale=1.0,
        )
        network_config = {
            "type": "BoundaryMaterialPFNN",
            "state_width": args.width,
            "state_depth": args.depth,
            "material_width": args.material_width,
            "material_depth": args.material_depth,
            "num_boundary_modes": int(args.boundary_modes),
            "use_boundary_layer": bool(settings["boundary_layer"]),
            "boundary_mode_type": "plate",
        }

    model = dde.Model(data, net)

    momentum_weight = float(args.physics_weight if args.momentum_weight is None else args.momentum_weight)
    constitutive_weight = float(args.physics_weight if args.constitutive_weight is None else args.constitutive_weight)
    loss_weights: list[float] = [momentum_weight] * 2 + [constitutive_weight] * 3
    loss_weights.extend([float(args.boundary_weight)] * 10)
    if settings["decoupled_obs"]:
        loss_weights.extend(
            [
                float(args.obs_material_weight),
                float(args.obs_material_weight),
                float(args.obs_boundary_weight),
            ]
        )
    else:
        loss_weights.append(float(args.obs_weight))
    if settings["anchor"]:
        loss_weights.append(float(args.anchor_weight))
    if settings["reaction"]:
        loss_weights.append(float(args.reaction_weight))
    if len(loss_weights) != len(loss_terms):
        raise RuntimeError(f"loss_weights length {len(loss_weights)} != loss_terms length {len(loss_terms)}")

    model.compile("adam", lr=args.lr, loss_weights=loss_weights, external_trainable_variables=external_vars)

    def _trainable_parameters(module) -> list[torch.nn.Parameter]:
        if module is None:
            return []
        return [p for p in module.parameters() if p.requires_grad]

    def _gradient_l2(loss, parameters: list[torch.nn.Parameter], retain_graph: bool) -> float:
        if not parameters or not getattr(loss, "requires_grad", False):
            return 0.0
        grads = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
        total = None
        for grad in grads:
            if grad is None:
                continue
            value = torch.sum(grad.detach() ** 2)
            total = value if total is None else total + value
        if total is None:
            return 0.0
        return float(torch.sqrt(total + 1.0e-30).detach().cpu().item())

    def _loss_group_indices() -> dict[str, list[int]]:
        groups = {
            "momentum": [i for i, name in enumerate(loss_terms) if name.startswith("momentum_")],
            "constitutive": [i for i, name in enumerate(loss_terms) if name.startswith("constitutive_")],
            "essential_boundary": [
                i
                for i, name in enumerate(loss_terms)
                if name
                in {
                    "bottom_ux",
                    "bottom_uy",
                    "left_ux",
                    "right_ux",
                    "left_sxy",
                    "right_sxy",
                    "top_free_syy",
                    "top_free_sxy",
                    "plate_ux",
                    "plate_uy",
                }
            ],
            "obs_material": [i for i, name in enumerate(loss_terms) if name.startswith("obs_material_")],
            "obs_boundary": [i for i, name in enumerate(loss_terms) if name.startswith("obs_boundary_")],
            "obs_u": [i for i, name in enumerate(loss_terms) if name == "obs_u"],
            "anchor": [i for i, name in enumerate(loss_terms) if name == "anchor_u"],
            "reaction": [i for i, name in enumerate(loss_terms) if "reaction" in name],
        }
        return {name: indices for name, indices in groups.items() if indices}

    class GradientDiagnosticLogger(dde.callbacks.Callback):
        def __init__(self, period: int):
            super().__init__()
            self.period = max(1, int(period))
            self.rows: list[dict[str, object]] = []
            self.groups = _loss_group_indices()

        def _parameter_groups(self) -> dict[str, list[torch.nn.Parameter]]:
            if hasattr(net, "state_net") and hasattr(net, "material_net"):
                groups = {
                    "state": _trainable_parameters(net.state_net),
                    "material": _trainable_parameters(net.material_net),
                    "boundary": [net.beta]
                    if isinstance(getattr(net, "beta", None), torch.nn.Parameter) and net.beta.requires_grad
                    else [],
                }
            else:
                groups = {"all": [p for p in net.parameters() if p.requires_grad]}
            return {name: params for name, params in groups.items() if params}

        def _record(self, step: int):
            previous_weights = self.model.loss_weights
            try:
                self.model.loss_weights = None
                _, raw_losses = self.model.outputs_losses_train(
                    self.model.train_state.X_train,
                    self.model.train_state.y_train,
                    self.model.train_state.train_aux_vars,
                )
            finally:
                self.model.loss_weights = previous_weights
            if isinstance(raw_losses, (list, tuple)):
                raw_losses = torch.stack(list(raw_losses))
            parameter_groups = self._parameter_groups()
            row: dict[str, object] = {"step": int(step), "groups": {}}
            ordered_tasks: list[tuple[str, str, torch.Tensor]] = []
            for group_name, indices in self.groups.items():
                group_loss = torch.sum(raw_losses[indices])
                row["groups"][group_name] = {
                    "loss": float(group_loss.detach().cpu().item()),
                    "indices": [int(i) for i in indices],
                }
                for param_name in parameter_groups:
                    ordered_tasks.append((group_name, param_name, group_loss))
            for task_index, (group_name, param_name, group_loss) in enumerate(ordered_tasks):
                retain_graph = task_index < len(ordered_tasks) - 1
                grad_norm = _gradient_l2(group_loss, parameter_groups[param_name], retain_graph=retain_graph)
                row["groups"][group_name][f"{param_name}_grad_norm"] = grad_norm
            self.rows.append(row)
            write_json(dirs["json"] / "gradient_diagnostics.json", {"rows": self.rows, "loss_terms": loss_terms})
            print("GRAD_DIAG", step, {k: v for k, v in row["groups"].items()}, flush=True)

        def on_epoch_end(self):
            step = int(self.model.train_state.iteration)
            if step > 0 and step % self.period == 0:
                self._record(step)

    class ParameterLogger(dde.callbacks.Callback):
        def __init__(self, path: Path, period: int):
            super().__init__()
            self.path = path
            self.period = max(1, int(period))
            self.rows: list[list[float]] = []

        def _row(self, step: int) -> list[float]:
            if amplitude_var is not None:
                beta = [float(amplitude_var.detach().cpu().item())]
            elif hasattr(net, "beta"):
                beta = [float(v) for v in net.beta.detach().cpu().numpy().ravel()]
            elif str(settings["top"]) == "true":
                beta = [true_A]
            else:
                beta = [fixed_A]
            return [float(step)] + beta

        def on_train_begin(self):
            self.rows.append(self._row(0))

        def on_epoch_end(self):
            step = int(self.model.train_state.iteration)
            if step % self.period == 0:
                row = self._row(step)
                self.rows.append(row)
                print("STEP", step, "BETA", " ".join(f"{v:.8f}" for v in row[1:]), flush=True)

        def on_train_end(self):
            max_len = max(len(row) for row in self.rows)
            arr = np.asarray([row + [np.nan] * (max_len - len(row)) for row in self.rows], dtype=float)
            header = "step " + " ".join(f"beta{i}" for i in range(max_len - 1))
            np.savetxt(self.path, arr, header=header)

    names = ["ux", "uy", "sxx", "syy", "sxy", "K", "mu", "E", "nu"]
    validation_x_for_monitor = make_validation_points(fem_config, args.val_count, args.seed)
    validation_truth_for_monitor = interp.evaluate(validation_x_for_monitor)

    class DynamicFigureLogger(dde.callbacks.Callback):
        def __init__(self, period: int):
            super().__init__()
            self.period = max(1, int(period))
            self.steps: list[int] = []
            self.loss_train: list[np.ndarray] = []
            self.loss_test: list[np.ndarray] = []
            self.material_rows: list[list[float]] = []

        def _record(self, step: int):
            loss_train = np.asarray(self.model.train_state.loss_train, dtype=float)
            loss_test = np.asarray(self.model.train_state.loss_test, dtype=float)
            self.steps.append(step)
            self.loss_train.append(loss_train)
            self.loss_test.append(loss_test)
            plot_loss_arrays(
                np.asarray(self.steps),
                np.asarray(self.loss_train),
                np.asarray(self.loss_test),
                loss_terms,
                dirs["png"] / "损失历史图.png",
                dirs["png"] / "损失分量图.png",
                dirs["npz"] / "动态损失历史.npz",
            )
            np.savetxt(
                dirs["dat"] / "动态损失历史.dat",
                np.column_stack([np.asarray(self.steps), np.asarray(self.loss_train)]),
                header="step train_loss_terms",
            )
            raw = self.model.predict(validation_x_for_monitor)
            pred = assemble_fields(raw)
            rel = relative_l2(validation_truth_for_monitor, pred, names)
            row = [float(step), rel["K"], rel["mu"], rel["E"], rel["nu"]]
            self.material_rows.append(row)
            arr = np.asarray(self.material_rows, dtype=float)
            np.savetxt(dirs["dat"] / "材料参数演化.dat", arr, header="step K_rel_l2 mu_rel_l2 E_rel_l2 nu_rel_l2")
            plot_material_error_history(arr, dirs["png"] / "材料参数演化图.png")

        def on_epoch_end(self):
            step = int(self.model.train_state.iteration)
            if step % self.period == 0 and self.model.train_state.loss_train is not None:
                self._record(step)

        def on_train_end(self):
            if not self.steps and self.model.train_state.loss_train is not None:
                self._record(int(self.model.train_state.iteration))

    param_log = ParameterLogger(dirs["dat"] / "boundary_parameter_history.dat", args.display_every)
    dynamic_figures = DynamicFigureLogger(args.display_every)
    callbacks = [param_log, dynamic_figures]
    if int(args.gradient_diagnostic_period) > 0:
        callbacks.append(GradientDiagnosticLogger(args.gradient_diagnostic_period))
    checkpoint = dde.callbacks.ModelCheckpoint(
        str(dirs["model"] / "best_model"),
        verbose=0,
        save_better_only=True,
        period=max(1, args.checkpoint_every),
        monitor="train loss",
    )
    callbacks.append(checkpoint)

    config = vars(args).copy()
    config.update(
        {
            "case_settings": settings,
            "backend": dde.backend.backend_name,
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "torch_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "network": network_config,
            "material_parameterization": "dual: K=K_ref*exp(zK), mu=mu_ref*exp(zmu)",
            "fem_config": fem_config.__dict__,
            "true_settlement": true_A,
            "fixed_settlement": fixed_A,
            "init_settlement": init_A,
            "true_reaction_y": true_reaction_y,
            "loss_terms": loss_terms,
            "loss_weights": loss_weights,
            "field_names": names,
        }
    )
    write_json(dirs["json"] / "config.json", config)
    write_json(dirs["json"] / "run_config.json", config)
    (args.run_dir / "实验说明.txt").write_text(fem_case_description(args.case), encoding="utf-8")
    (args.run_dir / "B组实验矩阵.md").write_text(fem_case_markdown_table(), encoding="utf-8")
    if str(args.run_note).strip():
        (args.run_dir / "变体说明.txt").write_text(str(args.run_note).strip() + "\n", encoding="utf-8")

    plot_sampling(obs_x, anchor_x, dirs["png"] / "采样点分布图.png")
    save_table(dirs["dat"] / "observation_points.dat", np.hstack([obs_x, obs_u]), "x y ux uy")
    save_table(dirs["dat"] / "reaction_points.dat", np.hstack([reaction_x, reaction_weights]), "x y integration_weight")
    if len(anchor_x):
        save_table(dirs["dat"] / "anchor_points.dat", np.hstack([anchor_x, anchor_u]), "x y ux uy")
    np.savez(dirs["npz"] / "observation_projection_data.npz", obs_x=obs_x, design=design)

    start = time.time()
    losshistory, train_state = model.train(
        iterations=args.iterations,
        display_every=args.display_every,
        callbacks=callbacks,
    )
    elapsed = time.time() - start
    final_model_path = model.save(str(dirs["model"] / "final_model"))
    eval_checkpoint_path = final_model_path
    if args.eval_checkpoint == "best":
        best_model_path = dirs["model"] / f"best_model-{int(train_state.best_step)}.pt"
        if best_model_path.exists():
            model.restore(str(best_model_path), verbose=0)
            eval_checkpoint_path = str(best_model_path)

    plot_loss_history(losshistory, dirs["png"] / "损失历史图.png", dirs["npz"] / "loss_history.npz", loss_terms)
    if param_log.rows:
        param_data = np.loadtxt(dirs["dat"] / "boundary_parameter_history.dat", comments="#")
        if param_data.ndim == 1:
            param_data = param_data.reshape(1, -1)
        np.savetxt(dirs["dat"] / "amplitude_history.dat", param_data[:, :2], header="step A_or_beta0")
        plot_amplitude_history(dirs["dat"] / "amplitude_history.dat", dirs["png"] / "边界参数演化图.png", true_A)

    loss_steps = np.asarray(losshistory.steps)
    loss_train = np.asarray(losshistory.loss_train, dtype=float)
    loss_test = np.asarray(losshistory.loss_test, dtype=float)
    np.savetxt(
        dirs["dat"] / "loss_history.dat",
        np.column_stack([loss_steps, loss_train, loss_test]) if loss_train.size else np.empty((0, 1)),
        header="step train_loss_terms test_loss_terms",
    )
    write_json(
        dirs["json"] / "loss_history.json",
        {"steps": loss_steps, "loss_train": loss_train, "loss_test": loss_test, "loss_terms": loss_terms},
    )

    split_defs = {
        "evaluation": structured_grid_points(args.test_nx, args.test_ny, fem_config.width, fem_config.depth),
        "validation": validation_x_for_monitor,
        "train": obs_x,
    }
    split_metrics: dict[str, dict[str, float]] = {}
    eval_payload = {}
    for split_name, x_split in split_defs.items():
        raw = model.predict(x_split)
        pred = assemble_fields(raw)
        truth = interp.evaluate(x_split)
        split_metrics[split_name] = relative_l2(truth, pred, names)
        np.savez(
            dirs["npz"] / f"{split_name}_predictions_data.npz",
            x=x_split,
            y_true=truth,
            y_pred=pred,
            y_pred_raw=raw,
            field_names=np.asarray(names),
        )
        save_table(
            dirs["txt"] / f"{split_name}_predictions_data.txt",
            np.column_stack([x_split, truth, pred, pred - truth]),
            "x y true_fields pred_fields error_fields",
        )
        if split_name == "evaluation":
            for i, name in enumerate(names):
                np.savez(
                    dirs["npz"] / f"{split_name}_predictions_{name}.npz",
                    x=x_split,
                    y_true=truth[:, i],
                    y_pred=pred[:, i],
                    error=pred[:, i] - truth[:, i],
                )
                plot_field_triplet(x_split, truth[:, i], pred[:, i], name, dirs["png"] / f"评估_{name}_真值预测误差图.png")
        eval_payload[split_name] = {"x": x_split, "truth": truth, "pred": pred, "raw": raw}

    x_test = split_defs["evaluation"]
    pred = eval_payload["evaluation"]["pred"]
    truth = eval_payload["evaluation"]["truth"]
    plot_k_mu_comparison(x_test, truth, pred, dirs["png"] / "评估_K与μ真值预测对比图.png")
    plot_displacement_vector(x_test, truth, pred, dirs["png"] / "位移矢量场图_评估集.png")
    for y_value in (0.75, 1.50, 2.25):
        plot_line_slice(
            x_test,
            {"K": (truth[:, 5], pred[:, 5]), "mu": (truth[:, 6], pred[:, 6]), "nu": (truth[:, 8], pred[:, 8])},
            dirs["png"] / f"材料参数切片对比图_y{str(y_value).replace('.', 'p')}.png",
            y_value=y_value,
        )

    top_pred_raw = model.predict(reaction_x)
    top_pred_reaction_density_stress_branch = top_pred_raw[:, 3:4]
    top_pred_constitutive_stress = predict_constitutive_stress_np(reaction_x)
    top_pred_reaction_density_constitutive = top_pred_constitutive_stress[:, 1:2]
    pred_reaction_y_stress_branch = float(np.sum(top_pred_reaction_density_stress_branch * reaction_weights))
    pred_reaction_y_constitutive = float(np.sum(top_pred_reaction_density_constitutive * reaction_weights))
    pred_reaction_y = pred_reaction_y_constitutive
    reaction_rel_error = abs(pred_reaction_y - true_reaction_y) / (abs(true_reaction_y) + 1.0e-12)

    plot_boundary_curve(
        reaction_x,
        interp.evaluate(reaction_x)[:, 1],
        top_pred_raw[:, 1],
        "Plate vertical displacement",
        dirs["png"] / "加载板竖向位移曲线图.png",
    )
    # FEM total reaction is a scalar; the reference density curve is only a uniform visual proxy.
    reference_density = np.full((len(reaction_x),), true_reaction_y / fem_config.plate_width, dtype=np.float32)
    plot_boundary_curve(
        reaction_x,
        reference_density,
        top_pred_reaction_density_constitutive[:, 0],
        "Plate reaction density from constitutive stress",
        dirs["png"] / "本构应力加载板反力密度曲线图.png",
    )

    final_beta = param_log.rows[-1][1:] if param_log.rows else []
    metrics = {
        "case": args.case,
        "iterations": args.iterations,
        "elapsed_seconds": elapsed,
        "final_beta": [float(v) for v in final_beta],
        "final_A_or_beta0": float(final_beta[0]) if final_beta else None,
        "true_A": true_A,
        "relative_l2": split_metrics["evaluation"],
        "split_relative_l2": split_metrics,
        "true_reaction_y": true_reaction_y,
        "pred_reaction_y": pred_reaction_y,
        "pred_reaction_y_stress_branch": pred_reaction_y_stress_branch,
        "pred_reaction_y_constitutive": pred_reaction_y_constitutive,
        "reaction_rel_error": float(reaction_rel_error),
        "best_step": int(train_state.best_step),
        "eval_checkpoint": args.eval_checkpoint,
        "eval_checkpoint_path": eval_checkpoint_path,
        "final_model_path": final_model_path,
        "final_train_loss_sum": float(np.sum(train_state.loss_train)),
        "final_test_loss_sum": float(np.sum(train_state.loss_test)),
        "independent_test_loss_available": False,
        "loss_history_interval": int(args.display_every),
        "num_trainable_parameters": int(net.num_trainable_parameters())
        if hasattr(net, "num_trainable_parameters")
        else int(sum(p.numel() for p in net.parameters() if p.requires_grad)),
    }
    np.savez(
        dirs["npz"] / "eval_fields.npz",
        x_test=x_test,
        y_true=truth,
        y_pred=pred,
        obs_x=obs_x,
        obs_u=obs_u,
        anchor_x=anchor_x,
        anchor_u=anchor_u,
        reaction_x=reaction_x,
        reaction_weights=reaction_weights,
        true_reaction_y=true_reaction_y,
        pred_reaction_y=pred_reaction_y,
        field_names=np.asarray(names),
        final_beta=np.asarray(final_beta, dtype=float),
    )
    write_json(dirs["metrics"] / "metrics.json", metrics)
    write_json(dirs["metrics"] / "evaluation_metrics.json", {"relative_l2": split_metrics["evaluation"]})
    write_json(dirs["metrics"] / "validation_metrics.json", {"relative_l2": split_metrics["validation"]})
    write_json(dirs["metrics"] / "train_metrics.json", {"relative_l2": split_metrics["train"]})
    write_json(dirs["json"] / "training_summary.json", metrics)
    write_json(dirs["json"] / "runtime_status.json", {"status": "completed", "elapsed_seconds": elapsed})
    summary_lines = [
        f"工况: {args.case}",
        f"运行备注: {args.run_note}",
        f"训练步数: {args.iterations}",
        f"真实加载板沉降: {true_A}",
        f"边界幅值或 beta0: {metrics['final_A_or_beta0']}",
        f"K 相对 L2 误差: {split_metrics['evaluation'].get('K')}",
        f"mu 相对 L2 误差: {split_metrics['evaluation'].get('mu')}",
        f"E 相对 L2 误差: {split_metrics['evaluation'].get('E')}",
        f"nu 相对 L2 误差: {split_metrics['evaluation'].get('nu')}",
        f"总反力相对误差: {reaction_rel_error}",
    ]
    (dirs["txt"] / "关键指标摘要.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("RUN_COMPLETE", metrics, flush=True)


if __name__ == "__main__":
    main()
