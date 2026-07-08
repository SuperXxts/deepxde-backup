#!/usr/bin/env python3
"""Multi-load FEM gate experiments for boundary-uncertainty inversion."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("DDE_BACKEND", "pytorch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--display-every", type=int, default=1000)
    parser.add_argument("--dynamic-figure-every", type=int, default=None)
    parser.add_argument("--material-monitor-every", type=int, default=None)
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
    parser.add_argument("--boundary-modes", type=int, default=1)
    parser.add_argument("--projection-ridge", type=float, default=1.0e-8)
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--fixed-A", type=float, default=0.85)
    parser.add_argument("--init-A", type=float, default=0.85)
    parser.add_argument("--physics-weight", type=float, default=1.0)
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--obs-weight", type=float, default=1.0)
    parser.add_argument("--obs-material-weight", type=float, default=20.0)
    parser.add_argument("--obs-boundary-weight", type=float, default=20.0)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--reaction-weight", type=float, default=5.0)
    parser.add_argument("--material-smoothness-weight", type=float, default=0.0)
    parser.add_argument("--k-min-factor", type=float, default=0.35)
    parser.add_argument("--k-max-factor", type=float, default=1.80)
    parser.add_argument("--mu-min-factor", type=float, default=0.40)
    parser.add_argument("--mu-max-factor", type=float, default=1.70)
    parser.add_argument("--eval-checkpoint", choices=("best", "final"), default="best")
    parser.add_argument("--run-note", type=str, default="")
    parser.add_argument("--fem-data-dir", type=Path, default=None)
    parser.add_argument("--fem-nx", type=int, default=96)
    parser.add_argument("--fem-ny", type=int, default=48)
    parser.add_argument("--domain-width", type=float, default=6.0)
    parser.add_argument("--domain-depth", type=float, default=3.0)
    parser.add_argument("--plate-width", type=float, default=1.0)
    parser.add_argument("--settlement", type=float, default=-0.05)
    parser.add_argument("--horizontal-displacement", type=float, default=0.035)
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
        k_mu_to_e_nu,
        load_fem_result,
        make_anchor_points,
        make_observation_points,
        make_reaction_points,
        make_validation_points,
        plate_mode_values_np,
        save_fem_result,
        solve_fem,
        structured_grid_points,
    )
    from bupinn.io import ensure_run_dirs, save_table, write_json
    from bupinn.multiload_cases import (
        MULTILOAD_CASE_CHOICES,
        multiload_case_description,
        multiload_case_markdown_table,
        multiload_case_settings,
    )
    from bupinn.networks import MultiLoadBoundaryMaterialNet
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

    if args.case not in MULTILOAD_CASE_CHOICES:
        raise ValueError(f"Unsupported case {args.case!r}. Choices: {MULTILOAD_CASE_CHOICES}")

    settings = multiload_case_settings(args.case)
    dirs = ensure_run_dirs(args.run_dir)

    np.random.seed(args.seed)
    dde.config.set_random_seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    all_load_specs = [
        {
            "name": "central_vertical",
            "load_type": "vertical",
            "plate_center_fraction": 0.50,
            "settlement": args.settlement,
            "horizontal_displacement": 0.0,
        },
        {
            "name": "central_horizontal",
            "load_type": "horizontal",
            "plate_center_fraction": 0.50,
            "settlement": 0.0,
            "horizontal_displacement": args.horizontal_displacement,
        },
        {
            "name": "eccentric_vertical",
            "load_type": "vertical",
            "plate_center_fraction": 0.35,
            "settlement": args.settlement,
            "horizontal_displacement": 0.0,
        },
    ]
    load_specs = all_load_specs[: int(settings["num_loads"])]

    fem_dir = args.fem_data_dir or (args.run_dir.parent / "_fem_multiload_data")
    fem_results = []
    interpolators = []
    fem_configs = []
    for load_index, spec in enumerate(load_specs):
        cfg = FEMConfig(
            width=args.domain_width,
            depth=args.domain_depth,
            plate_width=args.plate_width,
            plate_center_fraction=float(spec["plate_center_fraction"]),
            nx=args.fem_nx,
            ny=args.fem_ny,
            settlement=float(spec["settlement"]),
            horizontal_displacement=float(spec["horizontal_displacement"]),
            load_type=str(spec["load_type"]),
            seed=args.seed,
            k_ref=args.k_ref,
            mu_ref=args.mu_ref,
        )
        load_dir = fem_dir / f"L{load_index}_{spec['name']}"
        fem_npz = load_dir / "fem_solution.npz"
        if fem_npz.exists():
            result = load_fem_result(fem_npz)
        else:
            result = solve_fem(cfg)
            save_fem_result(result, load_dir)
        fem_configs.append(cfg)
        fem_results.append(result)
        interpolators.append(FEMInterpolator(result))

    base_config = fem_configs[0]
    geom = dde.geometry.Rectangle([0.0, 0.0], [base_config.width, base_config.depth])
    num_loads = len(fem_configs)
    k0 = float(args.k_ref)
    mu0 = float(args.mu_ref)
    k_min = float(args.k_min_factor) * k0
    k_max = float(args.k_max_factor) * k0
    mu_min = float(args.mu_min_factor) * mu0
    mu_max = float(args.mu_max_factor) * mu0
    material_offset = 5 * num_loads
    diag_offset = material_offset + 2
    names = ["ux", "uy", "sxx", "syy", "sxy", "K", "mu", "E", "nu"]
    dynamic_figure_every = args.dynamic_figure_every if args.dynamic_figure_every is not None else args.display_every
    material_monitor_every = args.material_monitor_every if args.material_monitor_every is not None else args.display_every
    use_material_smoothness = float(args.material_smoothness_weight) > 0.0

    def normalize_inputs_torch(x):
        x_center = torch.as_tensor(
            [0.5 * base_config.width, 0.5 * base_config.depth],
            dtype=x.dtype,
            device=x.device,
        )
        x_scale = torch.as_tensor(
            [0.5 * base_config.width, 0.5 * base_config.depth],
            dtype=x.dtype,
            device=x.device,
        )
        return (x - x_center) / x_scale

    def lambda_from_k_mu(k, mu):
        return k - 2.0 * mu / 3.0

    def stress_from_k_mu(k, mu, exx, eyy, exy):
        lam = lambda_from_k_mu(k, mu)
        sxx = (2.0 * mu + lam) * exx + lam * eyy
        syy = lam * exx + (2.0 * mu + lam) * eyy
        sxy = 2.0 * mu * exy
        return sxx, syy, sxy

    def material_from_output(y):
        k = k_min + (k_max - k_min) * torch.sigmoid(y[:, material_offset : material_offset + 1])
        mu = mu_min + (mu_max - mu_min) * torch.sigmoid(y[:, material_offset + 1 : material_offset + 2])
        return k, mu

    def assemble_fields(raw: np.ndarray, load_index: int) -> np.ndarray:
        off = 5 * int(load_index)
        pred_k = k_min + (k_max - k_min) / (1.0 + np.exp(-raw[:, material_offset : material_offset + 1]))
        pred_mu = mu_min + (mu_max - mu_min) / (1.0 + np.exp(-raw[:, material_offset + 1 : material_offset + 2]))
        pred_e = 9.0 * pred_k * pred_mu / (3.0 * pred_k + pred_mu)
        pred_nu = (3.0 * pred_k - 2.0 * pred_mu) / (2.0 * (3.0 * pred_k + pred_mu))
        return np.column_stack([raw[:, off : off + 5], pred_k, pred_mu, pred_e, pred_nu])

    def relative_l2(y_true: np.ndarray, y_pred: np.ndarray, field_names: list[str]) -> dict[str, float]:
        out = {}
        for i, name in enumerate(field_names):
            denom = np.linalg.norm(y_true[:, i]) + 1.0e-12
            out[name] = float(np.linalg.norm(y_pred[:, i] - y_true[:, i]) / denom)
        return out

    def pde(x, y):
        k_pred, mu_pred = material_from_output(y)
        residuals = []
        for load_index in range(num_loads):
            off = 5 * load_index
            ux_x = dde.grad.jacobian(y, x, i=off + 0, j=0)
            uy_y = dde.grad.jacobian(y, x, i=off + 1, j=1)
            ux_y = dde.grad.jacobian(y, x, i=off + 0, j=1)
            uy_x = dde.grad.jacobian(y, x, i=off + 1, j=0)
            exx = ux_x
            eyy = uy_y
            exy = 0.5 * (ux_y + uy_x)
            sxx_c, syy_c, sxy_c = stress_from_k_mu(k_pred, mu_pred, exx, eyy, exy)
            sxx_x = dde.grad.jacobian(y, x, i=off + 2, j=0)
            syy_y = dde.grad.jacobian(y, x, i=off + 3, j=1)
            sxy_x = dde.grad.jacobian(y, x, i=off + 4, j=0)
            sxy_y = dde.grad.jacobian(y, x, i=off + 4, j=1)
            residuals.extend(
                [
                    sxx_x + sxy_y,
                    sxy_x + syy_y,
                    sxx_c - y[:, off + 2 : off + 3],
                    syy_c - y[:, off + 3 : off + 4],
                    sxy_c - y[:, off + 4 : off + 5],
                ]
            )
        if use_material_smoothness:
            k_x = dde.grad.jacobian(k_pred, x, i=0, j=0) / k0
            k_y = dde.grad.jacobian(k_pred, x, i=0, j=1) / k0
            mu_x = dde.grad.jacobian(mu_pred, x, i=0, j=0) / mu0
            mu_y = dde.grad.jacobian(mu_pred, x, i=0, j=1) / mu0
            residuals.extend([k_x, k_y, mu_x, mu_y])
        return residuals

    def boundary_bottom(x, on_boundary):
        return on_boundary and dde.utils.isclose(x[1], 0.0)

    def boundary_left(x, on_boundary):
        return on_boundary and dde.utils.isclose(x[0], 0.0)

    def boundary_right(x, on_boundary):
        return on_boundary and dde.utils.isclose(x[0], base_config.width)

    def boundary_plate_factory(cfg):
        def boundary_plate(x, on_boundary):
            return (
                on_boundary
                and dde.utils.isclose(x[1], cfg.depth)
                and x[0] >= cfg.plate_left - 1.0e-10
                and x[0] <= cfg.plate_right + 1.0e-10
            )

        return boundary_plate

    def boundary_top_free_factory(cfg):
        def boundary_top_free(x, on_boundary):
            return (
                on_boundary
                and dde.utils.isclose(x[1], cfg.depth)
                and (x[0] < cfg.plate_left - 1.0e-10 or x[0] > cfg.plate_right + 1.0e-10)
            )

        return boundary_top_free

    external_vars = []
    amplitude_vars: list[list[object | None]] = []
    init_beta = []
    true_plate_displacements = []
    wrong_plate_displacements = []
    for cfg in fem_configs:
        true_ux, true_uy = cfg.plate_displacement
        true_vec = [float(true_ux), float(true_uy)]
        wrong_vec = [float(args.fixed_A) * true_vec[0], float(args.fixed_A) * true_vec[1]]
        true_plate_displacements.append(true_vec)
        wrong_plate_displacements.append(wrong_vec)
        vars_for_load: list[object | None] = []
        beta_row = [0.0] * (2 * int(args.boundary_modes))
        if int(args.boundary_modes) > 0:
            beta_row[0] = float(args.init_A) * true_vec[0]
            beta_row[int(args.boundary_modes)] = float(args.init_A) * true_vec[1]
        for component in range(2):
            if settings["learnable_A"] and abs(true_vec[component]) > 1.0e-12:
                var = dde.Variable(float(args.init_A) * true_vec[component])
                external_vars.append(var)
                vars_for_load.append(var)
            else:
                vars_for_load.append(None)
        amplitude_vars.append(vars_for_load)
        init_beta.append(beta_row)

    net = None

    def top_target(inputs, load_index: int, component: int):
        top_mode = str(settings["top"])
        if top_mode == "true":
            return true_plate_displacements[load_index][component] * torch.ones_like(inputs[:, 0:1])
        if top_mode == "wrong":
            return wrong_plate_displacements[load_index][component] * torch.ones_like(inputs[:, 0:1])
        if top_mode == "learnable_A":
            var = amplitude_vars[load_index][component]
            if var is None:
                return torch.zeros_like(inputs[:, 0:1])
            return var * torch.ones_like(inputs[:, 0:1])
        if top_mode == "network_top":
            return net.top_boundary_value(inputs, load_index, component)
        raise ValueError(f"Unknown top mode: {top_mode}")

    bcs = []
    loss_terms = []
    for load_index, cfg in enumerate(fem_configs):
        off = 5 * load_index
        plate_bc = boundary_plate_factory(cfg)
        top_free_bc = boundary_top_free_factory(cfg)

        def side_sxy_bc(inputs, outputs, _X, off=off):
            return outputs[:, off + 4 : off + 5]

        def top_syy_bc(inputs, outputs, _X, off=off):
            return outputs[:, off + 3 : off + 4]

        def top_sxy_bc(inputs, outputs, _X, off=off):
            return outputs[:, off + 4 : off + 5]

        def plate_ux_bc(inputs, outputs, _X, load_index=load_index, off=off):
            return outputs[:, off : off + 1] - top_target(inputs, load_index, 0)

        def plate_uy_bc(inputs, outputs, _X, load_index=load_index, off=off):
            return outputs[:, off + 1 : off + 2] - top_target(inputs, load_index, 1)

        bcs.extend(
            [
                dde.icbc.DirichletBC(geom, lambda x: 0.0, boundary_bottom, component=off + 0),
                dde.icbc.DirichletBC(geom, lambda x: 0.0, boundary_bottom, component=off + 1),
                dde.icbc.DirichletBC(geom, lambda x: 0.0, boundary_left, component=off + 0),
                dde.icbc.DirichletBC(geom, lambda x: 0.0, boundary_right, component=off + 0),
                dde.icbc.OperatorBC(geom, side_sxy_bc, boundary_left),
                dde.icbc.OperatorBC(geom, side_sxy_bc, boundary_right),
                dde.icbc.OperatorBC(geom, top_syy_bc, top_free_bc),
                dde.icbc.OperatorBC(geom, top_sxy_bc, top_free_bc),
                dde.icbc.OperatorBC(geom, plate_ux_bc, plate_bc),
                dde.icbc.OperatorBC(geom, plate_uy_bc, plate_bc),
            ]
        )
        load_prefix = f"L{load_index}"
        loss_terms.extend(
            [
                f"{load_prefix}_bottom_ux",
                f"{load_prefix}_bottom_uy",
                f"{load_prefix}_left_ux",
                f"{load_prefix}_right_ux",
                f"{load_prefix}_left_sxy",
                f"{load_prefix}_right_sxy",
                f"{load_prefix}_top_free_syy",
                f"{load_prefix}_top_free_sxy",
                f"{load_prefix}_plate_ux",
                f"{load_prefix}_plate_uy",
            ]
        )

    pde_loss_terms = []
    for load_index in range(num_loads):
        prefix = f"L{load_index}"
        pde_loss_terms.extend(
            [
                f"{prefix}_momentum_x",
                f"{prefix}_momentum_y",
                f"{prefix}_constitutive_sxx",
                f"{prefix}_constitutive_syy",
                f"{prefix}_constitutive_sxy",
            ]
        )
    if use_material_smoothness:
        pde_loss_terms.extend(["smooth_K_x", "smooth_K_y", "smooth_mu_x", "smooth_mu_y"])
    loss_terms = pde_loss_terms + loss_terms

    obs_x = make_observation_points(base_config, args.obs_count, args.seed)
    validation_x_for_monitor = make_validation_points(base_config, args.val_count, args.seed)

    def observation_design_uv_np(x: np.ndarray, num_modes: int, cfg: FEMConfig) -> np.ndarray:
        if num_modes <= 0:
            return np.empty((2 * len(x), 0), dtype=np.float32)
        lift = ((x[:, 1:2] - 0.0) / max(cfg.depth, 1.0e-12)).astype(np.float32)
        top_modes = plate_mode_values_np(x, num_modes, cfg)
        ux_modes = lift * top_modes
        uy_modes = lift * top_modes
        zeros = np.zeros_like(ux_modes)
        upper = np.hstack([ux_modes, zeros])
        lower = np.hstack([zeros, uy_modes])
        return np.vstack([upper, lower]).astype(np.float32)

    obs_duplicate_occurrence = 0
    obs_payload = []
    for load_index, (cfg, interp) in enumerate(zip(fem_configs, interpolators)):
        obs_u = interp.evaluate(obs_x)[:, 0:2]
        obs_u = add_displacement_noise(obs_u, args.noise_level, args.seed + 101 * load_index)
        obs_payload.append((obs_x, obs_u))
        off = 5 * load_index
        if settings["decoupled_obs"]:
            design = observation_design_uv_np(obs_x, int(args.boundary_modes), cfg)
            core_start = diag_offset + 4 * load_index
            boundary_start = core_start + 2
            projector = ObservationProjector(
                obs_x,
                obs_u,
                design,
                args.projection_ridge,
                core_start=core_start,
                boundary_start=boundary_start,
                cache_components=4,
            )
            zeros = np.zeros((len(obs_x), 1), dtype=np.float32)

            def obs_material_ux(inputs, outputs, X, projector=projector, occurrence=obs_duplicate_occurrence):
                return projector.projected_component(inputs, outputs, X, 0, "material", "material", occurrence)

            def obs_material_uy(inputs, outputs, X, projector=projector, occurrence=obs_duplicate_occurrence + 1):
                return projector.projected_component(inputs, outputs, X, 1, "material", "material", occurrence)

            def obs_boundary_ux(inputs, outputs, X, projector=projector, occurrence=obs_duplicate_occurrence + 2):
                return projector.projected_component(inputs, outputs, X, 0, "boundary", "boundary", occurrence)

            def obs_boundary_uy(inputs, outputs, X, projector=projector, occurrence=obs_duplicate_occurrence + 3):
                return projector.projected_component(inputs, outputs, X, 1, "boundary", "boundary", occurrence)

            bcs.extend(
                [
                    dde.icbc.PointSetOperatorBC(obs_x, zeros, obs_material_ux),
                    dde.icbc.PointSetOperatorBC(obs_x, zeros, obs_material_uy),
                    dde.icbc.PointSetOperatorBC(obs_x, zeros, obs_boundary_ux),
                    dde.icbc.PointSetOperatorBC(obs_x, zeros, obs_boundary_uy),
                ]
            )
            loss_terms.extend(
                [
                    f"L{load_index}_obs_material_ux",
                    f"L{load_index}_obs_material_uy",
                    f"L{load_index}_obs_boundary_ux",
                    f"L{load_index}_obs_boundary_uy",
                ]
            )
            obs_duplicate_occurrence += 4
        else:
            bcs.append(dde.icbc.PointSetBC(obs_x, obs_u, component=[off, off + 1]))
            loss_terms.append(f"L{load_index}_obs_u")
            obs_duplicate_occurrence += 1

    anchor_payload = []
    if settings["anchor"]:
        for load_index, (cfg, interp) in enumerate(zip(fem_configs, interpolators)):
            anchor_x = make_anchor_points(cfg, args.anchor_count)
            anchor_u = interp.evaluate(anchor_x)[:, 0:2] if len(anchor_x) else np.empty((0, 2), dtype=np.float32)
            anchor_payload.append((anchor_x, anchor_u))
            if len(anchor_x):
                off = 5 * load_index
                bcs.append(dde.icbc.PointSetBC(anchor_x, anchor_u, component=[off, off + 1]))
                loss_terms.append(f"L{load_index}_anchor_u")
    else:
        anchor_payload = [(np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)) for _ in fem_configs]

    reaction_payload = []
    reaction_occurrences_by_key = {}
    if settings["reaction"]:
        for load_index, (cfg, result) in enumerate(zip(fem_configs, fem_results)):
            reaction_x, reaction_weights = make_reaction_points(cfg, args.reaction_points)
            component = cfg.reaction_component
            true_reaction = result.total_reaction_x if component == 0 else result.total_reaction_y
            scale = abs(float(true_reaction)) + 1.0e-12
            key = (round(cfg.plate_left, 8), round(cfg.plate_right, 8), component)
            occurrence = reaction_occurrences_by_key.get(key, 0)
            reaction_occurrences_by_key[key] = occurrence + 1
            stress_index = 5 * load_index + (4 if component == 0 else 3)
            operator = ReactionIntegral(
                reaction_weights,
                points=reaction_x,
                stress_index=stress_index,
                scale=scale,
                occurrence=occurrence,
                fill_all=True,
            )
            target = np.full((len(reaction_x), 1), float(true_reaction) / scale, dtype=np.float32)
            bcs.append(dde.icbc.PointSetOperatorBC(reaction_x, target, operator))
            loss_terms.append(f"L{load_index}_global_reaction")
            reaction_payload.append((reaction_x, reaction_weights, true_reaction, component, scale))
    else:
        for cfg, result in zip(fem_configs, fem_results):
            reaction_x, reaction_weights = make_reaction_points(cfg, args.reaction_points)
            component = cfg.reaction_component
            true_reaction = result.total_reaction_x if component == 0 else result.total_reaction_y
            reaction_payload.append((reaction_x, reaction_weights, true_reaction, component, abs(float(true_reaction)) + 1.0e-12))

    data = dde.data.PDE(
        geom,
        pde,
        bcs,
        num_domain=args.num_domain,
        num_boundary=args.num_boundary,
        train_distribution="Hammersley",
        num_test=None,
    )

    if settings["boundary_layer"]:
        net = MultiLoadBoundaryMaterialNet(
            dde,
            num_cases=num_loads,
            state_width=args.width,
            state_depth=args.depth,
            material_width=args.material_width,
            material_depth=args.material_depth,
            num_boundary_modes=int(args.boundary_modes),
            init_beta=init_beta,
            trainable_boundary=True,
            use_boundary_layer=True,
            domain_x_min=0.0,
            domain_x_max=base_config.width,
            domain_y_bottom=0.0,
            domain_y_top=base_config.depth,
            plate_centers=[cfg.plate_center for cfg in fem_configs],
            plate_widths=[cfg.plate_width for cfg in fem_configs],
            include_diagnostics=bool(settings["decoupled_obs"]),
        )
        net.apply_feature_transform(normalize_inputs_torch)
        network_config = {
            "type": "MultiLoadBoundaryMaterialNet",
            "num_loads": num_loads,
            "shared_material_branch": True,
            "load_specific_state_branches": True,
            "boundary_layer": True,
            "diagnostic_outputs": bool(settings["decoupled_obs"]),
        }
    else:
        output_dim = 5 * num_loads + 2
        layer_sizes = [2] + [[int(args.width)] * output_dim for _ in range(int(args.depth))] + [output_dim]
        net = dde.nn.PFNN(layer_sizes, "tanh", "Glorot normal")
        net.apply_feature_transform(normalize_inputs_torch)
        network_config = {
            "type": "DeepXDE PFNN multi-load shared-material baseline",
            "layer_sizes": layer_sizes,
            "num_loads": num_loads,
            "shared_material_outputs": ["zK", "zmu"],
        }

    model = dde.Model(data, net)

    loss_weights = []
    for _ in range(num_loads):
        loss_weights.extend([float(args.physics_weight)] * 5)
    if use_material_smoothness:
        loss_weights.extend([float(args.material_smoothness_weight)] * 4)
    for _ in range(num_loads):
        loss_weights.extend([float(args.boundary_weight)] * 10)
    for _ in range(num_loads):
        if settings["decoupled_obs"]:
            loss_weights.extend(
                [
                    float(args.obs_material_weight),
                    float(args.obs_material_weight),
                    float(args.obs_boundary_weight),
                    float(args.obs_boundary_weight),
                ]
            )
        else:
            loss_weights.append(float(args.obs_weight))
    if settings["anchor"]:
        for _ in range(num_loads):
            loss_weights.append(float(args.anchor_weight))
    if settings["reaction"]:
        for _ in range(num_loads):
            loss_weights.append(float(args.reaction_weight))
    if len(loss_weights) != len(loss_terms):
        raise RuntimeError(f"loss_weights length {len(loss_weights)} != loss_terms length {len(loss_terms)}")

    model.compile("adam", lr=args.lr, loss_weights=loss_weights, external_trainable_variables=external_vars)

    def beta_row(step: int) -> list[float]:
        row = [float(step)]
        if settings["boundary_layer"]:
            row.extend([float(v) for v in net.beta.detach().cpu().numpy().ravel()])
        elif settings["learnable_A"]:
            for load_vars, true_vec in zip(amplitude_vars, true_plate_displacements):
                for component in range(2):
                    var = load_vars[component]
                    row.append(float(var.detach().cpu().item()) if var is not None else float(true_vec[component]))
        elif settings["top"] == "wrong":
            for vec in wrong_plate_displacements:
                row.extend(vec)
        else:
            for vec in true_plate_displacements:
                row.extend(vec)
        return row

    class ParameterLogger(dde.callbacks.Callback):
        def __init__(self, path: Path, period: int):
            super().__init__()
            self.path = path
            self.period = max(1, int(period))
            self.rows = []

        def on_train_begin(self):
            self.rows.append(beta_row(0))

        def on_epoch_end(self):
            step = int(self.model.train_state.iteration)
            if step % self.period == 0:
                row = beta_row(step)
                self.rows.append(row)
                print("STEP", step, "BETA", " ".join(f"{v:.8f}" for v in row[1:]), flush=True)

        def on_train_end(self):
            max_len = max(len(row) for row in self.rows)
            arr = np.asarray([row + [np.nan] * (max_len - len(row)) for row in self.rows], dtype=float)
            header = "step " + " ".join(f"beta{i}" for i in range(max_len - 1))
            np.savetxt(self.path, arr, header=header)

    validation_truth = interpolators[0].evaluate(validation_x_for_monitor)

    class DynamicFigureLogger(dde.callbacks.Callback):
        def __init__(self, loss_period: int, figure_period: int, material_period: int):
            super().__init__()
            self.loss_period = max(1, int(loss_period))
            self.figure_period = int(figure_period)
            self.material_period = int(material_period)
            self.steps = []
            self.loss_train = []
            self.loss_test = []
            self.material_rows = []

        def _record_loss(self, step: int):
            loss_train = np.asarray(self.model.train_state.loss_train, dtype=float)
            loss_test = np.asarray(self.model.train_state.loss_test, dtype=float)
            self.steps.append(step)
            self.loss_train.append(loss_train)
            self.loss_test.append(loss_test)
            if self.figure_period > 0 and step % self.figure_period == 0:
                plot_loss_arrays(
                    np.asarray(self.steps),
                    np.asarray(self.loss_train),
                    np.asarray(self.loss_test),
                    loss_terms,
                    dirs["png"] / "\u635f\u5931\u5386\u53f2\u56fe.png",
                    dirs["png"] / "\u635f\u5931\u5206\u91cf\u56fe.png",
                    dirs["npz"] / "\u52a8\u6001\u635f\u5931\u5386\u53f2.npz",
                )

        def _record_material(self, step: int):
            with torch.no_grad():
                raw = self.model.predict(validation_x_for_monitor)
            pred = assemble_fields(raw, 0)
            rel = relative_l2(validation_truth, pred, names)
            row = [float(step), rel["K"], rel["mu"], rel["E"], rel["nu"]]
            self.material_rows.append(row)
            arr = np.asarray(self.material_rows, dtype=float)
            np.savetxt(
                dirs["dat"] / "\u6750\u6599\u53c2\u6570\u6f14\u5316.dat",
                arr,
                header="step K_rel_l2 mu_rel_l2 E_rel_l2 nu_rel_l2",
            )
            if self.figure_period > 0 and step % self.figure_period == 0:
                plot_material_error_history(arr, dirs["png"] / "\u6750\u6599\u53c2\u6570\u6f14\u5316\u56fe.png")

        def on_epoch_end(self):
            step = int(self.model.train_state.iteration)
            if step % self.loss_period == 0 and self.model.train_state.loss_train is not None:
                self._record_loss(step)
            if self.material_period > 0 and step % self.material_period == 0:
                self._record_material(step)

    callbacks = [
        ParameterLogger(dirs["dat"] / "boundary_parameter_history.dat", args.display_every),
        DynamicFigureLogger(args.display_every, dynamic_figure_every, material_monitor_every),
        dde.callbacks.ModelCheckpoint(
            str(dirs["model"] / "best_model"),
            verbose=0,
            save_better_only=True,
            period=max(1, args.checkpoint_every),
            monitor="train loss",
        ),
    ]

    config_payload = vars(args).copy()
    config_payload.update(
        {
            "case_settings": settings,
            "load_specs": load_specs,
            "fem_configs": [cfg.__dict__ for cfg in fem_configs],
            "true_plate_displacements": true_plate_displacements,
            "wrong_plate_displacements": wrong_plate_displacements,
            "loss_terms": loss_terms,
            "loss_weights": loss_weights,
            "network": network_config,
            "field_names": names,
            "material_parameterization": "bounded dual sigmoid shared by all load cases",
            "material_bounds": {"K_min": k_min, "K_max": k_max, "mu_min": mu_min, "mu_max": mu_max},
        }
    )
    write_json(dirs["json"] / "config.json", config_payload)
    write_json(dirs["json"] / "run_config.json", config_payload)
    (args.run_dir / "实验说明.txt").write_text(multiload_case_description(args.case), encoding="utf-8")
    (args.run_dir / "C组实验矩阵.md").write_text(multiload_case_markdown_table(), encoding="utf-8")
    if str(args.run_note).strip():
        (args.run_dir / "变体说明.txt").write_text(str(args.run_note).strip() + "\n", encoding="utf-8")

    combined_anchors = np.vstack([x for x, _ in anchor_payload if len(x)]) if any(len(x) for x, _ in anchor_payload) else np.empty((0, 2), dtype=np.float32)
    plot_sampling(obs_x, combined_anchors, dirs["png"] / "采样点分布图.png")
    for load_index, ((ox, ou), (ax, au), reaction_info) in enumerate(zip(obs_payload, anchor_payload, reaction_payload)):
        save_table(dirs["dat"] / f"L{load_index}_observation_points.dat", np.hstack([ox, ou]), "x y ux uy")
        if len(ax):
            save_table(dirs["dat"] / f"L{load_index}_anchor_points.dat", np.hstack([ax, au]), "x y ux uy")
        rx, rw, true_reaction, component, scale = reaction_info
        save_table(dirs["dat"] / f"L{load_index}_reaction_points.dat", np.hstack([rx, rw]), "x y integration_weight")

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
    param_data = np.loadtxt(dirs["dat"] / "boundary_parameter_history.dat", comments="#")
    if param_data.ndim == 1:
        param_data = param_data.reshape(1, -1)
    np.savetxt(dirs["dat"] / "amplitude_history.dat", param_data[:, :2], header="step first_boundary_parameter")
    plot_amplitude_history(dirs["dat"] / "amplitude_history.dat", dirs["png"] / "边界参数演化图.png", true_plate_displacements[0][1])

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

    x_eval = structured_grid_points(args.test_nx, args.test_ny, base_config.width, base_config.depth)
    split_defs = {
        "evaluation": x_eval,
        "validation": validation_x_for_monitor,
        "train": obs_x,
    }
    split_metrics = {}
    eval_payload = {}
    for split_name, x_split in split_defs.items():
        raw = model.predict(x_split)
        split_metrics[split_name] = {}
        eval_payload[split_name] = {"x": x_split, "raw": raw, "loads": []}
        for load_index, interp in enumerate(interpolators):
            pred = assemble_fields(raw, load_index)
            truth = interp.evaluate(x_split)
            rel = relative_l2(truth, pred, names)
            split_metrics[split_name][f"L{load_index}"] = rel
            eval_payload[split_name]["loads"].append({"truth": truth, "pred": pred})
            np.savez(
                dirs["npz"] / f"{split_name}_L{load_index}_predictions_data.npz",
                x=x_split,
                y_true=truth,
                y_pred=pred,
                y_pred_raw=raw,
                field_names=np.asarray(names),
            )
            save_table(
                dirs["txt"] / f"{split_name}_L{load_index}_predictions_data.txt",
                np.column_stack([x_split, truth, pred, pred - truth]),
                "x y true_fields pred_fields error_fields",
            )
            if split_name == "evaluation":
                for i, name in enumerate(names):
                    np.savez(
                        dirs["npz"] / f"{split_name}_L{load_index}_predictions_{name}.npz",
                        x=x_split,
                        y_true=truth[:, i],
                        y_pred=pred[:, i],
                        error=pred[:, i] - truth[:, i],
                    )
        if split_name == "evaluation":
            truth0 = eval_payload[split_name]["loads"][0]["truth"]
            pred0 = eval_payload[split_name]["loads"][0]["pred"]
            for i, name in enumerate(names):
                plot_field_triplet(x_split, truth0[:, i], pred0[:, i], name, dirs["png"] / f"评估_{name}_真值预测误差图.png")

    truth0 = eval_payload["evaluation"]["loads"][0]["truth"]
    pred0 = eval_payload["evaluation"]["loads"][0]["pred"]
    plot_k_mu_comparison(x_eval, truth0, pred0, dirs["png"] / "评估_K与μ真值预测对比图.png")
    plot_displacement_vector(x_eval, truth0, pred0, dirs["png"] / "位移矢量场图_评估集_L0.png")
    for y_value in (0.75, 1.50, 2.25):
        plot_line_slice(
            x_eval,
            {"K": (truth0[:, 5], pred0[:, 5]), "mu": (truth0[:, 6], pred0[:, 6]), "nu": (truth0[:, 8], pred0[:, 8])},
            dirs["png"] / f"材料参数切片对比图_y{str(y_value).replace('.', 'p')}.png",
            y_value=y_value,
        )

    reaction_metrics = []
    for load_index, (rx, rw, true_reaction, component, scale) in enumerate(reaction_payload):
        raw = model.predict(rx)
        pred = assemble_fields(raw, load_index)
        density = pred[:, 4:5] if component == 0 else pred[:, 3:4]
        pred_reaction = float(np.sum(density * rw))
        rel_err = abs(pred_reaction - float(true_reaction)) / (abs(float(true_reaction)) + 1.0e-12)
        reaction_metrics.append(
            {
                "load_index": load_index,
                "component": "x" if component == 0 else "y",
                "true_reaction": float(true_reaction),
                "pred_reaction": pred_reaction,
                "relative_error": float(rel_err),
            }
        )
        true_density = np.full((len(rx),), float(true_reaction) / fem_configs[load_index].plate_width, dtype=np.float32)
        plot_boundary_curve(
            rx,
            true_density,
            density[:, 0],
            f"Load {load_index} reaction-density proxy",
            dirs["png"] / f"L{load_index}_反力密度曲线图.png",
        )

    material_rel = split_metrics["evaluation"]["L0"]
    mean_load_metrics = {
        name: float(np.mean([split_metrics["evaluation"][f"L{i}"][name] for i in range(num_loads)]))
        for name in ["ux", "uy", "sxx", "syy", "sxy"]
    }
    metrics = {
        "case": args.case,
        "iterations": args.iterations,
        "elapsed_seconds": elapsed,
        "num_loads": num_loads,
        "relative_l2": material_rel,
        "mean_load_relative_l2": mean_load_metrics,
        "split_relative_l2": split_metrics,
        "reaction_metrics": reaction_metrics,
        "final_boundary_parameters": beta_row(args.iterations)[1:],
        "best_step": int(train_state.best_step),
        "eval_checkpoint": args.eval_checkpoint,
        "eval_checkpoint_path": eval_checkpoint_path,
        "final_model_path": final_model_path,
        "final_train_loss_sum": float(np.sum(train_state.loss_train)),
        "final_test_loss_sum": float(np.sum(train_state.loss_test)),
        "loss_history_interval": int(args.display_every),
        "num_trainable_parameters": int(net.num_trainable_parameters())
        if hasattr(net, "num_trainable_parameters")
        else int(sum(p.numel() for p in net.parameters() if p.requires_grad)),
    }
    np.savez(
        dirs["npz"] / "eval_fields.npz",
        x_test=x_eval,
        y_true=np.stack([item["truth"] for item in eval_payload["evaluation"]["loads"]], axis=0),
        y_pred=np.stack([item["pred"] for item in eval_payload["evaluation"]["loads"]], axis=0),
        obs_x=obs_x,
        reaction_metrics=np.asarray(reaction_metrics, dtype=object),
        field_names=np.asarray(names),
        final_boundary_parameters=np.asarray(metrics["final_boundary_parameters"], dtype=float),
    )
    write_json(dirs["metrics"] / "metrics.json", metrics)
    write_json(dirs["metrics"] / "evaluation_metrics.json", {"relative_l2": split_metrics["evaluation"]})
    write_json(dirs["metrics"] / "validation_metrics.json", {"relative_l2": split_metrics["validation"]})
    write_json(dirs["metrics"] / "train_metrics.json", {"relative_l2": split_metrics["train"]})
    write_json(dirs["json"] / "training_summary.json", metrics)
    write_json(dirs["json"] / "runtime_status.json", {"status": "completed", "elapsed_seconds": elapsed})
    summary_lines = [
        f"Case: {args.case}",
        f"Loads: {num_loads}",
        f"Iterations: {args.iterations}",
        f"K relative L2 error: {material_rel.get('K')}",
        f"mu relative L2 error: {material_rel.get('mu')}",
        f"E relative L2 error: {material_rel.get('E')}",
        f"nu relative L2 error: {material_rel.get('nu')}",
        f"Mean ux error over loads: {mean_load_metrics.get('ux')}",
        f"Mean uy error over loads: {mean_load_metrics.get('uy')}",
    ]
    (dirs["txt"] / "关键指标摘要.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("RUN_COMPLETE", metrics, flush=True)


if __name__ == "__main__":
    main()
