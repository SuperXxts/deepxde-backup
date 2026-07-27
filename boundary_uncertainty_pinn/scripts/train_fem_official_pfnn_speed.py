#!/usr/bin/env python3
"""Official-style DeepXDE PFNN speed diagnostic on the FEM benchmark.

This script is intentionally narrower than the formal inversion trainers. It
uses the same FEM data, geometry, collocation counts, boundary conditions and
observation points, but removes the material branch, boundary branch, material
smoothness and decoupled observation projection. The goal is to measure whethe
the FEM setup itself is slow, or whether the overhead mainly comes from the
formal method modules.
"""

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
    parser.add_argument("--load-count", type=int, choices=(1, 3), default=1)
    parser.add_argument("--boundary-scale", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--display-every", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=3000)
    parser.add_argument("--num-domain", type=int, default=2000)
    parser.add_argument("--num-boundary", type=int, default=600)
    parser.add_argument("--obs-count", type=int, default=120)
    parser.add_argument("--val-count", type=int, default=500)
    parser.add_argument("--test-nx", type=int, default=121)
    parser.add_argument("--test-ny", type=int, default=61)
    parser.add_argument("--width", type=int, default=56)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1.0e-3)
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

    from bupinn.fem2d import (
        FEMConfig,
        FEMInterpolator,
        add_displacement_noise,
        load_fem_result,
        make_observation_points,
        make_validation_points,
        save_fem_result,
        solve_fem,
        structured_grid_points,
    )
    from bupinn.io import ensure_run_dirs, save_table, write_json
    from bupinn.plotting import plot_loss_arrays, plot_loss_history, plot_sampling

    dirs = ensure_run_dirs(args.run_dir)
    np.random.seed(args.seed)
    dde.config.set_random_seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    load_specs = [
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
    ][: args.load_count]

    fem_dir = args.fem_data_dir or (args.run_dir.parent / "_fem_multiload_data")
    fem_results = []
    fem_configs = []
    interpolators = []
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
    output_dim = 5 * num_loads
    lam_ref = float(args.k_ref) - 2.0 * float(args.mu_ref) / 3.0
    mu_ref = float(args.mu_ref)

    def normalize_inputs_torch(x):
        center = torch.as_tensor(
            [0.5 * base_config.width, 0.5 * base_config.depth],
            dtype=x.dtype,
            device=x.device,
        )
        scale = torch.as_tensor(
            [0.5 * base_config.width, 0.5 * base_config.depth],
            dtype=x.dtype,
            device=x.device,
        )
        return (x - center) / scale

    def pde(x, y):
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
            sxx_c = (2.0 * mu_ref + lam_ref) * exx + lam_ref * eyy
            syy_c = lam_ref * exx + (2.0 * mu_ref + lam_ref) * eyy
            sxy_c = 2.0 * mu_ref * exy
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

    bcs = []
    loss_terms = []
    true_plate_displacements = []
    imposed_plate_displacements = []
    for load_index, cfg in enumerate(fem_configs):
        off = 5 * load_index
        true_ux, true_uy = cfg.plate_displacement
        true_vec = [float(true_ux), float(true_uy)]
        imposed_vec = [float(args.boundary_scale) * true_vec[0], float(args.boundary_scale) * true_vec[1]]
        true_plate_displacements.append(true_vec)
        imposed_plate_displacements.append(imposed_vec)
        plate_bc = boundary_plate_factory(cfg)
        top_free_bc = boundary_top_free_factory(cfg)

        def side_sxy_bc(inputs, outputs, _X, off=off):
            return outputs[:, off + 4 : off + 5]

        def top_syy_bc(inputs, outputs, _X, off=off):
            return outputs[:, off + 3 : off + 4]

        def top_sxy_bc(inputs, outputs, _X, off=off):
            return outputs[:, off + 4 : off + 5]

        def plate_ux_bc(inputs, outputs, _X, target=imposed_vec[0], off=off):
            return outputs[:, off : off + 1] - float(target)

        def plate_uy_bc(inputs, outputs, _X, target=imposed_vec[1], off=off):
            return outputs[:, off + 1 : off + 2] - float(target)

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
        prefix = f"L{load_index}"
        loss_terms.extend(
            [
                f"{prefix}_bottom_ux",
                f"{prefix}_bottom_uy",
                f"{prefix}_left_ux",
                f"{prefix}_right_ux",
                f"{prefix}_left_sxy",
                f"{prefix}_right_sxy",
                f"{prefix}_top_free_syy",
                f"{prefix}_top_free_sxy",
                f"{prefix}_plate_ux",
                f"{prefix}_plate_uy",
            ]
        )

    pde_terms = []
    for load_index in range(num_loads):
        prefix = f"L{load_index}"
        pde_terms.extend(
            [
                f"{prefix}_momentum_x",
                f"{prefix}_momentum_y",
                f"{prefix}_constitutive_sxx",
                f"{prefix}_constitutive_syy",
                f"{prefix}_constitutive_sxy",
            ]
        )
    loss_terms = pde_terms + loss_terms

    obs_x = make_observation_points(base_config, args.obs_count, args.seed)
    obs_payload = []
    for load_index, interp in enumerate(interpolators):
        obs_u = interp.evaluate(obs_x)[:, 0:2]
        obs_u = add_displacement_noise(obs_u, args.noise_level, args.seed + 101 * load_index)
        obs_payload.append((obs_x, obs_u))
        off = 5 * load_index
        bcs.append(dde.icbc.PointSetBC(obs_x, obs_u, component=[off, off + 1]))
        loss_terms.append(f"L{load_index}_obs_u")

    data = dde.data.PDE(
        geom,
        pde,
        bcs,
        num_domain=args.num_domain,
        num_boundary=args.num_boundary,
        train_distribution="Hammersley",
        num_test=None,
    )

    layer_sizes = [2] + [[int(args.width)] * output_dim for _ in range(int(args.depth))] + [output_dim]
    net = dde.nn.PFNN(layer_sizes, "tanh", "Glorot normal")
    net.apply_feature_transform(normalize_inputs_torch)
    model = dde.Model(data, net)
    model.compile("adam", lr=args.lr)

    loss_steps: list[int] = []
    loss_train_rows: list[np.ndarray] = []
    loss_test_rows: list[np.ndarray] = []
    speed_rows: list[list[float]] = []

    class SpeedLogger(dde.callbacks.Callback):
        def __init__(self, period: int):
            super().__init__()
            self.period = max(1, int(period))
            self.start = time.time()
            self.last_time = self.start
            self.last_step = 0

        def on_epoch_end(self):
            step = int(self.model.train_state.iteration)
            if step % self.period != 0:
                return
            now = time.time()
            delta_step = max(step - self.last_step, 1)
            delta_time = now - self.last_time
            total_time = now - self.start
            sec_per_step = delta_time / float(delta_step)
            speed_rows.append([float(step), total_time, sec_per_step])
            self.last_time = now
            self.last_step = step
            if self.model.train_state.loss_train is not None:
                loss_steps.append(step)
                loss_train_rows.append(np.asarray(self.model.train_state.loss_train, dtype=float))
                loss_test_rows.append(np.asarray(self.model.train_state.loss_test, dtype=float))
                plot_loss_arrays(
                    np.asarray(loss_steps, dtype=float),
                    np.asarray(loss_train_rows, dtype=float),
                    np.asarray(loss_test_rows, dtype=float),
                    loss_terms,
                    dirs["png"] / "\u635f\u5931\u5386\u53f2\u56fe.png",
                    dirs["png"] / "\u635f\u5931\u5206\u91cf\u56fe.png",
                    dirs["npz"] / "\u52a8\u6001\u635f\u5931\u5386\u53f2.npz",
                )
            print(
                f"STEP {step} SPEED elapsed={total_time:.2f}s interval_sec_per_step={sec_per_step:.6f}",
                flush=True,
            )
            write_json(
                dirs["json"] / "runtime_status.json",
                {
                    "status": "running",
                    "step": step,
                    "elapsed_seconds": total_time,
                    "interval_sec_per_step": sec_per_step,
                    "load_count": num_loads,
                    "boundary_scale": float(args.boundary_scale),
                },
            )

    config_payload = vars(args).copy()
    config_payload.update(
        {
            "diagnostic": "official-style DeepXDE PFNN speed baseline",
            "load_specs": load_specs,
            "fem_configs": [cfg.__dict__ for cfg in fem_configs],
            "true_plate_displacements": true_plate_displacements,
            "imposed_plate_displacements": imposed_plate_displacements,
            "loss_terms": loss_terms,
            "network": {
                "type": "DeepXDE PFNN official-style state-only baseline",
                "layer_sizes": layer_sizes,
                "outputs_per_load": ["ux", "uy", "sxx", "syy", "sxy"],
                "fixed_lame_lambda": lam_ref,
                "fixed_mu": mu_ref,
                "removed_modules": [
                    "material_branch",
                    "boundary_branch",
                    "decoupled_observation_projection",
                    "material_smoothness",
                    "reaction_integral",
                    "anchor_points",
                ],
            },
        }
    )
    write_json(dirs["json"] / "config.json", config_payload)
    write_json(dirs["json"] / "run_config.json", config_payload)
    if str(args.run_note).strip():
        (args.run_dir / "实验说明.txt").write_text(str(args.run_note).strip() + "\n", encoding="utf-8")
    plot_sampling(obs_x, np.empty((0, 2), dtype=np.float32), dirs["png"] / "\u91c7\u6837\u70b9\u5206\u5e03\u56fe.png")
    for load_index, (ox, ou) in enumerate(obs_payload):
        save_table(dirs["dat"] / f"L{load_index}_observation_points.dat", np.hstack([ox, ou]), "x y ux uy")

    start = time.time()
    losshistory, train_state = model.train(
        iterations=args.iterations,
        display_every=args.display_every,
        callbacks=[
            SpeedLogger(args.display_every),
            dde.callbacks.ModelCheckpoint(
                str(dirs["model"] / "best_model"),
                verbose=0,
                save_better_only=True,
                period=max(1, args.checkpoint_every),
                monitor="train loss",
            ),
        ],
    )
    elapsed = time.time() - start
    final_model_path = model.save(str(dirs["model"] / "final_model"))
    plot_loss_history(losshistory, dirs["png"] / "\u635f\u5931\u5386\u53f2\u56fe.png", dirs["npz"] / "loss_history.npz", loss_terms)

    loss_steps_arr = np.asarray(losshistory.steps)
    loss_train = np.asarray(losshistory.loss_train, dtype=float)
    loss_test = np.asarray(losshistory.loss_test, dtype=float)
    if loss_train.size:
        np.savetxt(
            dirs["dat"] / "loss_history.dat",
            np.column_stack([loss_steps_arr, loss_train, loss_test]),
            header="step train_loss_terms test_loss_terms",
        )
    if speed_rows:
        np.savetxt(
            dirs["dat"] / "speed_history.dat",
            np.asarray(speed_rows, dtype=float),
            header="step elapsed_seconds interval_sec_per_step",
        )

    eval_x = structured_grid_points(args.test_nx, args.test_ny, base_config.width, base_config.depth)
    val_x = make_validation_points(base_config, args.val_count, args.seed)
    metrics_by_split: dict[str, dict[str, dict[str, float]]] = {}
    field_names = ["ux", "uy", "sxx", "syy", "sxy"]
    for split_name, points in [("evaluation", eval_x), ("validation", val_x), ("observations", obs_x)]:
        raw = model.predict(points)
        split_metrics: dict[str, dict[str, float]] = {}
        for load_index, interp in enumerate(interpolators):
            off = 5 * load_index
            truth = interp.evaluate(points)[:, 0:5]
            pred = raw[:, off : off + 5]
            rel = {}
            for i, name in enumerate(field_names):
                rel[name] = float(np.linalg.norm(pred[:, i] - truth[:, i]) / (np.linalg.norm(truth[:, i]) + 1.0e-12))
            split_metrics[f"L{load_index}"] = rel
            np.savez(
                dirs["npz"] / f"{split_name}_L{load_index}_state_prediction.npz",
                x=points,
                y_true=truth,
                y_pred=pred,
                field_names=np.asarray(field_names),
            )
        metrics_by_split[split_name] = split_metrics

    runtime = {
        "status": "done",
        "elapsed_seconds": float(elapsed),
        "iterations": int(args.iterations),
        "seconds_per_step": float(elapsed / max(args.iterations, 1)),
        "load_count": int(num_loads),
        "boundary_scale": float(args.boundary_scale),
        "final_model_path": str(final_model_path),
        "best_step": int(train_state.best_step),
    }
    write_json(dirs["json"] / "runtime_status.json", runtime)
    metrics = {
        **runtime,
        "diagnostic": "official-style DeepXDE PFNN speed baseline",
        "metrics_by_split": metrics_by_split,
    }
    write_json(dirs["metrics"] / "metrics.json", metrics)
    (dirs["txt"] / "summary.txt").write_text(
        "\n".join(
            [
                "Official-style DeepXDE PFNN speed diagnostic",
                f"load_count={num_loads}",
                f"boundary_scale={args.boundary_scale}",
                f"iterations={args.iterations}",
                f"elapsed_seconds={elapsed:.6f}",
                f"seconds_per_step={elapsed / max(args.iterations, 1):.9f}",
                f"best_step={train_state.best_step}",
                f"final_model_path={final_model_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
