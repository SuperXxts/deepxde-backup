#!/usr/bin/env python3
"""DeepXDE PFNN runner for boundary-uncertainty inversion experiments.

Version 2 uses a fixed-Poisson-ratio single material field by default:
K = K0 * exp(m), mu = mu0 * exp(m). This avoids turning the first
identifiability benchmark into an underconstrained two-material-field problem.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("DDE_BACKEND", "pytorch")


CASE_CHOICES = (
    "full_boundary_oracle",
    "full_boundary_oracle_reaction",
    "correct_top_amp_only",
    "correct_top_amp_only_reaction",
    "oracle_A",
    "wrong_fixed_A",
    "wrong_fixed_A_reaction",
    "learnable_A",
    "learnable_A_anchor",
    "learnable_A_reaction",
    "learnable_A_anchor_reaction",
    "full_boundary_aware",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--case", default="learnable_A_reaction", choices=CASE_CHOICES)
    parser.add_argument("--material-mode", choices=("single", "dual"), default="single")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--display-every", type=int, default=100)
    parser.add_argument("--num-domain", type=int, default=500)
    parser.add_argument("--num-boundary", type=int, default=160)
    parser.add_argument("--obs-grid", type=int, default=8)
    parser.add_argument("--val-grid", type=int, default=31)
    parser.add_argument("--test-grid", type=int, default=101)
    parser.add_argument("--anchor-count", type=int, default=4)
    parser.add_argument("--reaction-points", type=int, default=100)
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--nu", type=float, default=0.30)
    parser.add_argument("--true-A", type=float, default=1.0)
    parser.add_argument("--init-A", type=float, default=0.85)
    parser.add_argument("--fixed-A", type=float, default=0.85)
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

    from bupinn import analytical
    from bupinn.io import ensure_run_dirs, save_table, write_json
    from bupinn.plotting import (
        plot_amplitude_history,
        plot_boundary_curve,
        plot_field_triplet,
        plot_line_slice,
        plot_loss_history,
        plot_sampling,
    )

    dirs = ensure_run_dirs(args.run_dir)
    np.random.seed(args.seed)
    dde.config.set_random_seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    geom = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])
    pi = np.pi
    k0 = 1.0 / (3.0 * (1.0 - 2.0 * args.nu))
    mu0 = 1.0 / (2.0 * (1.0 + args.nu))
    output_dim = 6 if args.material_mode == "single" else 7

    def lambda_from_k_mu(k, mu):
        return k - 2.0 * mu / 3.0

    def bkd_material_factor(x):
        xp = x[:, 0:1]
        yp = x[:, 1:2]
        layer = 1.0 + 0.22 * torch.tanh((yp - 0.55) / 0.045)
        lens_soft = -0.18 * torch.exp(-((xp - 0.68) ** 2 / 0.018 + (yp - 0.35) ** 2 / 0.012))
        lens_stiff = 0.16 * torch.exp(-((xp - 0.32) ** 2 / 0.012 + (yp - 0.72) ** 2 / 0.018))
        ripple = 0.04 * torch.sin(2.0 * pi * xp) * torch.sin(pi * yp)
        return torch.clamp(layer + lens_soft + lens_stiff + ripple, 0.45, 1.65)

    def bkd_displacement_derivatives(x, amplitude):
        xp = x[:, 0:1]
        yp = x[:, 1:2]
        dux_dx = 0.05 * pi * torch.cos(pi * xp) * torch.sin(pi * yp) * (1.0 - yp)
        dux_dy = 0.05 * torch.sin(pi * xp) * (pi * torch.cos(pi * yp) * (1.0 - yp) - torch.sin(pi * yp))
        duy_dx = yp * (0.04 * pi * torch.cos(pi * xp) + 0.04 * pi * torch.cos(2.0 * pi * xp) * yp)
        duy_dy = 0.15 + 0.04 * torch.sin(pi * xp) + 0.04 * torch.sin(2.0 * pi * xp) * yp
        exx = amplitude * dux_dx
        eyy = amplitude * duy_dy
        exy = 0.5 * amplitude * (dux_dy + duy_dx)
        return exx, eyy, exy

    def stress_from_k_mu(k, mu, exx, eyy, exy):
        lam = lambda_from_k_mu(k, mu)
        sxx = (2.0 * mu + lam) * exx + lam * eyy
        syy = lam * exx + (2.0 * mu + lam) * eyy
        sxy = 2.0 * mu * exy
        return sxx, syy, sxy

    def true_stress_torch(x):
        factor = bkd_material_factor(x)
        k_true = k0 * factor
        mu_true = mu0 * factor
        exx, eyy, exy = bkd_displacement_derivatives(x, args.true_A)
        sxx, syy, sxy = stress_from_k_mu(k_true, mu_true, exx, eyy, exy)
        return torch.cat([sxx, syy, sxy], dim=1)

    def body_force(x):
        sig = true_stress_torch(x)
        sxx_x = dde.grad.jacobian(sig, x, i=0, j=0)
        sxy_y = dde.grad.jacobian(sig, x, i=2, j=1)
        sxy_x = dde.grad.jacobian(sig, x, i=2, j=0)
        syy_y = dde.grad.jacobian(sig, x, i=1, j=1)
        return sxx_x + sxy_y, sxy_x + syy_y

    def jacobian(y, x, i, j):
        return dde.grad.jacobian(y, x, i=i, j=j)

    def material_from_output(y):
        if args.material_mode == "single":
            z = torch.clamp(y[:, 5:6], min=-3.0, max=3.0)
            return k0 * torch.exp(z), mu0 * torch.exp(z)
        zk = torch.clamp(y[:, 5:6], min=-3.0, max=3.0)
        zmu = torch.clamp(y[:, 6:7], min=-3.0, max=3.0)
        return k0 * torch.exp(zk), mu0 * torch.exp(zmu)

    def pde(x, y):
        ux_x = jacobian(y, x, 0, 0)
        uy_y = jacobian(y, x, 1, 1)
        ux_y = jacobian(y, x, 0, 1)
        uy_x = jacobian(y, x, 1, 0)
        exx = ux_x
        eyy = uy_y
        exy = 0.5 * (ux_y + uy_x)

        k_pred, mu_pred = material_from_output(y)
        sxx_c, syy_c, sxy_c = stress_from_k_mu(k_pred, mu_pred, exx, eyy, exy)

        sxx_x = jacobian(y, x, 2, 0)
        syy_y = jacobian(y, x, 3, 1)
        sxy_x = jacobian(y, x, 4, 0)
        sxy_y = jacobian(y, x, 4, 1)
        fx, fy = body_force(x)
        return [
            sxx_x + sxy_y - fx,
            sxy_x + syy_y - fy,
            sxx_c - y[:, 2:3],
            syy_c - y[:, 3:4],
            sxy_c - y[:, 4:5],
        ]

    def top_shape_torch(x):
        xp = x[:, 0:1]
        return 0.15 + 0.04 * torch.sin(pi * xp) + 0.02 * torch.sin(2.0 * pi * xp)

    def boundary_bottom(x, on_boundary):
        return on_boundary and dde.utils.isclose(x[1], 0.0)

    def boundary_top(x, on_boundary):
        return on_boundary and dde.utils.isclose(x[1], 1.0)

    def boundary_left(x, on_boundary):
        return on_boundary and dde.utils.isclose(x[0], 0.0)

    def boundary_right(x, on_boundary):
        return on_boundary and dde.utils.isclose(x[0], 1.0)

    trainable_cases = {"learnable_A", "learnable_A_anchor", "learnable_A_reaction", "learnable_A_anchor_reaction", "full_boundary_aware"}
    anchor_cases = {"learnable_A_anchor", "learnable_A_anchor_reaction", "full_boundary_aware"}
    reaction_cases = {
        "full_boundary_oracle_reaction",
        "correct_top_amp_only_reaction",
        "learnable_A_reaction",
        "learnable_A_anchor_reaction",
        "full_boundary_aware",
        "wrong_fixed_A_reaction",
    }
    correct_top_cases = {
        "full_boundary_oracle",
        "full_boundary_oracle_reaction",
        "correct_top_amp_only",
        "correct_top_amp_only_reaction",
        "oracle_A",
    }

    uses_trainable_a = args.case in trainable_cases
    uses_anchor = args.case in anchor_cases
    uses_reaction = args.case in reaction_cases
    uses_full_boundary = args.case in {"full_boundary_oracle", "full_boundary_oracle_reaction", "full_boundary_aware"}

    if uses_trainable_a:
        amplitude_var = dde.Variable(args.init_A)
        external_vars = [amplitude_var]
        amplitude_for_bc = amplitude_var
    elif args.case in correct_top_cases:
        amplitude_var = None
        external_vars = []
        amplitude_for_bc = float(args.true_A)
    else:
        amplitude_var = None
        external_vars = []
        amplitude_for_bc = float(args.fixed_A)

    def top_uy_bc(inputs, outputs, _X):
        return outputs[:, 1:2] - amplitude_for_bc * top_shape_torch(inputs)

    def top_ux_bc(inputs, outputs, _X):
        return outputs[:, 0:1]

    bcs = [
        dde.icbc.DirichletBC(geom, lambda x: 0.0, boundary_bottom, component=0),
        dde.icbc.DirichletBC(geom, lambda x: 0.0, boundary_bottom, component=1),
        dde.icbc.OperatorBC(geom, top_ux_bc, boundary_top),
        dde.icbc.OperatorBC(geom, top_uy_bc, boundary_top),
    ]

    if uses_full_boundary:
        bcs.extend(
            [
                dde.icbc.DirichletBC(geom, lambda x: 0.0, boundary_left, component=0),
                dde.icbc.DirichletBC(geom, lambda x: args.true_A * 0.15 * x[:, 1:2], boundary_left, component=1),
                dde.icbc.DirichletBC(geom, lambda x: 0.0, boundary_right, component=0),
                dde.icbc.DirichletBC(geom, lambda x: args.true_A * 0.15 * x[:, 1:2], boundary_right, component=1),
            ]
        )

    obs_x = analytical.make_grid(args.obs_grid, include_boundary=False)
    obs_u = analytical.add_noise(analytical.displacement(obs_x, args.true_A), args.noise_level, args.seed)
    bcs.append(dde.icbc.PointSetBC(obs_x, obs_u, component=[0, 1]))

    anchor_x = analytical.boundary_anchor_points(args.anchor_count if uses_anchor else 0)
    anchor_u = analytical.displacement(anchor_x, args.true_A) if len(anchor_x) else np.empty((0, 2), dtype=np.float32)
    if len(anchor_x):
        bcs.append(dde.icbc.PointSetBC(anchor_x, anchor_u, component=[0, 1]))

    reaction_x = np.column_stack(
        [np.linspace(0.0, 1.0, max(2, args.reaction_points)), np.ones(max(2, args.reaction_points))]
    ).astype(np.float32)
    reaction_weights = np.ones((len(reaction_x), 1), dtype=np.float32) / max(1, len(reaction_x) - 1)
    reaction_weights[0, 0] *= 0.5
    reaction_weights[-1, 0] *= 0.5
    true_reaction_density = analytical.stress_np(reaction_x, args.true_A, args.nu)[:, 1:2]
    true_reaction_y = float(np.sum(true_reaction_density * reaction_weights))

    if uses_reaction:
        target_reaction = np.full((len(reaction_x), 1), true_reaction_y, dtype=np.float32)
        point_lookup = {
            tuple(np.round(pt, decimals=8).tolist()): float(reaction_weights[i, 0])
            for i, pt in enumerate(reaction_x)
        }

        def reaction_operator(inputs, outputs, _X):
            device = outputs.device
            dtype = outputs.dtype
            if torch.is_tensor(inputs):
                x_np = inputs.detach().cpu().numpy()
            else:
                x_np = np.asarray(inputs, dtype=float)
            weights_full = np.zeros((x_np.shape[0], 1), dtype=float)
            rounded = np.round(x_np[:, :2], decimals=8)
            for row_index, pt in enumerate(rounded):
                weights_full[row_index, 0] = point_lookup.get(tuple(pt.tolist()), 0.0)
            weights_t = torch.as_tensor(weights_full, dtype=dtype, device=device)
            syy = outputs[:, 3:4]
            resultant = torch.sum(syy * weights_t)
            return torch.ones((outputs.shape[0], 1), dtype=dtype, device=device) * resultant

        bcs.append(dde.icbc.PointSetOperatorBC(reaction_x, target_reaction, reaction_operator))

    data = dde.data.PDE(
        geom,
        pde,
        bcs,
        num_domain=args.num_domain,
        num_boundary=args.num_boundary,
        train_distribution="Hammersley",
        num_test=None,
    )
    layer_sizes = [2] + [[args.width] * output_dim for _ in range(args.depth)] + [output_dim]
    net = dde.nn.PFNN(layer_sizes, "tanh", "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=args.lr, external_trainable_variables=external_vars)

    class AmplitudeLogger(dde.callbacks.Callback):
        def __init__(self, path: Path, period: int = 100):
            super().__init__()
            self.path = path
            self.period = period
            self.rows: list[tuple[int, float]] = []

        def _value(self) -> float:
            if amplitude_var is None:
                return float(amplitude_for_bc)
            return float(amplitude_var.detach().cpu().item())

        def on_train_begin(self):
            self.rows.append((0, self._value()))

        def on_epoch_end(self):
            step = int(self.model.train_state.iteration)
            if step % self.period == 0:
                value = self._value()
                self.rows.append((step, value))
                print(f"STEP {step} A {value:.8f}", flush=True)

        def on_train_end(self):
            arr = np.asarray(self.rows, dtype=float)
            np.savetxt(self.path, arr, header="step A")

    amp_log = AmplitudeLogger(dirs["dat"] / "amplitude_history.dat", period=max(1, args.display_every))
    checkpoint = dde.callbacks.ModelCheckpoint(
        str(dirs["model"] / "best_model"),
        verbose=0,
        save_better_only=True,
        period=max(1, args.display_every),
        monitor="train loss",
    )

    loss_terms = [
        "momentum_x",
        "momentum_y",
        "constitutive_sxx",
        "constitutive_syy",
        "constitutive_sxy",
        "bottom_ux",
        "bottom_uy",
        "top_ux",
        "top_uy",
    ]
    if uses_full_boundary:
        loss_terms.extend(["left_ux", "left_uy", "right_ux", "right_uy"])
    loss_terms.append("obs_u")
    if uses_anchor:
        loss_terms.append("anchor_u")
    if uses_reaction:
        loss_terms.append("global_reaction_y")

    config = vars(args).copy()
    config.update(
        {
            "backend": dde.backend.backend_name,
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "torch_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "network": {"type": "DeepXDE PFNN", "layer_sizes": layer_sizes},
            "material_parameterization": (
                "single: K=K0*exp(m), mu=mu0*exp(m)"
                if args.material_mode == "single"
                else "dual: K=K0*exp(zK), mu=mu0*exp(zmu)"
            ),
            "k0": float(k0),
            "mu0": float(mu0),
            "uses_anchor": bool(uses_anchor),
            "uses_reaction": bool(uses_reaction),
            "uses_full_boundary": bool(uses_full_boundary),
            "true_reaction_y": float(true_reaction_y),
            "loss_terms": loss_terms,
            "field_names": ["ux", "uy", "sxx", "syy", "sxy", "K", "mu", "E", "nu", "m"],
        }
    )
    write_json(dirs["json"] / "config.json", config)
    write_json(dirs["json"] / "run_config.json", config)
    plot_sampling(obs_x, anchor_x, dirs["png"] / "sampling_layout.png")
    save_table(dirs["dat"] / "observation_points.dat", np.hstack([obs_x, obs_u]), "x y ux uy")
    save_table(
        dirs["dat"] / "reaction_points.dat",
        np.hstack([reaction_x, reaction_weights, true_reaction_density]),
        "x y integration_weight true_syy",
    )
    if len(anchor_x):
        save_table(dirs["dat"] / "anchor_points.dat", np.hstack([anchor_x, anchor_u]), "x y ux uy")

    start = time.time()
    losshistory, train_state = model.train(
        iterations=args.iterations,
        display_every=args.display_every,
        callbacks=[amp_log, checkpoint],
    )
    elapsed = time.time() - start
    model.save(str(dirs["model"] / "final_model"))

    plot_loss_history(losshistory, dirs["png"] / "loss_history.png", dirs["npz"] / "loss_history.npz")
    plot_amplitude_history(dirs["dat"] / "amplitude_history.dat", dirs["png"] / "amplitude_history.png", args.true_A)

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

    def assemble_fields(x: np.ndarray, raw: np.ndarray) -> np.ndarray:
        if args.material_mode == "single":
            m = np.clip(raw[:, 5:6], -3.0, 3.0)
            pred_k = k0 * np.exp(m)
            pred_mu = mu0 * np.exp(m)
        else:
            zk = np.clip(raw[:, 5:6], -3.0, 3.0)
            zmu = np.clip(raw[:, 6:7], -3.0, 3.0)
            pred_k = k0 * np.exp(zk)
            pred_mu = mu0 * np.exp(zmu)
            m = np.log(np.maximum(pred_k / k0, 1e-12))
        pred_e = 9.0 * pred_k * pred_mu / (3.0 * pred_k + pred_mu)
        pred_nu = (3.0 * pred_k - 2.0 * pred_mu) / (2.0 * (3.0 * pred_k + pred_mu))
        return np.column_stack([raw[:, 0:5], pred_k, pred_mu, pred_e, pred_nu, m])

    def true_fields(x: np.ndarray) -> np.ndarray:
        true_basic = analytical.all_fields(x, args.true_A, args.nu)
        true_e = true_basic[:, 5:6]
        true_k = k0 * true_e
        true_mu = mu0 * true_e
        true_nu = np.full_like(true_e, args.nu)
        true_m = np.log(np.maximum(true_e, 1e-12))
        return np.column_stack([true_basic[:, 0:5], true_k, true_mu, true_e, true_nu, true_m])

    def relative_l2(y_true: np.ndarray, y_pred: np.ndarray, names: list[str]) -> dict[str, float]:
        out = {}
        for i, name in enumerate(names):
            denom = np.linalg.norm(y_true[:, i]) + 1e-12
            out[name] = float(np.linalg.norm(y_pred[:, i] - y_true[:, i]) / denom)
        return out

    names = ["ux", "uy", "sxx", "syy", "sxy", "K", "mu", "E", "nu", "m"]
    split_defs = {
        "evaluation": analytical.make_grid(args.test_grid, include_boundary=True),
        "validation": analytical.make_grid(args.val_grid, include_boundary=False),
        "train": obs_x,
    }
    split_metrics: dict[str, dict[str, float]] = {}
    eval_payload = {}
    for split_name, x_split in split_defs.items():
        raw = model.predict(x_split)
        pred = assemble_fields(x_split, raw)
        truth = true_fields(x_split)
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
        write_json(
            dirs["json"] / f"{split_name}_predictions_metadata.json",
            {
                "split": split_name,
                "num_points": int(len(x_split)),
                "field_names": names,
                "columns": ["x", "y"] + [f"true_{n}" for n in names] + [f"pred_{n}" for n in names] + [f"err_{n}" for n in names],
            },
        )
        if split_name != "train":
            for i, name in enumerate(names):
                np.savez(
                    dirs["npz"] / f"{split_name}_predictions_{name}.npz",
                    x=x_split,
                    y_true=truth[:, i],
                    y_pred=pred[:, i],
                    error=pred[:, i] - truth[:, i],
                )
                plot_field_triplet(x_split, truth[:, i], pred[:, i], name, dirs["png"] / f"{split_name}_field_{name}.png")
        eval_payload[split_name] = {"x": x_split, "truth": truth, "pred": pred, "raw": raw}

    x_test = split_defs["evaluation"]
    pred = eval_payload["evaluation"]["pred"]
    truth = eval_payload["evaluation"]["truth"]
    plot_line_slice(
        x_test,
        {"uy": (truth[:, 1], pred[:, 1]), "K": (truth[:, 5], pred[:, 5]), "mu": (truth[:, 6], pred[:, 6])},
        dirs["png"] / "slice_y_0p5.png",
        y_value=0.5,
    )
    for y_value in (0.25, 0.50, 0.75):
        plot_line_slice(
            x_test,
            {"K": (truth[:, 5], pred[:, 5]), "mu": (truth[:, 6], pred[:, 6]), "m": (truth[:, 9], pred[:, 9])},
            dirs["png"] / f"slice_material_y_{str(y_value).replace('.', 'p')}.png",
            y_value=y_value,
        )

    top_pred_raw = model.predict(reaction_x)
    top_pred_reaction_density = top_pred_raw[:, 3:4]
    pred_reaction_y = float(np.sum(top_pred_reaction_density * reaction_weights))
    reaction_rel_error = abs(pred_reaction_y - true_reaction_y) / (abs(true_reaction_y) + 1e-12)
    plot_boundary_curve(
        reaction_x,
        analytical.displacement(reaction_x, args.true_A)[:, 1],
        top_pred_raw[:, 1],
        "top uy",
        dirs["png"] / "top_uy_curve.png",
    )
    plot_boundary_curve(
        reaction_x,
        true_reaction_density[:, 0],
        top_pred_reaction_density[:, 0],
        "top syy reaction density",
        dirs["png"] / "top_reaction_density_curve.png",
    )

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
    )
    final_A = amp_log.rows[-1][1] if amp_log.rows else float(amplitude_for_bc)
    metrics = {
        "case": args.case,
        "material_mode": args.material_mode,
        "iterations": args.iterations,
        "elapsed_seconds": elapsed,
        "final_A": float(final_A),
        "true_A": float(args.true_A),
        "relative_l2": split_metrics["evaluation"],
        "split_relative_l2": split_metrics,
        "true_reaction_y": float(true_reaction_y),
        "pred_reaction_y": float(pred_reaction_y),
        "reaction_rel_error": float(reaction_rel_error),
        "best_step": int(train_state.best_step),
        "final_train_loss_sum": float(np.sum(train_state.loss_train)),
        "final_test_loss_sum": float(np.sum(train_state.loss_test)),
    }
    write_json(dirs["metrics"] / "metrics.json", metrics)
    write_json(dirs["metrics"] / "evaluation_metrics.json", {"relative_l2": split_metrics["evaluation"]})
    write_json(dirs["metrics"] / "validation_metrics.json", {"relative_l2": split_metrics["validation"]})
    write_json(dirs["metrics"] / "train_metrics.json", {"relative_l2": split_metrics["train"]})
    write_json(dirs["json"] / "training_summary.json", metrics)
    write_json(dirs["json"] / "runtime_status.json", {"status": "completed", "elapsed_seconds": elapsed})
    print("RUN_COMPLETE", metrics, flush=True)


if __name__ == "__main__":
    main()
