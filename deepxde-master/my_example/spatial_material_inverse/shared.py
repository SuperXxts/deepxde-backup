import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_repo_root():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.abspath(os.path.join(current_dir, "../..")),
        os.path.abspath(os.path.join(current_dir, "../..", "deepxde_work", "deepxde-master")),
    ]
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "deepxde")):
            return candidate
    raise FileNotFoundError("Cannot resolve DeepXDE repository root.")


ROOT_DIR = resolve_repo_root()
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import deepxde as dde

from utils.data_saving_utils import _get_save_path, save_prediction_data
from utils.device_utils import print_gpu_info, set_random_seed
from utils.loss_callback import LossHistoryCallback
from utils.progress_callback import TqdmProgressCallback
from utils.save_results import (
    plot_all_loss_components,
    plot_and_save_loss_history,
    save_best_test_loss_json,
    save_loss_history_json,
)


PI = math.pi
FIELD_NAMES = ["ux", "uy", "sxx", "syy", "sxy", "lambda", "mu"]
STATE_FIELD_NAMES = FIELD_NAMES[:5]
MATERIAL_FIELD_NAMES = FIELD_NAMES[5:]
DEFAULT_EXP_ROOT = os.path.join(ROOT_DIR, "exp", "spatial_material_inverse")


@dataclass
class CaseConfig:
    name: str
    lambda_bg: float = 1.0
    lambda_ctr_1: float = 0.8
    lambda_ctr_2: float = 0.0
    mu_bg: float = 0.7
    mu_ctr_1: float = 0.45
    mu_ctr_2: float = 0.0
    interface_width: float = 0.04
    layer_y: float = 0.5
    circle_1_cx: float = 0.5
    circle_1_cy: float = 0.5
    circle_1_r: float = 0.2
    circle_2_cx: float = 0.72
    circle_2_cy: float = 0.32
    circle_2_r: float = 0.14
    amplitude_u: float = 0.12
    amplitude_v: float = -0.08


def get_case_config(case_name):
    if case_name == "layered":
        return CaseConfig(name="layered")
    if case_name == "single_inclusion":
        return CaseConfig(
            name="single_inclusion",
            lambda_ctr_1=1.1,
            mu_ctr_1=0.65,
            interface_width=0.03,
        )
    if case_name == "double_inclusion":
        return CaseConfig(
            name="double_inclusion",
            lambda_ctr_1=1.0,
            lambda_ctr_2=-0.35,
            mu_ctr_1=0.55,
            mu_ctr_2=-0.18,
            interface_width=0.03,
        )
    raise ValueError(f"Unsupported case: {case_name}")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def parse_hidden_layers(value):
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(v.strip()) for v in str(value).split(",") if v.strip()]


def smooth_step_numpy(value):
    return 0.5 * (1.0 + np.tanh(value))


def smooth_step_torch(value):
    return 0.5 * (1.0 + torch.tanh(value))


def smooth_disk_numpy(x, y, cx, cy, radius, width):
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return smooth_step_numpy((radius - dist) / width)


def smooth_disk_torch(x, y, cx, cy, radius, width):
    dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2 + 1e-12)
    return smooth_step_torch((radius - dist) / width)


def exact_material_numpy(points, case_config):
    x = points[:, 0:1]
    y = points[:, 1:2]
    width = case_config.interface_width

    if case_config.name == "layered":
        mask = smooth_step_numpy((y - case_config.layer_y) / width)
        lmbd = case_config.lambda_bg + case_config.lambda_ctr_1 * mask
        mu = case_config.mu_bg + case_config.mu_ctr_1 * mask
        return lmbd, mu

    mask_1 = smooth_disk_numpy(
        x,
        y,
        case_config.circle_1_cx,
        case_config.circle_1_cy,
        case_config.circle_1_r,
        width,
    )
    lmbd = case_config.lambda_bg + case_config.lambda_ctr_1 * mask_1
    mu = case_config.mu_bg + case_config.mu_ctr_1 * mask_1

    if case_config.name == "double_inclusion":
        mask_2 = smooth_disk_numpy(
            x,
            y,
            case_config.circle_2_cx,
            case_config.circle_2_cy,
            case_config.circle_2_r,
            width,
        )
        lmbd = lmbd + case_config.lambda_ctr_2 * mask_2
        mu = mu + case_config.mu_ctr_2 * mask_2
    return lmbd, mu


def exact_material_torch(x, case_config):
    px = x[:, 0:1]
    py = x[:, 1:2]
    width = case_config.interface_width

    if case_config.name == "layered":
        mask = smooth_step_torch((py - case_config.layer_y) / width)
        lmbd = case_config.lambda_bg + case_config.lambda_ctr_1 * mask
        mu = case_config.mu_bg + case_config.mu_ctr_1 * mask
        return lmbd, mu

    mask_1 = smooth_disk_torch(
        px,
        py,
        case_config.circle_1_cx,
        case_config.circle_1_cy,
        case_config.circle_1_r,
        width,
    )
    lmbd = case_config.lambda_bg + case_config.lambda_ctr_1 * mask_1
    mu = case_config.mu_bg + case_config.mu_ctr_1 * mask_1

    if case_config.name == "double_inclusion":
        mask_2 = smooth_disk_torch(
            px,
            py,
            case_config.circle_2_cx,
            case_config.circle_2_cy,
            case_config.circle_2_r,
            width,
        )
        lmbd = lmbd + case_config.lambda_ctr_2 * mask_2
        mu = mu + case_config.mu_ctr_2 * mask_2
    return lmbd, mu


def exact_displacement_numpy(points, case_config):
    x = points[:, 0:1]
    y = points[:, 1:2]
    ux = case_config.amplitude_u * np.sin(PI * x) * np.sin(PI * y)
    uy = case_config.amplitude_v * np.sin(2.0 * PI * x) * np.sin(PI * y)
    return ux, uy


def exact_strain_numpy(points, case_config):
    x = points[:, 0:1]
    y = points[:, 1:2]
    exx = case_config.amplitude_u * PI * np.cos(PI * x) * np.sin(PI * y)
    eyy = case_config.amplitude_v * PI * np.sin(2.0 * PI * x) * np.cos(PI * y)
    exy = 0.5 * (
        case_config.amplitude_u * PI * np.sin(PI * x) * np.cos(PI * y)
        + case_config.amplitude_v * 2.0 * PI * np.cos(2.0 * PI * x) * np.sin(PI * y)
    )
    return exx, eyy, exy


def exact_state_numpy(points, case_config):
    ux, uy = exact_displacement_numpy(points, case_config)
    exx, eyy, exy = exact_strain_numpy(points, case_config)
    lmbd, mu = exact_material_numpy(points, case_config)
    trace = exx + eyy
    sxx = lmbd * trace + 2.0 * mu * exx
    syy = lmbd * trace + 2.0 * mu * eyy
    sxy = 2.0 * mu * exy
    return np.hstack((ux, uy, sxx, syy, sxy, lmbd, mu))


def exact_state_torch(x, case_config):
    px = x[:, 0:1]
    py = x[:, 1:2]
    ux = case_config.amplitude_u * torch.sin(PI * px) * torch.sin(PI * py)
    uy = case_config.amplitude_v * torch.sin(2.0 * PI * px) * torch.sin(PI * py)
    exx = case_config.amplitude_u * PI * torch.cos(PI * px) * torch.sin(PI * py)
    eyy = case_config.amplitude_v * PI * torch.sin(2.0 * PI * px) * torch.cos(PI * py)
    exy = 0.5 * (
        case_config.amplitude_u * PI * torch.sin(PI * px) * torch.cos(PI * py)
        + case_config.amplitude_v * 2.0 * PI * torch.cos(2.0 * PI * px) * torch.sin(PI * py)
    )
    lmbd, mu = exact_material_torch(x, case_config)
    trace = exx + eyy
    sxx = lmbd * trace + 2.0 * mu * exx
    syy = lmbd * trace + 2.0 * mu * eyy
    sxy = 2.0 * mu * exy
    return torch.cat((ux, uy, sxx, syy, sxy, lmbd, mu), dim=1)


def exact_body_force_torch(x, case_config):
    exact_state = exact_state_torch(x, case_config)
    sxx_x = dde.grad.jacobian(exact_state, x, i=2, j=0)
    sxy_y = dde.grad.jacobian(exact_state, x, i=4, j=1)
    sxy_x = dde.grad.jacobian(exact_state, x, i=4, j=0)
    syy_y = dde.grad.jacobian(exact_state, x, i=3, j=1)
    fx = -(sxx_x + sxy_y)
    fy = -(sxy_x + syy_y)
    return fx, fy


def relative_l2(pred, true):
    denom = np.linalg.norm(true.reshape(-1))
    if denom < 1e-12:
        return float(np.linalg.norm(pred.reshape(-1)))
    return float(np.linalg.norm((pred - true).reshape(-1)) / denom)


def mean_abs(pred, true):
    return float(np.mean(np.abs(pred - true)))


def make_grid(nx, ny):
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    return points, xx, yy


def build_observation_points(num_observe, seed):
    rng = np.random.default_rng(seed)
    return rng.random((num_observe, 2))


def build_observation_splits(num_train, num_val, num_eval, seed):
    total = int(num_train) + int(num_val) + int(num_eval)
    if total <= 0:
        raise ValueError("At least one observation point is required.")
    points = build_observation_points(total, seed)
    train_end = int(num_train)
    val_end = train_end + int(num_val)
    return {
        "train": points[:train_end],
        "val": points[train_end:val_end],
        "eval": points[val_end:],
    }


def add_noise(values, noise_level, seed):
    if noise_level <= 0:
        return values.copy()
    rng = np.random.default_rng(seed)
    scale = np.std(values, axis=0, keepdims=True)
    scale = np.where(scale < 1e-8, 1.0, scale)
    noise = rng.normal(0.0, 1.0, size=values.shape) * scale * noise_level
    return values + noise


class FourierFeatureMap(nn.Module):
    def __init__(self, num_frequencies):
        super().__init__()
        self.num_frequencies = int(num_frequencies)
        if self.num_frequencies > 0:
            freq_bands = 2.0 ** torch.arange(self.num_frequencies, dtype=torch.float32)
            self.register_buffer("freq_bands", freq_bands)
        else:
            self.freq_bands = None

    @property
    def output_dim(self):
        return 2 + 4 * self.num_frequencies

    def forward(self, x):
        if self.num_frequencies <= 0:
            return x
        features = [x]
        for freq in self.freq_bands:
            features.append(torch.sin(2.0 * PI * freq * x))
            features.append(torch.cos(2.0 * PI * freq * x))
        return torch.cat(features, dim=1)


def activation_factory(name):
    name = name.lower()
    if name == "tanh":
        return nn.Tanh
    if name == "relu":
        return nn.ReLU
    if name == "gelu":
        return nn.GELU
    if name == "silu":
        return nn.SiLU
    raise ValueError(f"Unsupported activation: {name}")


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, activation="tanh"):
        super().__init__()
        activation_cls = activation_factory(activation)
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_layers:
            layer = nn.Linear(in_dim, hidden_dim)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            layers.append(activation_cls())
            in_dim = hidden_dim
        final_layer = nn.Linear(in_dim, output_dim)
        nn.init.xavier_uniform_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class VanillaMaterialFieldNet(dde.nn.pytorch.nn.NN):
    def __init__(self, hidden_layers, activation="tanh", num_frequencies=0):
        super().__init__()
        self.features = FourierFeatureMap(num_frequencies)
        self.backbone = SimpleMLP(
            input_dim=self.features.output_dim,
            hidden_layers=hidden_layers,
            output_dim=len(FIELD_NAMES),
            activation=activation,
        )

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(inputs)
        features = self.features(x)
        outputs = self.backbone(features)
        if self._output_transform is not None:
            outputs = self._output_transform(inputs, outputs)
        return outputs


def num_regions_for_case(case_name):
    if case_name in {"layered", "single_inclusion"}:
        return 1
    if case_name == "double_inclusion":
        return 2
    raise ValueError(f"Unsupported case: {case_name}")


class InterfaceAwareMaterialNet(dde.nn.pytorch.nn.NN):
    def __init__(
        self,
        case_name,
        state_hidden_layers,
        interface_hidden_layers,
        activation="tanh",
        num_frequencies=0,
        lambda_floor=0.1,
        mu_floor=0.1,
        interface_sharpness=10.0,
    ):
        super().__init__()
        self.features = FourierFeatureMap(num_frequencies)
        self.num_regions = num_regions_for_case(case_name)
        self.lambda_floor = float(lambda_floor)
        self.mu_floor = float(mu_floor)
        self.interface_sharpness = float(interface_sharpness)
        feature_dim = self.features.output_dim
        self.state_net = SimpleMLP(
            input_dim=feature_dim,
            hidden_layers=state_hidden_layers,
            output_dim=5,
            activation=activation,
        )
        self.interface_net = SimpleMLP(
            input_dim=feature_dim,
            hidden_layers=interface_hidden_layers,
            output_dim=self.num_regions,
            activation=activation,
        )
        self.raw_lambda_params = nn.Parameter(torch.zeros(self.num_regions + 1, dtype=torch.float32))
        self.raw_mu_params = nn.Parameter(torch.zeros(self.num_regions + 1, dtype=torch.float32))

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(inputs)
        features = self.features(x)
        state_outputs = self.state_net(features)
        region_logits = self.interface_sharpness * self.interface_net(features)
        background_logit = torch.zeros((inputs.shape[0], 1), dtype=region_logits.dtype, device=region_logits.device)
        class_logits = torch.cat((background_logit, region_logits), dim=1)
        class_probs = torch.softmax(class_logits, dim=1)

        lambda_regions = self.lambda_floor + F.softplus(self.raw_lambda_params)
        mu_regions = self.mu_floor + F.softplus(self.raw_mu_params)
        lmbd = torch.sum(class_probs * lambda_regions.unsqueeze(0), dim=1, keepdim=True)
        mu = torch.sum(class_probs * mu_regions.unsqueeze(0), dim=1, keepdim=True)

        gate = (
            inputs[:, 0:1]
            * (1.0 - inputs[:, 0:1])
            * inputs[:, 1:2]
            * (1.0 - inputs[:, 1:2])
        )
        ux = gate * state_outputs[:, 0:1]
        uy = gate * state_outputs[:, 1:2]
        outputs = torch.cat((ux, uy, state_outputs[:, 2:5], lmbd, mu), dim=1)
        return outputs


def make_output_transform(lambda_floor, mu_floor, method=None):
    def output_transform(inputs, outputs):
        gate = (
            inputs[:, 0:1]
            * (1.0 - inputs[:, 0:1])
            * inputs[:, 1:2]
            * (1.0 - inputs[:, 1:2])
        )
        ux = gate * outputs[:, 0:1]
        uy = gate * outputs[:, 1:2]
        sxx = outputs[:, 2:3]
        syy = outputs[:, 3:4]
        sxy = outputs[:, 4:5]
        lmbd = lambda_floor + F.softplus(outputs[:, 5:6])
        mu = mu_floor + F.softplus(outputs[:, 6:7])
        return torch.cat((ux, uy, sxx, syy, sxy, lmbd, mu), dim=1)

    return output_transform


def count_trainable_parameters(net):
    return int(sum(parameter.numel() for parameter in net.parameters() if parameter.requires_grad))


def build_network(args):
    if args.method == "iaminn":
        net = InterfaceAwareMaterialNet(
            case_name=args.case,
            state_hidden_layers=parse_hidden_layers(args.state_layers),
            interface_hidden_layers=parse_hidden_layers(args.interface_layers),
            activation=args.activation,
            num_frequencies=args.num_frequencies,
            lambda_floor=args.lambda_floor,
            mu_floor=args.mu_floor,
            interface_sharpness=args.interface_sharpness,
        )
    else:
        net = VanillaMaterialFieldNet(
            hidden_layers=parse_hidden_layers(args.hidden_layers),
            activation=args.activation,
            num_frequencies=args.num_frequencies,
        )
        net.apply_output_transform(make_output_transform(args.lambda_floor, args.mu_floor, args.method))
    return net



def build_pde(case_config, reg_weight, method=None):
    def pde(x, y):
        ux_x = dde.grad.jacobian(y, x, i=0, j=0)
        ux_y = dde.grad.jacobian(y, x, i=0, j=1)
        uy_x = dde.grad.jacobian(y, x, i=1, j=0)
        uy_y = dde.grad.jacobian(y, x, i=1, j=1)
        exx = ux_x
        eyy = uy_y
        exy = 0.5 * (ux_y + uy_x)
        lambda_idx, mu_idx = 5, 6
        lmbd = y[:, lambda_idx:lambda_idx + 1]
        mu = y[:, mu_idx:mu_idx + 1]

        constitutive_sxx = lmbd * (exx + eyy) + 2.0 * mu * exx
        constitutive_syy = lmbd * (exx + eyy) + 2.0 * mu * eyy
        constitutive_sxy = 2.0 * mu * exy

        sxx = y[:, 2:3]
        syy = y[:, 3:4]
        sxy = y[:, 4:5]

        sxx_x = dde.grad.jacobian(sxx, x, i=0, j=0)
        syy_y = dde.grad.jacobian(syy, x, i=0, j=1)
        sxy_x = dde.grad.jacobian(sxy, x, i=0, j=0)
        sxy_y = dde.grad.jacobian(sxy, x, i=0, j=1)

        fx, fy = exact_body_force_torch(x, case_config)
        residuals = [
            sxx_x + sxy_y + fx,
            sxy_x + syy_y + fy,
            y[:, 2:3] - constitutive_sxx,
            y[:, 3:4] - constitutive_syy,
            y[:, 4:5] - constitutive_sxy,
        ]
        if reg_weight > 0.0:
            residuals.extend(
                [
                    dde.grad.jacobian(y, x, i=lambda_idx, j=0),
                    dde.grad.jacobian(y, x, i=lambda_idx, j=1),
                    dde.grad.jacobian(y, x, i=mu_idx, j=0),
                    dde.grad.jacobian(y, x, i=mu_idx, j=1),
                ]
            )
        return residuals

    return pde


def pde_loss_names(reg_weight, method=None):
    names = ["momentum_x", "momentum_y", "constitutive_xx", "constitutive_yy", "constitutive_xy"]
    if reg_weight > 0.0:
        names.extend(["lambda_x", "lambda_y", "mu_x", "mu_y"])
    return names


def pde_loss_weights(reg_weight, method=None):
    weights = [1.0, 1.0, 1.0, 1.0, 1.0]
    if reg_weight > 0.0:
        weights.extend([reg_weight, reg_weight, reg_weight, reg_weight])
    return weights


def build_observation_payload(points, case_config, noise_level, seed):
    if len(points) == 0:
        return {
            "points": np.zeros((0, 2), dtype=float),
            "clean": np.zeros((0, 2), dtype=float),
            "noisy": np.zeros((0, 2), dtype=float),
            "true_state": np.zeros((0, len(FIELD_NAMES)), dtype=float),
        }
    exact_observation_state = exact_state_numpy(points, case_config)
    clean_observation = exact_observation_state[:, :2]
    noisy_observation = add_noise(clean_observation, noise_level, seed)
    return {
        "points": points,
        "clean": clean_observation,
        "noisy": noisy_observation,
        "true_state": exact_observation_state,
    }


def build_data(args, case_config):
    geom = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])
    observation_splits = build_observation_splits(
        num_train=args.num_observe,
        num_val=args.num_val_observe,
        num_eval=args.num_eval_observe,
        seed=args.seed + 17,
    )
    train_observation = build_observation_payload(
        observation_splits["train"], case_config, args.noise_level, args.seed + 123
    )
    val_observation = build_observation_payload(
        observation_splits["val"], case_config, args.noise_level, args.seed + 223
    )
    eval_observation = build_observation_payload(
        observation_splits["eval"], case_config, args.noise_level, args.seed + 323
    )

    observe_ux = dde.icbc.PointSetBC(train_observation["points"], train_observation["noisy"][:, 0:1], component=0)
    observe_uy = dde.icbc.PointSetBC(train_observation["points"], train_observation["noisy"][:, 1:2], component=1)
    data = dde.data.PDE(
        geom,
        build_pde(case_config, args.reg_weight, args.method),
        [observe_ux, observe_uy],
        num_domain=args.num_domain,
        num_boundary=0,
        num_test=args.num_test,
        train_distribution="pseudo",
    )
    metadata = {
        "train_observation": train_observation,
        "val_observation": val_observation,
        "eval_observation": eval_observation,
    }
    return geom, data, metadata


def resolve_run_name(args):
    if args.run_name:
        return args.run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"{args.case}_obs{args.num_observe}_val{args.num_val_observe}_eval{args.num_eval_observe}_noise{args.noise_level:.3f}_"
        f"seed{args.seed}_iter{args.iterations}_{timestamp}"
    )


def resolve_save_dir(args):
    run_name = resolve_run_name(args)
    return os.path.join(args.exp_root, args.case, args.method, run_name)


def save_json(path, payload):
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, dict):
            return {key: to_serializable(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_serializable(item) for item in obj]
        return obj

    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, indent=2, ensure_ascii=False)


def save_case_and_sampling_figure(
    save_dir,
    case_config,
    domain_points,
    train_observation_points,
    val_observation_points,
    eval_observation_points,
    title_suffix,
    filename="sampling_and_case.png",
):
    grid_points, xx, yy = make_grid(200, 200)
    exact_fields = exact_state_numpy(grid_points, case_config)
    lambda_field = exact_fields[:, 5].reshape(xx.shape)
    mu_field = exact_fields[:, 6].reshape(xx.shape)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for axis, field, label in zip(
        axes[:2],
        [lambda_field, mu_field],
        [r"True $\lambda(x,y)$", r"True $\mu(x,y)$"],
    ):
        image = axis.contourf(xx, yy, field, levels=100, cmap="viridis")
        axis.set_title(label)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_aspect("equal")
        fig.colorbar(image, ax=axis)

    axes[2].scatter(domain_points[:, 0], domain_points[:, 1], s=8, alpha=0.25, label="Domain points")
    if len(train_observation_points) > 0:
        axes[2].scatter(
            train_observation_points[:, 0],
            train_observation_points[:, 1],
            s=18,
            alpha=0.85,
            color="#d97706",
            label="Train observations",
        )
    if len(val_observation_points) > 0:
        axes[2].scatter(
            val_observation_points[:, 0],
            val_observation_points[:, 1],
            s=22,
            alpha=0.85,
            color="#0f766e",
            marker="^",
            label="Validation observations",
        )
    if len(eval_observation_points) > 0:
        axes[2].scatter(
            eval_observation_points[:, 0],
            eval_observation_points[:, 1],
            s=22,
            alpha=0.85,
            color="#7c3aed",
            marker="s",
            label="Evaluation observations",
        )
    axes[2].set_title(f"Sampling layout ({title_suffix})")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_xlim(0.0, 1.0)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_aspect("equal")
    axes[2].legend(frameon=False, loc="upper right")
    plt.tight_layout()
    plt.savefig(_get_save_path(save_dir, "png", filename), dpi=300)
    plt.close(fig)


def save_training_artifacts(
    args,
    save_dir,
    case_config,
    losshistory,
    metadata,
    net,
):
    config_payload = {
        "args": {key: value for key, value in vars(args).items() if not key.startswith("_")},
        "case": asdict(case_config),
        "parameter_count": count_trainable_parameters(net),
        "field_names": FIELD_NAMES,
        "pde_loss_names": pde_loss_names(args.reg_weight, args.method),
        "bc_loss_names": ["obs_ux", "obs_uy"],
        "selection_metric": "validation_observation_mse",
    }
    save_json(_get_save_path(save_dir, "json", "run_config.json"), config_payload)
    np.savez(
        _get_save_path(save_dir, "npz", "observation_data.npz"),
        train_observation_points=metadata["train_observation"]["points"],
        train_observation_clean=metadata["train_observation"]["clean"],
        train_observation_noisy=metadata["train_observation"]["noisy"],
        train_observation_true_state=metadata["train_observation"]["true_state"],
        val_observation_points=metadata["val_observation"]["points"],
        val_observation_clean=metadata["val_observation"]["clean"],
        val_observation_noisy=metadata["val_observation"]["noisy"],
        val_observation_true_state=metadata["val_observation"]["true_state"],
        eval_observation_points=metadata["eval_observation"]["points"],
        eval_observation_clean=metadata["eval_observation"]["clean"],
        eval_observation_noisy=metadata["eval_observation"]["noisy"],
        eval_observation_true_state=metadata["eval_observation"]["true_state"],
    )

    domain_points = getattr(args, "_domain_points_for_plot", None)
    if domain_points is not None:
        np.savez(_get_save_path(save_dir, "npz", "domain_points.npz"), domain_points=domain_points)
        save_case_and_sampling_figure(
            save_dir=save_dir,
            case_config=case_config,
            domain_points=domain_points,
            train_observation_points=metadata["train_observation"]["points"],
            val_observation_points=metadata["val_observation"]["points"],
            eval_observation_points=metadata["eval_observation"]["points"],
            title_suffix=args.case,
        )

    if losshistory is not None:
        plot_and_save_loss_history(
            losshistory,
            save_dir,
            filename="loss_history.png",
            num_pde_losses=len(pde_loss_names(args.reg_weight, args.method)),
            num_bc_losses=2,
            pde_label="Physics Loss",
            bc_label="Observation Loss",
            bc_loss_names=["obs_ux", "obs_uy"],
            data_loss_prefix="obs_",
        )
        plot_all_loss_components(
            losshistory,
            save_dir,
            filename="loss_components.png",
            num_pde_losses=len(pde_loss_names(args.reg_weight, args.method)),
            num_bc_losses=2,
            pde_loss_names=pde_loss_names(args.reg_weight, args.method),
            bc_loss_names=["obs_ux", "obs_uy"],
        )
        save_loss_history_json(losshistory, save_dir, filename="loss_history.json")
        save_best_test_loss_json(losshistory, save_dir, filename="best_test_loss.json")


def latest_model_prefix(model_dir, prefix_name):
    if not os.path.isdir(model_dir):
        return None
    candidates = []
    for filename in os.listdir(model_dir):
        if not filename.startswith(prefix_name):
            continue
        candidates.append(os.path.join(model_dir, filename))
    if not candidates:
        return None
    candidates = sorted(
        set(candidates),
        key=os.path.getmtime,
    )
    return candidates[-1]


def find_model_path(save_dir, model_path=None, prefer="best_model"):
    if model_path:
        return model_path
    model_dir = os.path.join(save_dir, "model")
    candidate = latest_model_prefix(model_dir, prefer)
    if candidate is not None:
        return candidate
    fallback = latest_model_prefix(model_dir, "last_model")
    if fallback is not None:
        return fallback
    raise FileNotFoundError(f"No checkpoint found under {model_dir}")


def compute_metrics(prediction, truth, observation_prediction, observation_truth, pde_residual):
    metrics = {
        "field_relative_l2": {},
        "field_mae": {},
        "observation_mse": float(np.mean((observation_prediction - observation_truth) ** 2)),
        "pde_residual_mean_abs": float(np.mean(np.abs(pde_residual))),
    }
    for index, field_name in enumerate(FIELD_NAMES):
        metrics["field_relative_l2"][field_name] = relative_l2(
            prediction[:, index:index + 1],
            truth[:, index:index + 1],
        )
        metrics["field_mae"][field_name] = mean_abs(
            prediction[:, index:index + 1],
            truth[:, index:index + 1],
        )
    material_error = np.linalg.norm(prediction[:, 5:7] - truth[:, 5:7], axis=1)
    metrics["material_field_mean_abs_vector_error"] = float(np.mean(material_error))
    return metrics


def split_residual_prediction(residual_prediction):
    if isinstance(residual_prediction, list):
        return np.concatenate([np.asarray(item) for item in residual_prediction], axis=1)
    array = np.asarray(residual_prediction)
    if array.ndim == 1:
        return array[:, None]
    return array


def predict_full_fields(model, points, args, batch_size=4096):
    return np.asarray(model.predict(points))


def compute_observation_mse(model, args, observation_points, observation_truth):
    if len(observation_points) == 0:
        return float("nan")
    observation_prediction = predict_full_fields(model, observation_points, args)[:, :2]
    return float(np.mean((observation_prediction - observation_truth) ** 2))


def save_validation_history(history, save_dir):
    save_json(_get_save_path(save_dir, "json", "validation_history.json"), history)
    if not history["steps"]:
        return
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(history["steps"], history["validation_observation_mse"], color="#0f766e", linewidth=2.0)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Validation observation MSE")
    ax.set_title("Validation observation history")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(_get_save_path(save_dir, "png", "validation_history.png"), dpi=300)
    plt.close(fig)


class ValidationObservationCheckpoint(dde.callbacks.Callback):
    def __init__(self, filepath, save_dir, observation_points, observation_truth_clean, args, period=1, verbose=1):
        super().__init__()
        self.filepath = filepath
        self.save_dir = save_dir
        self.observation_points = np.asarray(observation_points, dtype=float)
        self.observation_truth_clean = np.asarray(observation_truth_clean, dtype=float)
        self.args = args
        self.period = period
        self.verbose = verbose
        self.epochs_since_last_save = 0
        self.best = np.inf
        self.best_file = None
        self.best_step = 0
        self.history = {
            "metric_name": "validation_observation_mse",
            "steps": [],
            "validation_observation_mse": [],
        }

    def on_train_begin(self):
        self._save_history()

    def on_epoch_end(self):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save < self.period:
            return
        self.epochs_since_last_save = 0
        current = compute_observation_mse(
            self.model,
            self.args,
            self.observation_points,
            self.observation_truth_clean,
        )
        step = int(self.model.train_state.iteration)
        self.history["steps"].append(step)
        self.history["validation_observation_mse"].append(float(current))
        self._save_history()
        if current < self.best:
            if self.best_file and os.path.exists(self.best_file):
                try:
                    os.remove(self.best_file)
                except OSError:
                    pass
            previous_best = self.best
            save_path = self.model.save(self.filepath, verbose=0)
            self.best_file = save_path
            self.best = float(current)
            self.best_step = step
            info = {
                "step": step,
                "monitor": "validation_observation_mse",
                "best_value": float(current),
                "loss_train_components": [float(x) for x in self.model.train_state.loss_train],
                "loss_test_components": [float(x) for x in self.model.train_state.loss_test]
                if self.model.train_state.loss_test is not None and len(self.model.train_state.loss_test) > 0
                else [],
                "model_path": os.path.basename(save_path),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            save_json(_get_save_path(self.save_dir, "json", "best_validation_info.json"), info)
            if self.verbose > 0:
                prev_text = "inf" if not np.isfinite(previous_best) else f"{previous_best:.2e}"
                print(
                    "Epoch {}: validation_observation_mse improved from {} to {:.2e}, saving model to {} ...\n".format(
                        step,
                        prev_text,
                        current,
                        save_path,
                    )
                )

    def on_train_end(self):
        self._save_history()

    def _save_history(self):
        save_validation_history(self.history, self.save_dir)


def reshape_grid(values, ny, nx):
    return values.reshape(ny, nx)


def plot_field_triplet(save_dir, xx, yy, truth_grid, pred_grid, field_name):
    abs_error = np.abs(pred_grid - truth_grid)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    titles = [
        f"True {field_name}",
        f"Predicted {field_name}",
        f"Absolute error of {field_name}",
    ]
    cmaps = ["viridis", "viridis", "magma"]
    for axis, grid, title, cmap in zip(axes, [truth_grid, pred_grid, abs_error], titles, cmaps):
        image = axis.contourf(xx, yy, grid, levels=100, cmap=cmap)
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_aspect("equal")
        plt.colorbar(image, ax=axis)
    plt.tight_layout()
    plt.savefig(_get_save_path(save_dir, "png", f"{field_name}_comparison.png"), dpi=300)
    plt.close(fig)


def plot_material_overlay(save_dir, xx, yy, truth_grid, pred_grid, field_name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    levels = np.linspace(
        min(np.min(truth_grid), np.min(pred_grid)),
        max(np.max(truth_grid), np.max(pred_grid)),
        24,
    )
    for axis, grid, title in zip(
        axes,
        [truth_grid, pred_grid],
        [f"True {field_name}", f"Predicted {field_name}"],
    ):
        image = axis.contourf(xx, yy, grid, levels=levels, cmap="viridis")
        axis.contour(xx, yy, grid, levels=levels[::3], colors="white", linewidths=0.5)
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_aspect("equal")
        plt.colorbar(image, ax=axis)
    plt.tight_layout()
    plt.savefig(_get_save_path(save_dir, "png", f"{field_name}_material_map.png"), dpi=300)
    plt.close(fig)


def plot_observation_fit(save_dir, observation_truth, observation_prediction):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, index, label in zip(axes, [0, 1], ["ux", "uy"]):
        axis.scatter(
            observation_truth[:, index],
            observation_prediction[:, index],
            s=16,
            alpha=0.75,
            color="#355F94",
        )
        limits = [
            float(min(np.min(observation_truth[:, index]), np.min(observation_prediction[:, index]))),
            float(max(np.max(observation_truth[:, index]), np.max(observation_prediction[:, index]))),
        ]
        axis.plot(limits, limits, linestyle="--", color="#d97706", linewidth=1.2)
        axis.set_xlabel(f"Observed {label}")
        axis.set_ylabel(f"Predicted {label}")
        axis.set_title(f"Observation fit for {label}")
        axis.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(_get_save_path(save_dir, "png", "observation_fit.png"), dpi=300)
    plt.close(fig)


def evaluate_model(
    args,
    model,
    save_dir,
    case_config,
    observation_points,
    observation_truth_clean,
    observation_split_name="evaluation",
    save_artifacts=True,
):
    eval_points, xx, yy = make_grid(args.eval_nx, args.eval_ny)
    truth = exact_state_numpy(eval_points, case_config)
    prediction = predict_full_fields(model, eval_points, args)
    residual = split_residual_prediction(
        model.predict(eval_points, operator=build_pde(case_config, args.reg_weight, args.method))
    )
    observation_prediction = predict_full_fields(model, observation_points, args)[:, :2]

    metrics = compute_metrics(
        prediction=prediction,
        truth=truth,
        observation_prediction=observation_prediction,
        observation_truth=observation_truth_clean,
        pde_residual=residual,
    )

    if save_artifacts:
        save_prediction_data(
            eval_points,
            prediction,
            truth,
            test_delta=None,
            save_dir=save_dir,
            prefix="evaluation_predictions",
            field_names=FIELD_NAMES,
        )
        np.savez(
            _get_save_path(save_dir, "npz", "evaluation_grid.npz"),
            points=eval_points,
            prediction=prediction,
            truth=truth,
            xx=xx,
            yy=yy,
            pde_residual=residual,
        )
        save_json(_get_save_path(save_dir, "metrics", f"{observation_split_name}_metrics.json"), metrics)

        for index, field_name in enumerate(FIELD_NAMES):
            truth_grid = reshape_grid(truth[:, index], args.eval_ny, args.eval_nx)
            pred_grid = reshape_grid(prediction[:, index], args.eval_ny, args.eval_nx)
            plot_field_triplet(save_dir, xx, yy, truth_grid, pred_grid, field_name)
            if field_name in MATERIAL_FIELD_NAMES:
                plot_material_overlay(save_dir, xx, yy, truth_grid, pred_grid, field_name)

        plot_observation_fit(save_dir, observation_truth_clean, observation_prediction)
    return metrics


def build_model(args, data):
    net = build_network(args)
    model = dde.Model(data, net)
    loss_weights = pde_loss_weights(args.reg_weight, args.method) + [args.data_weight, args.data_weight]
    model.compile("adam", lr=args.lr, loss_weights=loss_weights)
    return model, net


def make_callbacks(args, save_dir, metadata):
    num_pde = len(pde_loss_names(args.reg_weight, args.method))
    model_dir = ensure_dir(os.path.join(save_dir, "model"))
    callbacks = [
        LossHistoryCallback(
            save_dir=save_dir,
            period=args.display_every,
            filename="loss_history.png",
            num_pde_losses=num_pde,
            num_bc_losses=2,
            pde_loss_names=pde_loss_names(args.reg_weight, args.method),
            bc_loss_names=["obs_ux", "obs_uy"],
            pde_label="Physics Loss",
            bc_label="Observation Loss",
            save_all_components=True,
        ),
        ValidationObservationCheckpoint(
            filepath=os.path.join(model_dir, "best_model"),
            save_dir=save_dir,
            observation_points=metadata["val_observation"]["points"],
            observation_truth_clean=metadata["val_observation"]["noisy"],
            args=args,
            period=args.display_every,
            verbose=1,
        ),
        TqdmProgressCallback(
            total_steps=args.iterations,
            display_every=args.display_every,
            metric_name="relL2",
        ),
    ]
    return callbacks


def save_last_model(model, save_dir):
    model_dir = ensure_dir(os.path.join(save_dir, "model"))
    model.save(os.path.join(model_dir, "last_model"))


def build_common_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--method",
        choices=["pinn", "iaminn"],
        default="pinn",
    )
    parser.add_argument(
        "--case",
        choices=["layered", "single_inclusion", "double_inclusion"],
        default="single_inclusion",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--display_every", type=int, default=1000)
    parser.add_argument("--num_domain", type=int, default=4000)
    parser.add_argument("--num_test", type=int, default=2000)
    parser.add_argument("--num_observe", type=int, default=500)
    parser.add_argument("--num_val_observe", type=int, default=100)
    parser.add_argument("--num_eval_observe", type=int, default=250)
    parser.add_argument("--noise_level", type=float, default=0.0)
    parser.add_argument("--reg_weight", type=float, default=1e-4)
    parser.add_argument("--data_weight", type=float, default=20.0)
    parser.add_argument("--lambda_floor", type=float, default=0.1)
    parser.add_argument("--mu_floor", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument("--hidden_layers", type=str, default="128,128,128,128")
    parser.add_argument("--state_layers", type=str, default="128,128,128,128")
    parser.add_argument("--interface_layers", type=str, default="64,64,64")
    parser.add_argument("--interface_sharpness", type=float, default=10.0)
    parser.add_argument("--num_frequencies", type=int, default=4)
    parser.add_argument("--exp_root", type=str, default=DEFAULT_EXP_ROOT)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--eval_nx", type=int, default=141)
    parser.add_argument("--eval_ny", type=int, default=141)
    parser.add_argument("--device_debug", action="store_true")
    return parser


def prepare_run(args):
    if args.num_val_observe <= 0:
        raise ValueError("num_val_observe must be positive for fair model selection.")
    if args.num_eval_observe <= 0:
        raise ValueError("num_eval_observe must be positive for held-out evaluation.")
    set_random_seed(args.seed)
    if args.device_debug:
        print_gpu_info()
    case_config = get_case_config(args.case)
    if args.save_dir is None:
        args.save_dir = resolve_save_dir(args)
    ensure_dir(args.save_dir)
    return case_config
