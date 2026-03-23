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
DEFAULT_EXP_ROOT = os.path.join(ROOT_DIR, "exp")
DEFAULT_OBSERVATION_CACHE_DIR = os.path.join(DEFAULT_EXP_ROOT, "_shared_observation_splits", "spatial_material_inverse")


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


def parse_load_scales(value):
    if isinstance(value, (list, tuple)):
        scales = [float(v) for v in value]
    else:
        scales = [float(v.strip()) for v in str(value).split(",") if v.strip()]
    if not scales:
        raise ValueError("At least one load scale is required.")
    return scales


def compact_num_loads(args_or_method, load_scales=None):
    if hasattr(args_or_method, "method"):
        method = args_or_method.method
        load_scale_value = getattr(args_or_method, "load_scales", "1.0")
    else:
        method = str(args_or_method)
        load_scale_value = load_scales if load_scales is not None else "1.0"
    if method == "geoiaminn_v3":
        return len(parse_load_scales(load_scale_value))
    return 1


def compact_material_indices(args):
    num_loads = compact_num_loads(args)
    if args.method == "geoiaminn_v3":
        return 2 * num_loads, 2 * num_loads + 1
    return 2, 3


def compact_state_indices(args, load_index=0):
    if args.method == "geoiaminn_v3":
        start = 2 * int(load_index)
        return start, start + 1
    return 0, 1


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


def exact_displacement_numpy(points, case_config, load_scale=1.0):
    x = points[:, 0:1]
    y = points[:, 1:2]
    ux = load_scale * case_config.amplitude_u * np.sin(PI * x) * np.sin(PI * y)
    uy = load_scale * case_config.amplitude_v * np.sin(2.0 * PI * x) * np.sin(PI * y)
    return ux, uy


def exact_strain_numpy(points, case_config, load_scale=1.0):
    x = points[:, 0:1]
    y = points[:, 1:2]
    exx = load_scale * case_config.amplitude_u * PI * np.cos(PI * x) * np.sin(PI * y)
    eyy = load_scale * case_config.amplitude_v * PI * np.sin(2.0 * PI * x) * np.cos(PI * y)
    exy = 0.5 * (
        load_scale * case_config.amplitude_u * PI * np.sin(PI * x) * np.cos(PI * y)
        + load_scale * case_config.amplitude_v * 2.0 * PI * np.cos(2.0 * PI * x) * np.sin(PI * y)
    )
    return exx, eyy, exy


def exact_state_numpy(points, case_config, load_scale=1.0):
    ux, uy = exact_displacement_numpy(points, case_config, load_scale=load_scale)
    exx, eyy, exy = exact_strain_numpy(points, case_config, load_scale=load_scale)
    lmbd, mu = exact_material_numpy(points, case_config)
    trace = exx + eyy
    sxx = lmbd * trace + 2.0 * mu * exx
    syy = lmbd * trace + 2.0 * mu * eyy
    sxy = 2.0 * mu * exy
    return np.hstack((ux, uy, sxx, syy, sxy, lmbd, mu))


def exact_state_torch(x, case_config, load_scale=1.0):
    px = x[:, 0:1]
    py = x[:, 1:2]
    ux = load_scale * case_config.amplitude_u * torch.sin(PI * px) * torch.sin(PI * py)
    uy = load_scale * case_config.amplitude_v * torch.sin(2.0 * PI * px) * torch.sin(PI * py)
    exx = load_scale * case_config.amplitude_u * PI * torch.cos(PI * px) * torch.sin(PI * py)
    eyy = load_scale * case_config.amplitude_v * PI * torch.sin(2.0 * PI * px) * torch.cos(PI * py)
    exy = 0.5 * (
        load_scale * case_config.amplitude_u * PI * torch.sin(PI * px) * torch.cos(PI * py)
        + load_scale * case_config.amplitude_v * 2.0 * PI * torch.cos(2.0 * PI * px) * torch.sin(PI * py)
    )
    lmbd, mu = exact_material_torch(x, case_config)
    trace = exx + eyy
    sxx = lmbd * trace + 2.0 * mu * exx
    syy = lmbd * trace + 2.0 * mu * eyy
    sxy = 2.0 * mu * exy
    return torch.cat((ux, uy, sxx, syy, sxy, lmbd, mu), dim=1)


def exact_body_force_torch(x, case_config, load_scale=1.0):
    exact_state = exact_state_torch(x, case_config, load_scale=load_scale)
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


def resolve_observation_split_cache_path(cache_dir, case_name, seed, num_train, num_val, num_eval, tag):
    filename = (
        f"{case_name}_seed{seed}_train{int(num_train)}_val{int(num_val)}_eval{int(num_eval)}_{tag}.npz"
    )
    return os.path.join(cache_dir, filename)


def build_observation_splits(num_train, num_val, num_eval, seed, case_name, cache_dir=None, tag="official_softbc_v1"):
    total = int(num_train) + int(num_val) + int(num_eval)
    if total <= 0:
        raise ValueError("At least one observation point is required.")

    cache_path = None
    if cache_dir:
        ensure_dir(cache_dir)
        cache_path = resolve_observation_split_cache_path(
            cache_dir=cache_dir,
            case_name=case_name,
            seed=seed,
            num_train=num_train,
            num_val=num_val,
            num_eval=num_eval,
            tag=tag,
        )
        if os.path.exists(cache_path):
            cached = np.load(cache_path)
            return {
                "train": np.asarray(cached["train"], dtype=float),
                "val": np.asarray(cached["val"], dtype=float),
                "eval": np.asarray(cached["eval"], dtype=float),
                "cache_path": cache_path,
            }

    points = build_observation_points(total, seed)
    train_end = int(num_train)
    val_end = train_end + int(num_val)
    splits = {
        "train": np.asarray(points[:train_end], dtype=float),
        "val": np.asarray(points[train_end:val_end], dtype=float),
        "eval": np.asarray(points[val_end:], dtype=float),
        "cache_path": cache_path,
    }
    if cache_path is not None:
        np.savez(cache_path, train=splits["train"], val=splits["val"], eval=splits["eval"])
    return splits


def build_boundary_points(num_boundary):
    num_boundary = int(num_boundary)
    if num_boundary <= 0:
        return np.zeros((0, 2), dtype=float)
    perimeter = np.linspace(0.0, 4.0, num_boundary, endpoint=False, dtype=float)
    points = np.zeros((num_boundary, 2), dtype=float)
    for idx, value in enumerate(perimeter):
        if value < 1.0:
            points[idx] = [value, 0.0]
        elif value < 2.0:
            points[idx] = [1.0, value - 1.0]
        elif value < 3.0:
            points[idx] = [3.0 - value, 1.0]
        else:
            points[idx] = [0.0, 4.0 - value]
    return points


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


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, activation="tanh"):
        super().__init__()
        activation_cls = activation_factory(activation)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = activation_cls()
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        residual = x
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        return self.activation(x + residual)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, activation="tanh"):
        super().__init__()
        if not hidden_layers:
            raise ValueError("ResidualMLP requires at least one hidden layer.")
        activation_cls = activation_factory(activation)
        self.input_layer = nn.Linear(input_dim, hidden_layers[0])
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        self.input_activation = activation_cls()
        stages = []
        current_dim = hidden_layers[0]
        for hidden_dim in hidden_layers:
            if current_dim != hidden_dim:
                projection = nn.Linear(current_dim, hidden_dim)
                nn.init.xavier_uniform_(projection.weight)
                nn.init.zeros_(projection.bias)
                stages.append(nn.Sequential(projection, activation_cls()))
                current_dim = hidden_dim
            stages.append(ResidualBlock(current_dim, activation=activation))
        self.stages = nn.ModuleList(stages)
        self.output_layer = nn.Linear(current_dim, output_dim)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        x = self.input_activation(self.input_layer(x))
        for stage in self.stages:
            x = stage(x)
        return self.output_layer(x)


def build_backbone(input_dim, hidden_layers, output_dim, activation="tanh", backbone_type="mlp"):
    backbone_type = backbone_type.lower()
    if backbone_type == "mlp":
        return SimpleMLP(input_dim, hidden_layers, output_dim, activation=activation)
    if backbone_type == "resmlp":
        return ResidualMLP(input_dim, hidden_layers, output_dim, activation=activation)
    raise ValueError(f"Unsupported backbone_type: {backbone_type}")


class VanillaMaterialFieldNet(dde.nn.pytorch.nn.NN):
    def __init__(self, hidden_layers, activation="tanh", num_frequencies=0, backbone_type="mlp"):
        super().__init__()
        self.features = FourierFeatureMap(num_frequencies)
        self.backbone = build_backbone(
            input_dim=self.features.output_dim,
            hidden_layers=hidden_layers,
            output_dim=len(FIELD_NAMES),
            activation=activation,
            backbone_type=backbone_type,
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


class InterfaceAwareMaterialNetV2(dde.nn.pytorch.nn.NN):
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
        backbone_type="mlp",
    ):
        super().__init__()
        self.features = FourierFeatureMap(num_frequencies)
        self.num_regions = num_regions_for_case(case_name)
        self.case_name = case_name
        self.lambda_floor = float(lambda_floor)
        self.mu_floor = float(mu_floor)
        self.interface_sharpness = float(interface_sharpness)
        feature_dim = self.features.output_dim
        self.state_net = build_backbone(
            input_dim=feature_dim,
            hidden_layers=state_hidden_layers,
            output_dim=2,
            activation=activation,
            backbone_type=backbone_type,
        )
        self.interface_net = build_backbone(
            input_dim=feature_dim,
            hidden_layers=interface_hidden_layers,
            output_dim=self.num_regions,
            activation=activation,
            backbone_type=backbone_type,
        )
        lambda_init = torch.linspace(-0.35, 0.35, steps=self.num_regions + 1, dtype=torch.float32)
        mu_init = torch.linspace(-0.2, 0.2, steps=self.num_regions + 1, dtype=torch.float32)
        self.raw_lambda_params = nn.Parameter(lambda_init.clone())
        self.raw_mu_params = nn.Parameter(mu_init.clone())

    def _material_from_features(self, features):
        region_logits = self.interface_sharpness * self.interface_net(features)
        if self.num_regions == 1:
            region_probability = torch.sigmoid(region_logits)
            class_probs = torch.cat((1.0 - region_probability, region_probability), dim=1)
            interface_indicator = region_logits
        else:
            background_logit = torch.zeros((features.shape[0], 1), dtype=region_logits.dtype, device=region_logits.device)
            class_logits = torch.cat((background_logit, region_logits), dim=1)
            class_probs = torch.softmax(class_logits, dim=1)
            interface_indicator = region_logits[:, :1]
        lambda_regions = self.lambda_floor + F.softplus(self.raw_lambda_params)
        mu_regions = self.mu_floor + F.softplus(self.raw_mu_params)
        lmbd = torch.sum(class_probs * lambda_regions.unsqueeze(0), dim=1, keepdim=True)
        mu = torch.sum(class_probs * mu_regions.unsqueeze(0), dim=1, keepdim=True)
        return lmbd, mu, interface_indicator, class_probs

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(inputs)
        features = self.features(x)
        state_outputs = self.state_net(features)
        lmbd, mu, _, _ = self._material_from_features(features)
        outputs = torch.cat((state_outputs, lmbd, mu), dim=1)
        return outputs

    def predict_material_diagnostics(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(inputs)
        features = self.features(x)
        lmbd, mu, interface_indicator, class_probs = self._material_from_features(features)
        return {
            "lambda": lmbd,
            "mu": mu,
            "interface_indicator": interface_indicator,
            "class_probs": class_probs,
            "lambda_regions": self.lambda_floor + F.softplus(self.raw_lambda_params),
            "mu_regions": self.mu_floor + F.softplus(self.raw_mu_params),
        }


def bounded_sigmoid(raw_value, lower, upper):
    return lower + (upper - lower) * torch.sigmoid(raw_value)


class GeometryAwareMaterialNet(dde.nn.pytorch.nn.NN):
    def __init__(
        self,
        case_name,
        state_hidden_layers,
        activation="tanh",
        num_frequencies=0,
        lambda_floor=0.1,
        mu_floor=0.1,
        interface_sharpness=40.0,
        backbone_type="mlp",
    ):
        super().__init__()
        self.features = FourierFeatureMap(num_frequencies)
        self.case_name = case_name
        self.num_regions = num_regions_for_case(case_name)
        self.lambda_floor = float(lambda_floor)
        self.mu_floor = float(mu_floor)
        self.interface_sharpness = float(interface_sharpness)
        self.state_net = build_backbone(
            input_dim=self.features.output_dim,
            hidden_layers=state_hidden_layers,
            output_dim=2,
            activation=activation,
            backbone_type=backbone_type,
        )
        lambda_init = torch.linspace(-0.35, 0.35, steps=self.num_regions + 1, dtype=torch.float32)
        mu_init = torch.linspace(-0.2, 0.2, steps=self.num_regions + 1, dtype=torch.float32)
        self.raw_lambda_params = nn.Parameter(lambda_init.clone())
        self.raw_mu_params = nn.Parameter(mu_init.clone())

        if case_name == "layered":
            self.raw_layer_y = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        elif case_name == "single_inclusion":
            self.raw_circle_1_cx = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            self.raw_circle_1_cy = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            self.raw_circle_1_r = nn.Parameter(torch.tensor(-0.2, dtype=torch.float32))
        elif case_name == "double_inclusion":
            self.raw_circle_1_cx = nn.Parameter(torch.tensor(-0.25, dtype=torch.float32))
            self.raw_circle_1_cy = nn.Parameter(torch.tensor(0.15, dtype=torch.float32))
            self.raw_circle_1_r = nn.Parameter(torch.tensor(-0.2, dtype=torch.float32))
            self.raw_circle_2_cx = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
            self.raw_circle_2_cy = nn.Parameter(torch.tensor(-0.15, dtype=torch.float32))
            self.raw_circle_2_r = nn.Parameter(torch.tensor(-0.35, dtype=torch.float32))
        else:
            raise ValueError(f"Unsupported case: {case_name}")

    def _geometry_logits(self, inputs):
        x = inputs[:, 0:1]
        y = inputs[:, 1:2]
        if self.case_name == "layered":
            layer_y = bounded_sigmoid(self.raw_layer_y, 0.15, 0.85)
            indicator = y - layer_y
            region_probability = torch.sigmoid(self.interface_sharpness * indicator)
            class_probs = torch.cat((1.0 - region_probability, region_probability), dim=1)
            geometry = {"layer_y": layer_y}
            return indicator, class_probs, geometry

        if self.case_name == "single_inclusion":
            cx = bounded_sigmoid(self.raw_circle_1_cx, 0.2, 0.8)
            cy = bounded_sigmoid(self.raw_circle_1_cy, 0.2, 0.8)
            radius = bounded_sigmoid(self.raw_circle_1_r, 0.08, 0.35)
            signed_distance = radius**2 - ((x - cx) ** 2 + (y - cy) ** 2)
            region_probability = torch.sigmoid(self.interface_sharpness * signed_distance)
            class_probs = torch.cat((1.0 - region_probability, region_probability), dim=1)
            geometry = {"circle_1_cx": cx, "circle_1_cy": cy, "circle_1_r": radius}
            return signed_distance, class_probs, geometry

        cx1 = bounded_sigmoid(self.raw_circle_1_cx, 0.15, 0.55)
        cy1 = bounded_sigmoid(self.raw_circle_1_cy, 0.45, 0.85)
        r1 = bounded_sigmoid(self.raw_circle_1_r, 0.08, 0.24)
        cx2 = bounded_sigmoid(self.raw_circle_2_cx, 0.45, 0.85)
        cy2 = bounded_sigmoid(self.raw_circle_2_cy, 0.15, 0.55)
        r2 = bounded_sigmoid(self.raw_circle_2_r, 0.06, 0.18)
        logit_1 = self.interface_sharpness * (r1**2 - ((x - cx1) ** 2 + (y - cy1) ** 2))
        logit_2 = self.interface_sharpness * (r2**2 - ((x - cx2) ** 2 + (y - cy2) ** 2))
        background_logit = torch.zeros_like(logit_1)
        class_probs = torch.softmax(torch.cat((background_logit, logit_1, logit_2), dim=1), dim=1)
        geometry = {
            "circle_1_cx": cx1,
            "circle_1_cy": cy1,
            "circle_1_r": r1,
            "circle_2_cx": cx2,
            "circle_2_cy": cy2,
            "circle_2_r": r2,
        }
        return logit_1, class_probs, geometry

    def _material_from_inputs(self, inputs):
        interface_indicator, class_probs, geometry = self._geometry_logits(inputs)
        lambda_regions = self.lambda_floor + F.softplus(self.raw_lambda_params)
        mu_regions = self.mu_floor + F.softplus(self.raw_mu_params)
        lmbd = torch.sum(class_probs * lambda_regions.unsqueeze(0), dim=1, keepdim=True)
        mu = torch.sum(class_probs * mu_regions.unsqueeze(0), dim=1, keepdim=True)
        return lmbd, mu, interface_indicator, class_probs, geometry

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(inputs)
        features = self.features(x)
        state_outputs = self.state_net(features)
        lmbd, mu, _, _, _ = self._material_from_inputs(inputs)
        outputs = torch.cat((state_outputs, lmbd, mu), dim=1)
        return outputs

    def predict_material_diagnostics(self, inputs):
        lmbd, mu, interface_indicator, class_probs, geometry = self._material_from_inputs(inputs)
        diagnostics = {
            "lambda": lmbd,
            "mu": mu,
            "interface_indicator": interface_indicator,
            "class_probs": class_probs,
            "lambda_regions": self.lambda_floor + F.softplus(self.raw_lambda_params),
            "mu_regions": self.mu_floor + F.softplus(self.raw_mu_params),
        }
        diagnostics.update(geometry)
        return diagnostics


class SmoothGeometryAwareMaterialNetV3(dde.nn.pytorch.nn.NN):
    def __init__(
        self,
        num_loads,
        state_hidden_layers,
        geometry_hidden_layers,
        activation="tanh",
        num_frequencies=0,
        lambda_floor=0.1,
        mu_floor=0.1,
        k_floor=0.2,
        interface_sharpness=12.0,
        parameterization="bulkmu",
        backbone_type="mlp",
    ):
        super().__init__()
        self.num_loads = int(num_loads)
        self.num_regions = 1
        self.lambda_floor = float(lambda_floor)
        self.mu_floor = float(mu_floor)
        self.k_floor = float(k_floor)
        self.interface_sharpness = float(interface_sharpness)
        self.parameterization = str(parameterization)
        self.features = FourierFeatureMap(num_frequencies)
        feature_dim = self.features.output_dim
        self.state_net = build_backbone(
            input_dim=feature_dim,
            hidden_layers=state_hidden_layers,
            output_dim=2 * self.num_loads,
            activation=activation,
            backbone_type=backbone_type,
        )
        self.geometry_net = build_backbone(
            input_dim=feature_dim,
            hidden_layers=geometry_hidden_layers,
            output_dim=1,
            activation=activation,
            backbone_type=backbone_type,
        )
        if self.parameterization == "bulkmu":
            self.raw_k_params = nn.Parameter(torch.tensor([-0.15, 0.35], dtype=torch.float32))
            self.raw_mu_params = nn.Parameter(torch.tensor([-0.05, 0.2], dtype=torch.float32))
        elif self.parameterization == "lamemu":
            self.raw_lambda_params = nn.Parameter(torch.tensor([-0.15, 0.35], dtype=torch.float32))
            self.raw_mu_params = nn.Parameter(torch.tensor([-0.05, 0.2], dtype=torch.float32))
        else:
            raise ValueError(f"Unsupported material parameterization: {self.parameterization}")

    def _material_from_features(self, features):
        phase_logits = self.geometry_net(features)
        phase_probability = torch.sigmoid(self.interface_sharpness * phase_logits)
        class_probs = torch.cat((1.0 - phase_probability, phase_probability), dim=1)
        if self.parameterization == "bulkmu":
            bulk_regions = self.k_floor + F.softplus(self.raw_k_params)
            mu_regions = self.mu_floor + F.softplus(self.raw_mu_params)
            bulk = torch.sum(class_probs * bulk_regions.unsqueeze(0), dim=1, keepdim=True)
            mu = torch.sum(class_probs * mu_regions.unsqueeze(0), dim=1, keepdim=True)
            lmbd = torch.clamp(bulk - (2.0 / 3.0) * mu, min=self.lambda_floor)
            lambda_regions = torch.clamp(bulk_regions - (2.0 / 3.0) * mu_regions, min=self.lambda_floor)
            aux = {
                "bulk": bulk,
                "bulk_regions": bulk_regions,
                "lambda_regions": lambda_regions,
                "mu_regions": mu_regions,
            }
        else:
            lambda_regions = self.lambda_floor + F.softplus(self.raw_lambda_params)
            mu_regions = self.mu_floor + F.softplus(self.raw_mu_params)
            lmbd = torch.sum(class_probs * lambda_regions.unsqueeze(0), dim=1, keepdim=True)
            mu = torch.sum(class_probs * mu_regions.unsqueeze(0), dim=1, keepdim=True)
            aux = {
                "lambda_regions": lambda_regions,
                "mu_regions": mu_regions,
            }
        return lmbd, mu, phase_logits, class_probs, aux

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(inputs)
        features = self.features(x)
        state_outputs = self.state_net(features)
        lmbd, mu, _, _, _ = self._material_from_features(features)
        return torch.cat((state_outputs, lmbd, mu), dim=1)

    def predict_material_diagnostics(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(inputs)
        features = self.features(x)
        lmbd, mu, phase_logits, class_probs, aux = self._material_from_features(features)
        diagnostics = {
            "lambda": lmbd,
            "mu": mu,
            "interface_indicator": phase_logits,
            "class_probs": class_probs,
        }
        diagnostics.update(aux)
        return diagnostics


def make_output_transform(lambda_floor, mu_floor, method=None):
    def output_transform(inputs, outputs):
        ux = outputs[:, 0:1]
        uy = outputs[:, 1:2]
        sxx = outputs[:, 2:3]
        syy = outputs[:, 3:4]
        sxy = outputs[:, 4:5]
        lmbd = lambda_floor + F.softplus(outputs[:, 5:6])
        mu = mu_floor + F.softplus(outputs[:, 6:7])
        return torch.cat((ux, uy, sxx, syy, sxy, lmbd, mu), dim=1)

    return output_transform


def count_trainable_parameters(net):
    return int(sum(parameter.numel() for parameter in net.parameters() if parameter.requires_grad))


def is_compact_material_method(method):
    return method in {"iaminn_v2", "geoiaminn", "geoiaminn_v3"}


def build_network(args):
    if args.method == "iaminn_v2":
        net = InterfaceAwareMaterialNetV2(
            case_name=args.case,
            state_hidden_layers=parse_hidden_layers(args.state_layers),
            interface_hidden_layers=parse_hidden_layers(args.interface_layers),
            activation=args.activation,
            num_frequencies=args.num_frequencies,
            lambda_floor=args.lambda_floor,
            mu_floor=args.mu_floor,
            interface_sharpness=args.interface_sharpness,
            backbone_type=args.backbone_type,
        )
    elif args.method == "geoiaminn":
        net = GeometryAwareMaterialNet(
            case_name=args.case,
            state_hidden_layers=parse_hidden_layers(args.state_layers),
            activation=args.activation,
            num_frequencies=args.num_frequencies,
            lambda_floor=args.lambda_floor,
            mu_floor=args.mu_floor,
            interface_sharpness=args.interface_sharpness,
            backbone_type=args.backbone_type,
        )
    elif args.method == "geoiaminn_v3":
        net = SmoothGeometryAwareMaterialNetV3(
            num_loads=compact_num_loads(args),
            state_hidden_layers=parse_hidden_layers(args.state_layers),
            geometry_hidden_layers=parse_hidden_layers(args.geometry_layers),
            activation=args.activation,
            num_frequencies=args.num_frequencies,
            lambda_floor=args.lambda_floor,
            mu_floor=args.mu_floor,
            k_floor=args.k_floor,
            interface_sharpness=args.interface_sharpness,
            parameterization=args.material_parameterization,
            backbone_type=args.backbone_type,
        )
    else:
        net = VanillaMaterialFieldNet(
            hidden_layers=parse_hidden_layers(args.hidden_layers),
            activation=args.activation,
            num_frequencies=args.num_frequencies,
            backbone_type=args.backbone_type,
        )
        net.apply_output_transform(make_output_transform(args.lambda_floor, args.mu_floor, args.method))
    return net



def build_pde(case_config, reg_weight, method=None, load_scales=None):
    def pde(x, y):
        if is_compact_material_method(method):
            compact_load_scales = parse_load_scales(load_scales if load_scales is not None else "1.0")
            lambda_idx = 2 * len(compact_load_scales)
            mu_idx = lambda_idx + 1
            lmbd = y[:, lambda_idx:lambda_idx + 1]
            mu = y[:, mu_idx:mu_idx + 1]
            residuals = []
            for load_index, load_scale in enumerate(compact_load_scales):
                ux = y[:, 2 * load_index : 2 * load_index + 1]
                uy = y[:, 2 * load_index + 1 : 2 * load_index + 2]
                ux_x = dde.grad.jacobian(ux, x, i=0, j=0)
                ux_y = dde.grad.jacobian(ux, x, i=0, j=1)
                uy_x = dde.grad.jacobian(uy, x, i=0, j=0)
                uy_y = dde.grad.jacobian(uy, x, i=0, j=1)
                exx = ux_x
                eyy = uy_y
                exy = 0.5 * (ux_y + uy_x)
                constitutive_sxx = lmbd * (exx + eyy) + 2.0 * mu * exx
                constitutive_syy = lmbd * (exx + eyy) + 2.0 * mu * eyy
                constitutive_sxy = 2.0 * mu * exy
                sxx_x = dde.grad.jacobian(constitutive_sxx, x, i=0, j=0)
                syy_y = dde.grad.jacobian(constitutive_syy, x, i=0, j=1)
                sxy_x = dde.grad.jacobian(constitutive_sxy, x, i=0, j=0)
                sxy_y = dde.grad.jacobian(constitutive_sxy, x, i=0, j=1)
                fx, fy = exact_body_force_torch(x, case_config, load_scale=load_scale)
                residuals.extend(
                    [
                        sxx_x + sxy_y + fx,
                        sxy_x + syy_y + fy,
                    ]
                )
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

        fx, fy = exact_body_force_torch(x, case_config, load_scale=1.0)
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


def pde_loss_names(reg_weight, method=None, load_scales=None):
    if is_compact_material_method(method):
        compact_load_scales = parse_load_scales(load_scales if load_scales is not None else "1.0")
        names = []
        for load_index in range(len(compact_load_scales)):
            names.extend([f"momentum_x_l{load_index}", f"momentum_y_l{load_index}"])
    else:
        names = ["momentum_x", "momentum_y", "constitutive_xx", "constitutive_yy", "constitutive_xy"]
    if reg_weight > 0.0:
        names.extend(["lambda_x", "lambda_y", "mu_x", "mu_y"])
    return names


def pde_loss_weights(reg_weight, method=None, load_scales=None):
    if is_compact_material_method(method):
        compact_load_scales = parse_load_scales(load_scales if load_scales is not None else "1.0")
        weights = []
        for _ in compact_load_scales:
            weights.extend([1.0, 1.0])
    else:
        weights = [1.0, 1.0, 1.0, 1.0, 1.0]
    if reg_weight > 0.0:
        weights.extend([reg_weight, reg_weight, reg_weight, reg_weight])
    return weights


def resolve_loss_weights(
    args,
    physics_scale=1.0,
    reg_scale=1.0,
    boundary_scale=1.0,
    data_scale=1.0,
):
    weights = []
    for loss_name in pde_loss_names(args.reg_weight, args.method, getattr(args, "load_scales", "1.0")):
        if loss_name in {"momentum_x", "momentum_y", "constitutive_xx", "constitutive_yy", "constitutive_xy"}:
            weights.append(float(physics_scale))
        elif loss_name.startswith("momentum_"):
            weights.append(float(physics_scale))
        else:
            weights.append(float(args.reg_weight) * float(reg_scale))
    num_loads = compact_num_loads(args)
    for _ in range(num_loads):
        weights.extend(
            [
                float(args.boundary_weight) * float(boundary_scale),
                float(args.boundary_weight) * float(boundary_scale),
                float(args.data_weight) * float(data_scale),
                float(args.data_weight) * float(data_scale),
            ]
        )
    return weights


def build_observation_payload(points, case_config, noise_level, seed, load_scale=1.0):
    if len(points) == 0:
        return {
            "points": np.zeros((0, 2), dtype=float),
            "clean": np.zeros((0, 2), dtype=float),
            "noisy": np.zeros((0, 2), dtype=float),
            "true_state": np.zeros((0, len(FIELD_NAMES)), dtype=float),
            "load_scale": float(load_scale),
        }
    exact_observation_state = exact_state_numpy(points, case_config, load_scale=load_scale)
    clean_observation = exact_observation_state[:, :2]
    noisy_observation = add_noise(clean_observation, noise_level, seed)
    return {
        "points": points,
        "clean": clean_observation,
        "noisy": noisy_observation,
        "true_state": exact_observation_state,
        "load_scale": float(load_scale),
    }


def build_data(args, case_config):
    geom = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])
    load_scales = parse_load_scales(getattr(args, "load_scales", "1.0"))
    observation_splits = build_observation_splits(
        num_train=args.num_observe,
        num_val=args.num_val_observe,
        num_eval=args.num_eval_observe,
        seed=args.seed + 17,
        case_name=args.case,
        cache_dir=args.observation_cache_dir,
            tag=args.observation_split_tag,
        )
    boundary_points = build_boundary_points(args.num_boundary)
    train_observation_loads = []
    val_observation_loads = []
    eval_observation_loads = []
    boundary_observation_loads = []
    for load_index, load_scale in enumerate(load_scales):
        train_observation_loads.append(
            build_observation_payload(
                observation_splits["train"],
                case_config,
                args.noise_level,
                args.seed + 123 + 1000 * load_index,
                load_scale=load_scale,
            )
        )
        val_observation_loads.append(
            build_observation_payload(
                observation_splits["val"],
                case_config,
                args.noise_level,
                args.seed + 223 + 1000 * load_index,
                load_scale=load_scale,
            )
        )
        eval_observation_loads.append(
            build_observation_payload(
                observation_splits["eval"],
                case_config,
                args.noise_level,
                args.seed + 323 + 1000 * load_index,
                load_scale=load_scale,
            )
        )
        boundary_observation_loads.append(
            build_observation_payload(
                boundary_points,
                case_config,
                0.0,
                args.seed + 423 + 1000 * load_index,
                load_scale=load_scale,
            )
        )

    bcs = []
    num_loads = compact_num_loads(args)
    for load_index in range(num_loads):
        component_ux, component_uy = compact_state_indices(args, load_index)
        boundary_observation = boundary_observation_loads[load_index]
        train_observation = train_observation_loads[load_index]
        bcs.extend(
            [
                dde.icbc.PointSetBC(boundary_observation["points"], boundary_observation["clean"][:, 0:1], component=component_ux),
                dde.icbc.PointSetBC(boundary_observation["points"], boundary_observation["clean"][:, 1:2], component=component_uy),
                dde.icbc.PointSetBC(train_observation["points"], train_observation["noisy"][:, 0:1], component=component_ux),
                dde.icbc.PointSetBC(train_observation["points"], train_observation["noisy"][:, 1:2], component=component_uy),
            ]
        )
    data = dde.data.PDE(
        geom,
        build_pde(case_config, args.reg_weight, args.method, load_scales=load_scales),
        bcs,
        num_domain=args.num_domain,
        num_boundary=0,
        num_test=args.num_test,
        train_distribution="pseudo",
    )
    metadata = {
        "boundary_observation": boundary_observation_loads[0],
        "train_observation": train_observation_loads[0],
        "val_observation": val_observation_loads[0],
        "eval_observation": eval_observation_loads[0],
        "boundary_observation_loads": boundary_observation_loads,
        "train_observation_loads": train_observation_loads,
        "val_observation_loads": val_observation_loads,
        "eval_observation_loads": eval_observation_loads,
        "load_scales": load_scales,
        "observation_cache_path": observation_splits.get("cache_path"),
    }
    return geom, data, metadata


def resolve_run_name(args):
    if args.run_name:
        return args.run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"{args.case}_bc{args.num_boundary}_obs{args.num_observe}_val{args.num_val_observe}_eval{args.num_eval_observe}_noise{args.noise_level:.3f}_"
        f"seed{args.seed}_iter{args.iterations}_{timestamp}"
    )


def default_experiment_group(method):
    mapping = {
        "pinn": "PINN-baseline",
        "iaminn_v2": "IAMINN-v2",
        "geoiaminn": "GeoIAMINN",
        "geoiaminn_v3": "GeoIAMINN-v3",
    }
    return mapping.get(method, method.upper())


def resolve_save_dir(args):
    run_name = resolve_run_name(args)
    group_name = args.experiment_group or default_experiment_group(args.method)
    return os.path.join(args.exp_root, group_name, args.case, run_name)


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


def save_text(path, text):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def save_json_as_text(path, payload):
    save_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def save_array_txt(path, array, header):
    ensure_dir(os.path.dirname(path))
    np.savetxt(path, np.asarray(array), fmt="%.10e", header=header, comments="")


def save_metrics_text(path, payload):
    lines = []
    for key, value in payload.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for inner_key, inner_value in value.items():
                lines.append(f"  {inner_key}: {inner_value}")
        else:
            lines.append(f"{key}: {value}")
    save_text(path, "\n".join(lines) + "\n")


def save_observation_split_files(save_dir, split_name, payload):
    array_payload = {
        "points": payload["points"],
        "clean": payload["clean"],
        "noisy": payload["noisy"],
        "true_state": payload["true_state"],
    }
    np.savez(_get_save_path(save_dir, "npz", f"{split_name}_set_data.npz"), **array_payload)
    combined = np.concatenate(
        [
            payload["points"],
            payload["clean"],
            payload["noisy"],
            payload["true_state"],
        ],
        axis=1,
    )
    header = "x y clean_ux clean_uy noisy_ux noisy_uy true_ux true_uy true_sxx true_syy true_sxy true_lambda true_mu"
    save_array_txt(_get_save_path(save_dir, "txt", f"{split_name}_set_data.txt"), combined, header)
    metadata = {
        "split": split_name,
        "num_points": int(len(payload["points"])),
        "field_names": FIELD_NAMES,
    }
    save_json(_get_save_path(save_dir, "json", f"{split_name}_set_metadata.json"), metadata)


def save_loss_history_dat(losshistory, save_dir, filename="loss_history.dat"):
    if losshistory is None:
        return
    steps = np.asarray(getattr(losshistory, "steps", []), dtype=float)
    if steps.size == 0:
        return
    loss_train = np.asarray(getattr(losshistory, "loss_train", []), dtype=float)
    loss_test = np.asarray(getattr(losshistory, "loss_test", []), dtype=float)
    width = max(loss_train.shape[1] if loss_train.ndim == 2 else 0, loss_test.shape[1] if loss_test.ndim == 2 else 0)
    train_pad = np.full((steps.shape[0], width), np.nan, dtype=float)
    test_pad = np.full((steps.shape[0], width), np.nan, dtype=float)
    if loss_train.ndim == 2 and loss_train.shape[1] > 0:
        train_pad[:, : loss_train.shape[1]] = loss_train
    if loss_test.ndim == 2 and loss_test.shape[1] > 0:
        test_pad[:, : loss_test.shape[1]] = loss_test
    table = np.concatenate([steps[:, None], train_pad, test_pad], axis=1)
    train_headers = [f"train_loss_{idx}" for idx in range(width)]
    test_headers = [f"test_loss_{idx}" for idx in range(width)]
    header = "step " + " ".join(train_headers + test_headers)
    save_array_txt(_get_save_path(save_dir, "dat", filename), table, header)

def save_case_and_sampling_figure(
    save_dir,
    case_config,
    domain_points,
    boundary_points,
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
    if len(boundary_points) > 0:
        axes[2].scatter(
            boundary_points[:, 0],
            boundary_points[:, 1],
            s=22,
            alpha=0.9,
            color="#111827",
            marker="x",
            label="Boundary points",
        )
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
    num_loads = compact_num_loads(args)
    bc_loss_names = []
    for load_index in range(num_loads):
        bc_loss_names.extend(
            [
                f"boundary_ux_l{load_index}",
                f"boundary_uy_l{load_index}",
                f"obs_ux_l{load_index}",
                f"obs_uy_l{load_index}",
            ]
        )
    config_payload = {
        "args": {key: value for key, value in vars(args).items() if not key.startswith("_")},
        "case": asdict(case_config),
        "parameter_count": count_trainable_parameters(net),
        "field_names": FIELD_NAMES,
        "pde_loss_names": pde_loss_names(args.reg_weight, args.method, getattr(args, "load_scales", "1.0")),
        "bc_loss_names": bc_loss_names,
        "selection_metric": "validation_observation_mse",
        "observation_cache_path": metadata.get("observation_cache_path"),
    }
    save_json(_get_save_path(save_dir, "json", "run_config.json"), config_payload)
    save_json_as_text(os.path.join(save_dir, "配置.txt"), config_payload)
    np.savez(
        _get_save_path(save_dir, "npz", "observation_data.npz"),
        boundary_points=metadata["boundary_observation"]["points"],
        boundary_observation_clean=metadata["boundary_observation"]["clean"],
        boundary_observation_true_state=metadata["boundary_observation"]["true_state"],
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
    save_observation_split_files(save_dir, "boundary", metadata["boundary_observation"])
    save_observation_split_files(save_dir, "train", metadata["train_observation"])
    save_observation_split_files(save_dir, "validation", metadata["val_observation"])
    save_observation_split_files(save_dir, "evaluation", metadata["eval_observation"])

    domain_points = getattr(args, "_domain_points_for_plot", None)
    if domain_points is not None:
        np.savez(_get_save_path(save_dir, "npz", "domain_points.npz"), domain_points=domain_points)
        save_array_txt(_get_save_path(save_dir, "txt", "domain_points.txt"), domain_points, "x y")
        save_case_and_sampling_figure(
            save_dir=save_dir,
            case_config=case_config,
            domain_points=domain_points,
            boundary_points=metadata["boundary_observation"]["points"],
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
            num_pde_losses=len(pde_loss_names(args.reg_weight, args.method, getattr(args, "load_scales", "1.0"))),
            num_bc_losses=len(bc_loss_names),
            pde_label="Physics Loss",
            bc_label="Boundary + Observation Loss",
            bc_loss_names=bc_loss_names,
            data_loss_prefix="obs_",
        )
        plot_all_loss_components(
            losshistory,
            save_dir,
            filename="loss_components.png",
            num_pde_losses=len(pde_loss_names(args.reg_weight, args.method, getattr(args, "load_scales", "1.0"))),
            num_bc_losses=len(bc_loss_names),
            pde_loss_names=pde_loss_names(args.reg_weight, args.method, getattr(args, "load_scales", "1.0")),
            bc_loss_names=bc_loss_names,
        )
        save_loss_history_json(losshistory, save_dir, filename="loss_history.json")
        save_best_test_loss_json(losshistory, save_dir, filename="best_test_loss.json")
        save_loss_history_dat(losshistory, save_dir)

    if args.method == "iaminn_v2" and hasattr(net, "predict_material_diagnostics"):
        with torch.no_grad():
            lambda_regions = net.lambda_floor + F.softplus(net.raw_lambda_params)
            mu_regions = net.mu_floor + F.softplus(net.raw_mu_params)
        region_payload = {
            "lambda_regions": [float(v) for v in lambda_regions.detach().cpu().numpy().tolist()],
            "mu_regions": [float(v) for v in mu_regions.detach().cpu().numpy().tolist()],
            "num_regions": int(net.num_regions),
        }
        save_json(_get_save_path(save_dir, "json", "material_region_parameters.json"), region_payload)
        save_metrics_text(_get_save_path(save_dir, "txt", "material_region_parameters.txt"), region_payload)
    elif args.method in {"geoiaminn", "geoiaminn_v3"} and hasattr(net, "predict_material_diagnostics"):
        device = next(net.parameters()).device
        with torch.no_grad():
            diagnostics = net.predict_material_diagnostics(torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device))
        region_payload = {
            "lambda_regions": [float(v) for v in diagnostics["lambda_regions"].detach().cpu().numpy().tolist()],
            "mu_regions": [float(v) for v in diagnostics["mu_regions"].detach().cpu().numpy().tolist()],
            "num_regions": int(net.num_regions),
        }
        for key, value in diagnostics.items():
            if key in {"lambda", "mu", "interface_indicator", "class_probs", "lambda_regions", "mu_regions"}:
                continue
            if torch.is_tensor(value) and value.numel() == 1:
                region_payload[key] = float(value.detach().cpu().item())
        save_json(_get_save_path(save_dir, "json", "material_region_parameters.json"), region_payload)
        save_metrics_text(_get_save_path(save_dir, "txt", "material_region_parameters.txt"), region_payload)


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


def _manual_fourier_features(x, num_frequencies):
    num_frequencies = int(num_frequencies)
    if num_frequencies <= 0:
        return x
    features = [x]
    freq_bands = 2.0 ** torch.arange(num_frequencies, dtype=x.dtype, device=x.device)
    for freq in freq_bands:
        features.append(torch.sin(2.0 * PI * freq * x))
        features.append(torch.cos(2.0 * PI * freq * x))
    return torch.cat(features, dim=1)


def _forward_compact_raw_tensor(net, x, args):
    if getattr(args, "method", "") == "geoiaminn_v3" and hasattr(net, "state_net") and hasattr(net, "features"):
        state_inputs = x
        if getattr(net, "_input_transform", None) is not None:
            state_inputs = net._input_transform(x)
        num_frequencies = getattr(getattr(net, "features", None), "num_frequencies", 0)
        features = _manual_fourier_features(state_inputs, num_frequencies)
        state_outputs = net.state_net(features)
        if hasattr(net, "_material_from_features"):
            lmbd, mu, _, _, _ = net._material_from_features(features)
            return torch.cat((state_outputs, lmbd, mu), dim=1)
        if hasattr(net, "_material_from_inputs"):
            lmbd, mu, _, _, _ = net._material_from_inputs(x)
            return torch.cat((state_outputs, lmbd, mu), dim=1)
    return net(x)


def _predict_compact_raw(model, points, args, batch_size=4096):
    net = model.net
    device = next(net.parameters()).device
    outputs = []
    net.eval()
    for start in range(0, len(points), batch_size):
        batch_points = points[start : start + batch_size]
        x = torch.tensor(batch_points, dtype=torch.float32, device=device)
        with torch.no_grad():
            batch_output = _forward_compact_raw_tensor(net, x, args)
        outputs.append(batch_output.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _predict_compact_full_fields(model, points, args, batch_size=2048, load_index=0):
    net = model.net
    device = next(net.parameters()).device
    predictions = []
    net.eval()
    for start in range(0, len(points), batch_size):
        batch_points = points[start : start + batch_size]
        x = torch.tensor(batch_points, dtype=torch.float32, device=device, requires_grad=True)
        raw = _forward_compact_raw_tensor(net, x, args)
        ux_idx, uy_idx = compact_state_indices(args, load_index)
        lambda_idx, mu_idx = compact_material_indices(args)
        ux = raw[:, ux_idx:ux_idx + 1]
        uy = raw[:, uy_idx:uy_idx + 1]
        lmbd = raw[:, lambda_idx:lambda_idx + 1]
        mu = raw[:, mu_idx:mu_idx + 1]
        ux_grad = torch.autograd.grad(ux, x, grad_outputs=torch.ones_like(ux), create_graph=False, retain_graph=True)[0]
        uy_grad = torch.autograd.grad(uy, x, grad_outputs=torch.ones_like(uy), create_graph=False, retain_graph=False)[0]
        exx = ux_grad[:, 0:1]
        eyy = uy_grad[:, 1:2]
        exy = 0.5 * (ux_grad[:, 1:2] + uy_grad[:, 0:1])
        sxx = lmbd * (exx + eyy) + 2.0 * mu * exx
        syy = lmbd * (exx + eyy) + 2.0 * mu * eyy
        sxy = 2.0 * mu * exy
        predictions.append(torch.cat((ux, uy, sxx, syy, sxy, lmbd, mu), dim=1).detach().cpu().numpy())
    return np.concatenate(predictions, axis=0)


def predict_material_diagnostics(model, points, args, batch_size=4096):
    if not is_compact_material_method(args.method) or not hasattr(model.net, "predict_material_diagnostics"):
        return None
    net = model.net
    device = next(net.parameters()).device
    interface_indicator_batches = []
    class_prob_batches = []
    net.eval()
    for start in range(0, len(points), batch_size):
        batch_points = points[start : start + batch_size]
        x = torch.tensor(batch_points, dtype=torch.float32, device=device)
        with torch.no_grad():
            diagnostics = net.predict_material_diagnostics(x)
        interface_indicator_batches.append(diagnostics["interface_indicator"].detach().cpu().numpy())
        class_prob_batches.append(diagnostics["class_probs"].detach().cpu().numpy())
    with torch.no_grad():
        summary = net.predict_material_diagnostics(torch.tensor(points[: min(len(points), 1)], dtype=torch.float32, device=device))
    return {
        "interface_indicator": np.concatenate(interface_indicator_batches, axis=0),
        "class_probs": np.concatenate(class_prob_batches, axis=0),
        "lambda_regions": summary["lambda_regions"].detach().cpu().numpy(),
        "mu_regions": summary["mu_regions"].detach().cpu().numpy(),
    }


def predict_full_fields(model, points, args, batch_size=4096, load_index=0):
    if is_compact_material_method(args.method):
        return _predict_compact_full_fields(model, points, args, batch_size=batch_size, load_index=load_index)
    return np.asarray(model.predict(points))


def compute_observation_mse(model, args, observation_points, observation_truth):
    if isinstance(observation_points, list):
        mses = []
        for load_index, (points, truth) in enumerate(zip(observation_points, observation_truth)):
            if len(points) == 0:
                continue
            observation_prediction = _predict_compact_raw(model, points, args, batch_size=4096)[:, compact_state_indices(args, load_index)[0]:compact_state_indices(args, load_index)[0] + 2] if is_compact_material_method(args.method) else predict_full_fields(model, points, args, load_index=load_index)[:, :2]
            mses.append(float(np.mean((observation_prediction - truth) ** 2)))
        return float(np.mean(mses)) if mses else float("nan")
    if len(observation_points) == 0:
        return float("nan")
    observation_prediction = _predict_compact_raw(model, observation_points, args, batch_size=4096)[:, compact_state_indices(args, 0)[0]:compact_state_indices(args, 0)[0] + 2] if is_compact_material_method(args.method) else predict_full_fields(model, observation_points, args)[:, :2]
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


def plot_observation_fit(save_dir, observation_truth, observation_prediction, filename="observation_fit.png"):
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
    plt.savefig(_get_save_path(save_dir, "png", filename), dpi=300)
    plt.close(fig)


def plot_interface_diagnostics(save_dir, xx, yy, interface_indicator_grid, class_probability_grid, split_name):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    for axis, grid, title in zip(
        axes,
        [interface_indicator_grid, class_probability_grid],
        ["Interface indicator", "Region probability"],
    ):
        image = axis.contourf(xx, yy, grid, levels=100, cmap="coolwarm")
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_aspect("equal")
        plt.colorbar(image, ax=axis)
    plt.tight_layout()
    plt.savefig(_get_save_path(save_dir, "png", f"{split_name}_interface_diagnostics.png"), dpi=300)
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
    primary_load_index = int(getattr(args, "primary_load_index", 0))
    load_scales = parse_load_scales(getattr(args, "load_scales", "1.0"))
    primary_scale = load_scales[min(primary_load_index, len(load_scales) - 1)]
    truth = exact_state_numpy(eval_points, case_config, load_scale=primary_scale)
    prediction = predict_full_fields(model, eval_points, args, load_index=primary_load_index)
    residual = split_residual_prediction(
        model.predict(
            eval_points,
            operator=build_pde(case_config, args.reg_weight, args.method, load_scales=getattr(args, "load_scales", "1.0")),
        )
    )
    observation_prediction = predict_full_fields(model, observation_points, args, load_index=primary_load_index)[:, :2]
    material_diagnostics = predict_material_diagnostics(model, eval_points, args)

    metrics = compute_metrics(
        prediction=prediction,
        truth=truth,
        observation_prediction=observation_prediction,
        observation_truth=observation_truth_clean,
        pde_residual=residual,
    )

    if save_artifacts:
        prediction_prefix = f"{observation_split_name}_predictions"
        save_prediction_data(
            eval_points,
            prediction,
            truth,
            test_delta=None,
            save_dir=save_dir,
            prefix=prediction_prefix,
            field_names=FIELD_NAMES,
        )
        np.savez(
            _get_save_path(save_dir, "npz", f"{observation_split_name}_grid.npz"),
            points=eval_points,
            prediction=prediction,
            truth=truth,
            xx=xx,
            yy=yy,
            pde_residual=residual,
        )
        save_json(_get_save_path(save_dir, "metrics", f"{observation_split_name}_metrics.json"), metrics)
        save_metrics_text(_get_save_path(save_dir, "metrics", f"{observation_split_name}_metrics.txt"), metrics)
        np.savez(
            _get_save_path(save_dir, "npz", f"{observation_split_name}_observation_fit.npz"),
            points=observation_points,
            prediction=observation_prediction,
            truth=observation_truth_clean,
        )
        save_array_txt(
            _get_save_path(save_dir, "txt", f"{observation_split_name}_observation_fit.txt"),
            np.concatenate([observation_points, observation_truth_clean, observation_prediction], axis=1),
            "x y true_ux true_uy pred_ux pred_uy",
        )

        for index, field_name in enumerate(FIELD_NAMES):
            truth_grid = reshape_grid(truth[:, index], args.eval_ny, args.eval_nx)
            pred_grid = reshape_grid(prediction[:, index], args.eval_ny, args.eval_nx)
            plot_field_triplet(save_dir, xx, yy, truth_grid, pred_grid, f"{observation_split_name}_{field_name}")
            if field_name in MATERIAL_FIELD_NAMES:
                plot_material_overlay(save_dir, xx, yy, truth_grid, pred_grid, f"{observation_split_name}_{field_name}")

        if material_diagnostics is not None:
            interface_indicator_grid = reshape_grid(material_diagnostics["interface_indicator"][:, 0], args.eval_ny, args.eval_nx)
            class_probability_grid = reshape_grid(material_diagnostics["class_probs"][:, -1], args.eval_ny, args.eval_nx)
            np.savez(
                _get_save_path(save_dir, "npz", f"{observation_split_name}_interface_diagnostics.npz"),
                interface_indicator=material_diagnostics["interface_indicator"],
                class_probs=material_diagnostics["class_probs"],
                lambda_regions=material_diagnostics["lambda_regions"],
                mu_regions=material_diagnostics["mu_regions"],
            )
            plot_interface_diagnostics(
                save_dir,
                xx,
                yy,
                interface_indicator_grid,
                class_probability_grid,
                observation_split_name,
            )
            save_json(
                _get_save_path(save_dir, "json", f"{observation_split_name}_interface_summary.json"),
                {
                    "lambda_regions": material_diagnostics["lambda_regions"].tolist(),
                    "mu_regions": material_diagnostics["mu_regions"].tolist(),
                    **{
                        key: float(value.detach().cpu().item())
                        for key, value in material_diagnostics.items()
                        if key
                        not in {
                            "lambda",
                            "mu",
                            "interface_indicator",
                            "class_probs",
                            "lambda_regions",
                            "mu_regions",
                        }
                    },
                },
            )
        plot_observation_fit(
            save_dir,
            observation_truth_clean,
            observation_prediction,
            filename=f"{observation_split_name}_observation_fit.png",
        )
    return metrics


def build_model(args, data):
    net = build_network(args)
    model = dde.Model(data, net)
    loss_weights = resolve_loss_weights(args)
    model.compile("adam", lr=args.lr, loss_weights=loss_weights)
    return model, net


def make_callbacks(args, save_dir, metadata):
    num_pde = len(pde_loss_names(args.reg_weight, args.method, getattr(args, "load_scales", "1.0")))
    num_loads = compact_num_loads(args)
    bc_loss_names = []
    for load_index in range(num_loads):
        bc_loss_names.extend(
            [
                f"boundary_ux_l{load_index}",
                f"boundary_uy_l{load_index}",
                f"obs_ux_l{load_index}",
                f"obs_uy_l{load_index}",
            ]
        )
    model_dir = ensure_dir(os.path.join(save_dir, "model"))
    callbacks = [
        LossHistoryCallback(
            save_dir=save_dir,
            period=args.display_every,
            filename="loss_history.png",
            num_pde_losses=num_pde,
            num_bc_losses=len(bc_loss_names),
            pde_loss_names=pde_loss_names(args.reg_weight, args.method, getattr(args, "load_scales", "1.0")),
            bc_loss_names=bc_loss_names,
            pde_label="Physics Loss",
            bc_label="Boundary + Observation Loss",
            save_all_components=True,
        ),
        ValidationObservationCheckpoint(
            filepath=os.path.join(model_dir, "best_model"),
            save_dir=save_dir,
            observation_points=[payload["points"] for payload in metadata.get("val_observation_loads", [metadata["val_observation"]])],
            observation_truth_clean=[payload["noisy"] for payload in metadata.get("val_observation_loads", [metadata["val_observation"]])],
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
        choices=["pinn", "iaminn_v2", "geoiaminn", "geoiaminn_v3"],
        default="pinn",
    )
    parser.add_argument(
        "--case",
        choices=["layered", "single_inclusion", "double_inclusion"],
        default="single_inclusion",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--display_every", type=int, default=1000)
    parser.add_argument("--num_domain", type=int, default=4000)
    parser.add_argument("--num_test", type=int, default=2000)
    parser.add_argument("--num_boundary", type=int, default=400)
    parser.add_argument("--num_observe", type=int, default=500)
    parser.add_argument("--num_val_observe", type=int, default=200)
    parser.add_argument("--num_eval_observe", type=int, default=500)
    parser.add_argument("--noise_level", type=float, default=0.0)
    parser.add_argument("--reg_weight", type=float, default=1e-4)
    parser.add_argument("--data_weight", type=float, default=20.0)
    parser.add_argument("--boundary_weight", type=float, default=20.0)
    parser.add_argument("--lambda_floor", type=float, default=0.1)
    parser.add_argument("--mu_floor", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument("--backbone_type", choices=["mlp", "resmlp"], default="mlp")
    parser.add_argument("--hidden_layers", type=str, default="128,128,128,128")
    parser.add_argument("--state_layers", type=str, default="128,128,128,128")
    parser.add_argument("--geometry_layers", type=str, default="64,64,64")
    parser.add_argument("--interface_layers", type=str, default="128,128,128,128")
    parser.add_argument("--interface_sharpness", type=float, default=10.0)
    parser.add_argument("--num_frequencies", type=int, default=4)
    parser.add_argument("--material_parameterization", choices=["bulkmu", "lamemu"], default="bulkmu")
    parser.add_argument("--k_floor", type=float, default=0.2)
    parser.add_argument("--load_scales", type=str, default="1.0")
    parser.add_argument("--primary_load_index", type=int, default=0)
    parser.add_argument("--observation_split_tag", type=str, default="official_softbc_v1")
    parser.add_argument("--observation_cache_dir", type=str, default=DEFAULT_OBSERVATION_CACHE_DIR)
    parser.add_argument("--exp_root", type=str, default=DEFAULT_EXP_ROOT)
    parser.add_argument("--experiment_group", type=str, default=None)
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
    if args.num_boundary <= 0:
        raise ValueError("num_boundary must be positive when using unified soft boundary constraints.")
    load_scales = parse_load_scales(getattr(args, "load_scales", "1.0"))
    if int(getattr(args, "primary_load_index", 0)) < 0 or int(getattr(args, "primary_load_index", 0)) >= len(load_scales):
        raise ValueError("primary_load_index is out of range for load_scales.")
    set_random_seed(args.seed)
    if args.device_debug:
        print_gpu_info()
    case_config = get_case_config(args.case)
    if args.save_dir is None:
        args.save_dir = resolve_save_dir(args)
    ensure_dir(args.save_dir)
    return case_config
