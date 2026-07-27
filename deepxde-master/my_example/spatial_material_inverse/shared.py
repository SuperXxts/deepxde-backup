import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from types import SimpleNamespace

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
from utils.material_inverse_plot_utils import (
    plot_case_sampling_layout,
    plot_field_triplet,
    plot_interface_diagnostics,
    plot_k_mu_truth_prediction_comparison,
    plot_material_overlay,
    plot_material_probe_evolution,
    plot_observation_fit,
    plot_validation_history,
    reshape_grid,
)
from utils.save_results import (
    plot_all_loss_components,
    plot_and_save_loss_history,
    plot_region_parameter_evolution,
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
    weak_y_bottom: float = 0.30
    weak_y_top: float = 0.40
    amplitude_u: float = 0.12
    amplitude_v: float = -0.08


def get_case_config(case_name):
    if case_name == "layered":
        return CaseConfig(name="layered")
    if case_name == "single_material":
        return CaseConfig(
            name="single_material",
            lambda_ctr_1=0.0,
            mu_ctr_1=0.0,
            interface_width=0.03,
        )
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
    if case_name == "weak_interlayer":
        return CaseConfig(
            name="weak_interlayer",
            lambda_bg=1.8,
            lambda_ctr_1=-0.8,
            mu_bg=1.0,
            mu_ctr_1=-0.55,
            interface_width=0.03,
            weak_y_bottom=0.30,
            weak_y_top=0.40,
        )
    raise ValueError(f"Unsupported case: {case_name}")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def bulk_from_lambda_mu(lmbd, mu):
    return np.asarray(lmbd, dtype=float) + (2.0 / 3.0) * np.asarray(mu, dtype=float)


def primary_material_specs(parameterization):
    parameterization = str(parameterization).lower()
    if parameterization == "bulkmu":
        return [
            ("bulk", "Bulk modulus K"),
            ("mu", "Shear modulus Mu"),
        ]
    return [
        ("lambda", "Lambda"),
        ("mu", "Shear modulus Mu"),
    ]


def build_material_probe_specs(case_config):
    background = {
        "name": "background",
        "point": [0.1, 0.1],
        "lambda": float(case_config.lambda_bg),
        "mu": float(case_config.mu_bg),
    }
    background["bulk"] = float(background["lambda"] + (2.0 / 3.0) * background["mu"])

    if case_config.name == "single_material":
        probe_b = {
            "name": "probe_b",
            "point": [0.8, 0.8],
            "lambda": float(case_config.lambda_bg),
            "mu": float(case_config.mu_bg),
        }
        probe_b["bulk"] = float(probe_b["lambda"] + (2.0 / 3.0) * probe_b["mu"])
        return [background, probe_b]

    if case_config.name == "layered":
        upper = {
            "name": "layer",
            "point": [0.5, min(0.9, float(case_config.layer_y) + 0.2)],
            "lambda": float(case_config.lambda_bg + case_config.lambda_ctr_1),
            "mu": float(case_config.mu_bg + case_config.mu_ctr_1),
        }
        upper["bulk"] = float(upper["lambda"] + (2.0 / 3.0) * upper["mu"])
        return [background, upper]

    if case_config.name == "weak_interlayer":
        weak = {
            "name": "weak_interlayer",
            "point": [0.5, 0.5 * (float(case_config.weak_y_bottom) + float(case_config.weak_y_top))],
            "lambda": float(case_config.lambda_bg + case_config.lambda_ctr_1),
            "mu": float(case_config.mu_bg + case_config.mu_ctr_1),
        }
        weak["bulk"] = float(weak["lambda"] + (2.0 / 3.0) * weak["mu"])
        return [background, weak]

    inclusion = {
        "name": "inclusion",
        "point": [float(case_config.circle_1_cx), float(case_config.circle_1_cy)],
        "lambda": float(case_config.lambda_bg + case_config.lambda_ctr_1),
        "mu": float(case_config.mu_bg + case_config.mu_ctr_1),
    }
    inclusion["bulk"] = float(inclusion["lambda"] + (2.0 / 3.0) * inclusion["mu"])
    return [background, inclusion]


def init_material_probe_history_payload(case_config, args):
    probe_specs = build_material_probe_specs(case_config)
    payload = {
        "parameterization": str(getattr(args, "material_parameterization", "lamemu")),
        "probe_points": {spec["name"]: [float(spec["point"][0]), float(spec["point"][1])] for spec in probe_specs},
        "truth": {
            spec["name"]: {
                "lambda": float(spec["lambda"]),
                "mu": float(spec["mu"]),
                "bulk": float(spec["bulk"]),
            }
            for spec in probe_specs
        },
        "history": [],
        "snapshots": {},
    }
    return probe_specs, payload


def record_material_probe_snapshot(model, save_dir, case_config, args, step, tag="selected_best"):
    probe_specs, default_payload = init_material_probe_history_payload(case_config, args)
    payload_path = _get_save_path(save_dir, "json", "material_parameter_evolution.json")
    if os.path.exists(payload_path):
        try:
            with open(payload_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            payload = default_payload
    else:
        payload = default_payload
    points = np.asarray([spec["point"] for spec in probe_specs], dtype=float)
    predictions = predict_material_components(model, points, args, batch_size=max(len(points), 1))
    row = {"step": int(step), "probes": {}}
    for index, spec in enumerate(probe_specs):
        row["probes"][spec["name"]] = {
            "lambda": float(predictions["lambda"][index, 0]),
            "mu": float(predictions["mu"][index, 0]),
            "bulk": float(predictions["bulk"][index, 0]),
        }
    payload.setdefault("snapshots", {})[str(tag)] = row
    save_json(payload_path, payload)
    plot_material_probe_evolution(save_dir, payload)
    return row


def _infer_checkpoint_kind(model_path):
    name = os.path.basename(str(model_path or ""))
    if name.startswith("best_model"):
        return "best_model"
    if name.startswith("last_model"):
        return "last_model"
    return "checkpoint"


def _infer_checkpoint_step(model_path, default=None):
    stem = os.path.splitext(os.path.basename(str(model_path or "")))[0]
    if "-" in stem:
        suffix = stem.rsplit("-", 1)[-1]
        if suffix.isdigit():
            return int(suffix)
    if default is None:
        return None
    return int(default)


def resolve_checkpoint_snapshot_step(save_dir, model_path, fallback_step=None):
    checkpoint_kind = _infer_checkpoint_kind(model_path)
    if checkpoint_kind == "best_model":
        best_info = load_json_if_exists(_get_save_path(save_dir, "json", "best_validation_info.json"), default={})
        if isinstance(best_info, dict):
            model_name = os.path.basename(str(best_info.get("model_path", "") or ""))
            step = best_info.get("step")
            if step is not None and (not model_name or model_name == os.path.basename(str(model_path or ""))):
                return int(step)
    inferred_step = _infer_checkpoint_step(model_path, default=None)
    if inferred_step is not None:
        return inferred_step
    if fallback_step is not None:
        return int(fallback_step)
    return 0


def record_checkpoint_material_probe_snapshot(
    model,
    save_dir,
    case_config,
    args,
    checkpoint_preference="best_model",
    tag=None,
    model_path=None,
    restore_after=False,
    restore_device=None,
    verbose=0,
):
    checkpoint_path = find_model_path(save_dir, model_path, prefer=checkpoint_preference)
    checkpoint_kind = _infer_checkpoint_kind(checkpoint_path)
    snapshot_tag = tag
    if snapshot_tag is None:
        if checkpoint_kind == "best_model":
            snapshot_tag = "selected_best"
        elif checkpoint_kind == "last_model":
            snapshot_tag = "selected_last"
        else:
            snapshot_tag = "selected_checkpoint"
    fallback_step = int(
        getattr(getattr(model, "train_state", None), "iteration", 0)
        or getattr(getattr(model, "train_state", None), "step", 0)
        or 0
    )
    snapshot_step = resolve_checkpoint_snapshot_step(save_dir, checkpoint_path, fallback_step=fallback_step)
    restore_path = None
    if restore_after:
        restore_path = find_model_path(save_dir, None, prefer="last_model")
    model.restore(checkpoint_path, device=restore_device, verbose=verbose)
    row = record_material_probe_snapshot(
        model=model,
        save_dir=save_dir,
        case_config=case_config,
        args=args,
        step=snapshot_step,
        tag=snapshot_tag,
    )
    if restore_after and restore_path and os.path.abspath(restore_path) != os.path.abspath(checkpoint_path):
        model.restore(restore_path, device=restore_device, verbose=0)
    return {
        "checkpoint_path": checkpoint_path,
        "checkpoint_kind": checkpoint_kind,
        "step": int(snapshot_step),
        "tag": str(snapshot_tag),
        "row": row,
    }


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


def parse_load_modes(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        modes = [str(v).strip().lower() for v in value if str(v).strip()]
    else:
        text = str(value).strip()
        if not text:
            return []
        modes = [v.strip().lower() for v in text.split(",") if v.strip()]
    valid_modes = {
        "legacy",
        "x_tension",
        "y_tension",
        "shear",
        "biaxial_bulk",
        "uniaxial_x",
        "uniaxial_y",
        "normal_to_layer",
        "cross_layer_shear",
        "pure_shear",
        "bending_y",
        "top_nonuniform_compression",
        "edge_patch_load",
    }
    for mode in modes:
        if mode not in valid_modes:
            raise ValueError(f"Unsupported load mode: {mode}. Valid modes: {sorted(valid_modes)}")
    return modes


def _normalize_inverse_square_scales(scales, epsilon=1e-6):
    scales = np.asarray(scales, dtype=float)
    safe = np.maximum(scales, float(epsilon))
    inv = 1.0 / np.square(safe)
    mean_inv = float(np.mean(inv)) if len(inv) > 0 else 1.0
    if not np.isfinite(mean_inv) or mean_inv <= 0.0:
        return np.ones_like(inv)
    return inv / mean_inv


def compute_load_balance_profile(args, case_config=None):
    cached = getattr(args, "_cached_load_balance_profile", None)
    if cached is not None:
        return cached

    load_specs = resolve_load_specs(getattr(args, "load_scales", "1.0"), getattr(args, "load_modes", ""))
    num_loads = len(load_specs)
    default_profile = {
        "physics": [1.0] * num_loads,
        "boundary": [1.0] * num_loads,
        "data": [1.0] * num_loads,
        "physics_raw": [1.0] * num_loads,
        "boundary_raw": [1.0] * num_loads,
        "data_raw": [1.0] * num_loads,
    }
    if num_loads <= 1 or str(getattr(args, "load_balance_mode", "reference_rms")).lower() == "none":
        args._cached_load_balance_profile = default_profile
        return default_profile

    reference_data = getattr(args, "_reference_data", None)
    if isinstance(reference_data, dict) and reference_data.get("source") == "fem":
        try:
            physics_scales = []
            boundary_scales = []
            data_scales = []
            for load_index in range(num_loads):
                grid_state = np.asarray(reference_data["evaluation_grid_true_state_by_load"][load_index], dtype=float)
                boundary_state = np.asarray(reference_data["boundary_true_state_by_load"][load_index], dtype=float)
                physics_scales.append(float(np.sqrt(np.mean(np.square(grid_state[:, 2:5])))))
                boundary_scales.append(float(np.sqrt(np.mean(np.square(boundary_state[:, :2])))))
                data_scales.append(float(np.sqrt(np.mean(np.square(grid_state[:, :2])))))
            epsilon = float(getattr(args, "load_balance_epsilon", 1e-6))
            profile = {
                "physics": _normalize_inverse_square_scales(physics_scales, epsilon).tolist(),
                "boundary": _normalize_inverse_square_scales(boundary_scales, epsilon).tolist(),
                "data": _normalize_inverse_square_scales(data_scales, epsilon).tolist(),
                "physics_raw": [float(v) for v in physics_scales],
                "boundary_raw": [float(v) for v in boundary_scales],
                "data_raw": [float(v) for v in data_scales],
            }
            args._cached_load_balance_profile = profile
            return profile
        except Exception:
            pass

    case_config = case_config or get_case_config(args.case)
    eval_points, _, _ = make_grid(
        int(getattr(args, "load_balance_nx", 48)),
        int(getattr(args, "load_balance_ny", 48)),
    )
    boundary_points = build_boundary_points(max(int(getattr(args, "num_boundary", 0)), 400))

    physics_scales = []
    boundary_scales = []
    data_scales = []
    for spec in load_specs:
        load_scale = float(spec["scale"])
        load_mode = str(spec["mode"])
        state_grid = exact_state_numpy(eval_points, case_config, load_scale=load_scale, load_mode=load_mode)
        boundary_state = exact_state_numpy(boundary_points, case_config, load_scale=load_scale, load_mode=load_mode)
        physics_scales.append(float(np.sqrt(np.mean(np.square(state_grid[:, 2:5])))))
        boundary_scales.append(float(np.sqrt(np.mean(np.square(boundary_state[:, :2])))))
        data_scales.append(float(np.sqrt(np.mean(np.square(state_grid[:, :2])))))

    epsilon = float(getattr(args, "load_balance_epsilon", 1e-6))
    profile = {
        "physics": _normalize_inverse_square_scales(physics_scales, epsilon).tolist(),
        "boundary": _normalize_inverse_square_scales(boundary_scales, epsilon).tolist(),
        "data": _normalize_inverse_square_scales(data_scales, epsilon).tolist(),
        "physics_raw": [float(v) for v in physics_scales],
        "boundary_raw": [float(v) for v in boundary_scales],
        "data_raw": [float(v) for v in data_scales],
    }
    args._cached_load_balance_profile = profile
    return profile


def resolve_load_specs(load_scales, load_modes=None):
    scales = parse_load_scales(load_scales)
    modes = parse_load_modes(load_modes)
    if not modes:
        modes = ["legacy"] * len(scales)
    elif len(modes) == 1 and len(scales) > 1:
        modes = modes * len(scales)
    elif len(scales) == 1 and len(modes) > 1:
        scales = scales * len(modes)
    elif len(modes) != len(scales):
        raise ValueError("load_modes and load_scales must have the same length, or one of them must have length 1.")
    return [{"scale": float(scale), "mode": str(mode)} for scale, mode in zip(scales, modes)]


def compact_num_loads(args_or_method, load_scales=None, load_modes=None):
    if hasattr(args_or_method, "method"):
        load_scale_value = getattr(args_or_method, "load_scales", "1.0")
        load_mode_value = getattr(args_or_method, "load_modes", "")
    else:
        load_scale_value = load_scales if load_scales is not None else "1.0"
        load_mode_value = load_modes if load_modes is not None else ""
    return len(resolve_load_specs(load_scale_value, load_mode_value))


def compact_material_indices(args):
    num_loads = compact_num_loads(args)
    if is_compact_material_method(args.method):
        return 2 * num_loads, 2 * num_loads + 1
    return 5 * num_loads, 5 * num_loads + 1


def compact_state_indices(args, load_index=0):
    if is_compact_material_method(args.method):
        start = 2 * int(load_index)
    else:
        start = 5 * int(load_index)
    return start, start + 1


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


def smooth_horizontal_band_numpy(y, y_bottom, y_top, width):
    lower = smooth_step_numpy((y - y_bottom) / width)
    upper = smooth_step_numpy((y_top - y) / width)
    return lower * upper


def smooth_horizontal_band_torch(y, y_bottom, y_top, width):
    lower = smooth_step_torch((y - y_bottom) / width)
    upper = smooth_step_torch((y_top - y) / width)
    return lower * upper


def exact_material_numpy(points, case_config):
    x = points[:, 0:1]
    y = points[:, 1:2]
    width = case_config.interface_width

    if case_config.name == "single_material":
        lmbd = np.full_like(x, float(case_config.lambda_bg), dtype=float)
        mu = np.full_like(x, float(case_config.mu_bg), dtype=float)
        return lmbd, mu

    if case_config.name == "layered":
        mask = smooth_step_numpy((y - case_config.layer_y) / width)
        lmbd = case_config.lambda_bg + case_config.lambda_ctr_1 * mask
        mu = case_config.mu_bg + case_config.mu_ctr_1 * mask
        return lmbd, mu

    if case_config.name == "weak_interlayer":
        mask = smooth_horizontal_band_numpy(y, case_config.weak_y_bottom, case_config.weak_y_top, width)
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

    if case_config.name == "single_material":
        lmbd = torch.full_like(px, float(case_config.lambda_bg))
        mu = torch.full_like(px, float(case_config.mu_bg))
        return lmbd, mu

    if case_config.name == "layered":
        mask = smooth_step_torch((py - case_config.layer_y) / width)
        lmbd = case_config.lambda_bg + case_config.lambda_ctr_1 * mask
        mu = case_config.mu_bg + case_config.mu_ctr_1 * mask
        return lmbd, mu

    if case_config.name == "weak_interlayer":
        mask = smooth_horizontal_band_torch(py, case_config.weak_y_bottom, case_config.weak_y_top, width)
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


def exact_displacement_numpy(points, case_config, load_scale=1.0, load_mode="legacy"):
    x = points[:, 0:1]
    y = points[:, 1:2]
    mode = str(load_mode).strip().lower()
    base_amp = max(abs(case_config.amplitude_u), abs(case_config.amplitude_v), 1e-8)
    if mode in {"", "legacy"}:
        ux = load_scale * case_config.amplitude_u * np.sin(PI * x) * np.sin(PI * y)
        uy = load_scale * case_config.amplitude_v * np.sin(2.0 * PI * x) * np.sin(PI * y)
    elif mode == "x_tension":
        ux = load_scale * base_amp * np.sin(PI * x) * np.sin(PI * y)
        uy = np.zeros_like(ux)
    elif mode == "y_tension":
        uy = load_scale * base_amp * np.sin(PI * x) * np.sin(PI * y)
        ux = np.zeros_like(uy)
    elif mode == "shear":
        ux = load_scale * base_amp * np.sin(PI * x) * np.sin(2.0 * PI * y)
        uy = load_scale * base_amp * np.sin(2.0 * PI * x) * np.sin(PI * y)
    elif mode == "biaxial_bulk":
        ux = load_scale * base_amp * np.sin(PI * x)
        uy = load_scale * base_amp * np.sin(PI * y)
    elif mode == "uniaxial_x":
        ux = load_scale * base_amp * np.sin(PI * x)
        uy = np.zeros_like(ux)
    elif mode in {"uniaxial_y", "normal_to_layer"}:
        uy = load_scale * base_amp * np.sin(PI * y)
        ux = np.zeros_like(uy)
    elif mode == "cross_layer_shear":
        ux = load_scale * base_amp * np.sin(PI * y)
        uy = np.zeros_like(ux)
    elif mode == "pure_shear":
        ux = load_scale * base_amp * np.sin(PI * y)
        uy = load_scale * base_amp * np.sin(PI * x)
    elif mode == "bending_y":
        ux = -load_scale * base_amp * (y - 0.5) * np.cos(PI * x)
        uy = 0.25 * load_scale * base_amp * np.sin(PI * x)
    elif mode == "top_nonuniform_compression":
        ux = np.zeros_like(x)
        uy = load_scale * base_amp * y * np.sin(PI * x)
    elif mode == "edge_patch_load":
        patch_center = 0.25
        patch_width = 0.02
        patch_profile = np.exp(-((x - patch_center) ** 2) / patch_width)
        ux = np.zeros_like(x)
        uy = load_scale * base_amp * y * patch_profile
    else:
        raise ValueError(f"Unsupported load_mode: {load_mode}")
    return ux, uy


def exact_strain_numpy(points, case_config, load_scale=1.0, load_mode="legacy"):
    x = points[:, 0:1]
    y = points[:, 1:2]
    mode = str(load_mode).strip().lower()
    base_amp = max(abs(case_config.amplitude_u), abs(case_config.amplitude_v), 1e-8)
    if mode in {"", "legacy"}:
        exx = load_scale * case_config.amplitude_u * PI * np.cos(PI * x) * np.sin(PI * y)
        eyy = load_scale * case_config.amplitude_v * PI * np.sin(2.0 * PI * x) * np.cos(PI * y)
        exy = 0.5 * (
            load_scale * case_config.amplitude_u * PI * np.sin(PI * x) * np.cos(PI * y)
            + load_scale * case_config.amplitude_v * 2.0 * PI * np.cos(2.0 * PI * x) * np.sin(PI * y)
        )
    elif mode == "x_tension":
        exx = load_scale * base_amp * PI * np.cos(PI * x) * np.sin(PI * y)
        eyy = np.zeros_like(exx)
        exy = 0.5 * load_scale * base_amp * PI * np.sin(PI * x) * np.cos(PI * y)
    elif mode == "y_tension":
        exx = np.zeros_like(x)
        eyy = load_scale * base_amp * PI * np.sin(PI * x) * np.cos(PI * y)
        exy = 0.5 * load_scale * base_amp * PI * np.cos(PI * x) * np.sin(PI * y)
    elif mode == "shear":
        exx = load_scale * base_amp * PI * np.cos(PI * x) * np.sin(2.0 * PI * y)
        eyy = load_scale * base_amp * PI * np.sin(2.0 * PI * x) * np.cos(PI * y)
        exy = load_scale * base_amp * PI * (
            np.sin(PI * x) * np.cos(2.0 * PI * y) + np.cos(2.0 * PI * x) * np.sin(PI * y)
        )
    elif mode == "biaxial_bulk":
        exx = load_scale * base_amp * PI * np.cos(PI * x)
        eyy = load_scale * base_amp * PI * np.cos(PI * y)
        exy = np.zeros_like(exx)
    elif mode == "uniaxial_x":
        exx = load_scale * base_amp * PI * np.cos(PI * x)
        eyy = np.zeros_like(exx)
        exy = np.zeros_like(exx)
    elif mode in {"uniaxial_y", "normal_to_layer"}:
        exx = np.zeros_like(x)
        eyy = load_scale * base_amp * PI * np.cos(PI * y)
        exy = np.zeros_like(eyy)
    elif mode == "cross_layer_shear":
        exx = np.zeros_like(x)
        eyy = np.zeros_like(y)
        exy = 0.5 * load_scale * base_amp * PI * np.cos(PI * y)
    elif mode == "pure_shear":
        exx = np.zeros_like(x)
        eyy = np.zeros_like(y)
        exy = 0.5 * load_scale * base_amp * PI * (np.cos(PI * y) + np.cos(PI * x))
    elif mode == "bending_y":
        exx = load_scale * base_amp * (y - 0.5) * PI * np.sin(PI * x)
        eyy = np.zeros_like(exx)
        exy = 0.5 * load_scale * base_amp * np.cos(PI * x) * (0.25 * PI - 1.0)
    elif mode == "top_nonuniform_compression":
        exx = np.zeros_like(x)
        eyy = load_scale * base_amp * np.sin(PI * x)
        exy = 0.5 * load_scale * base_amp * y * PI * np.cos(PI * x)
    elif mode == "edge_patch_load":
        patch_center = 0.25
        patch_width = 0.02
        patch_profile = np.exp(-((x - patch_center) ** 2) / patch_width)
        patch_dx = (-2.0 * (x - patch_center) / patch_width) * patch_profile
        exx = np.zeros_like(x)
        eyy = load_scale * base_amp * patch_profile
        exy = 0.5 * load_scale * base_amp * y * patch_dx
    else:
        raise ValueError(f"Unsupported load_mode: {load_mode}")
    return exx, eyy, exy


def exact_state_numpy(points, case_config, load_scale=1.0, load_mode="legacy"):
    ux, uy = exact_displacement_numpy(points, case_config, load_scale=load_scale, load_mode=load_mode)
    exx, eyy, exy = exact_strain_numpy(points, case_config, load_scale=load_scale, load_mode=load_mode)
    lmbd, mu = exact_material_numpy(points, case_config)
    trace = exx + eyy
    sxx = lmbd * trace + 2.0 * mu * exx
    syy = lmbd * trace + 2.0 * mu * eyy
    sxy = 2.0 * mu * exy
    return np.hstack((ux, uy, sxx, syy, sxy, lmbd, mu))


def exact_state_torch(x, case_config, load_scale=1.0, load_mode="legacy"):
    px = x[:, 0:1]
    py = x[:, 1:2]
    mode = str(load_mode).strip().lower()
    base_amp = max(abs(case_config.amplitude_u), abs(case_config.amplitude_v), 1e-8)
    if mode in {"", "legacy"}:
        ux = load_scale * case_config.amplitude_u * torch.sin(PI * px) * torch.sin(PI * py)
        uy = load_scale * case_config.amplitude_v * torch.sin(2.0 * PI * px) * torch.sin(PI * py)
        exx = load_scale * case_config.amplitude_u * PI * torch.cos(PI * px) * torch.sin(PI * py)
        eyy = load_scale * case_config.amplitude_v * PI * torch.sin(2.0 * PI * px) * torch.cos(PI * py)
        exy = 0.5 * (
            load_scale * case_config.amplitude_u * PI * torch.sin(PI * px) * torch.cos(PI * py)
            + load_scale * case_config.amplitude_v * 2.0 * PI * torch.cos(2.0 * PI * px) * torch.sin(PI * py)
        )
    elif mode == "x_tension":
        ux = load_scale * base_amp * torch.sin(PI * px) * torch.sin(PI * py)
        uy = torch.zeros_like(ux)
        exx = load_scale * base_amp * PI * torch.cos(PI * px) * torch.sin(PI * py)
        eyy = torch.zeros_like(exx)
        exy = 0.5 * load_scale * base_amp * PI * torch.sin(PI * px) * torch.cos(PI * py)
    elif mode == "y_tension":
        uy = load_scale * base_amp * torch.sin(PI * px) * torch.sin(PI * py)
        ux = torch.zeros_like(uy)
        exx = torch.zeros_like(ux)
        eyy = load_scale * base_amp * PI * torch.sin(PI * px) * torch.cos(PI * py)
        exy = 0.5 * load_scale * base_amp * PI * torch.cos(PI * px) * torch.sin(PI * py)
    elif mode == "shear":
        ux = load_scale * base_amp * torch.sin(PI * px) * torch.sin(2.0 * PI * py)
        uy = load_scale * base_amp * torch.sin(2.0 * PI * px) * torch.sin(PI * py)
        exx = load_scale * base_amp * PI * torch.cos(PI * px) * torch.sin(2.0 * PI * py)
        eyy = load_scale * base_amp * PI * torch.sin(2.0 * PI * px) * torch.cos(PI * py)
        exy = load_scale * base_amp * PI * (
            torch.sin(PI * px) * torch.cos(2.0 * PI * py) + torch.cos(2.0 * PI * px) * torch.sin(PI * py)
        )
    elif mode == "biaxial_bulk":
        ux = load_scale * base_amp * torch.sin(PI * px)
        uy = load_scale * base_amp * torch.sin(PI * py)
        exx = load_scale * base_amp * PI * torch.cos(PI * px)
        eyy = load_scale * base_amp * PI * torch.cos(PI * py)
        exy = torch.zeros_like(exx)
    elif mode == "uniaxial_x":
        ux = load_scale * base_amp * torch.sin(PI * px)
        uy = torch.zeros_like(ux)
        exx = load_scale * base_amp * PI * torch.cos(PI * px)
        eyy = torch.zeros_like(exx)
        exy = torch.zeros_like(exx)
    elif mode in {"uniaxial_y", "normal_to_layer"}:
        uy = load_scale * base_amp * torch.sin(PI * py)
        ux = torch.zeros_like(uy)
        exx = torch.zeros_like(px)
        eyy = load_scale * base_amp * PI * torch.cos(PI * py)
        exy = torch.zeros_like(eyy)
    elif mode == "cross_layer_shear":
        ux = load_scale * base_amp * torch.sin(PI * py)
        uy = torch.zeros_like(ux)
        exx = torch.zeros_like(px)
        eyy = torch.zeros_like(py)
        exy = 0.5 * load_scale * base_amp * PI * torch.cos(PI * py)
    elif mode == "pure_shear":
        ux = load_scale * base_amp * torch.sin(PI * py)
        uy = load_scale * base_amp * torch.sin(PI * px)
        exx = torch.zeros_like(px)
        eyy = torch.zeros_like(py)
        exy = 0.5 * load_scale * base_amp * PI * (torch.cos(PI * py) + torch.cos(PI * px))
    elif mode == "bending_y":
        ux = -load_scale * base_amp * (py - 0.5) * torch.cos(PI * px)
        uy = 0.25 * load_scale * base_amp * torch.sin(PI * px)
        exx = load_scale * base_amp * (py - 0.5) * PI * torch.sin(PI * px)
        eyy = torch.zeros_like(exx)
        exy = 0.5 * load_scale * base_amp * torch.cos(PI * px) * (0.25 * PI - 1.0)
    elif mode == "top_nonuniform_compression":
        ux = torch.zeros_like(px)
        uy = load_scale * base_amp * py * torch.sin(PI * px)
        exx = torch.zeros_like(px)
        eyy = load_scale * base_amp * torch.sin(PI * px)
        exy = 0.5 * load_scale * base_amp * py * PI * torch.cos(PI * px)
    elif mode == "edge_patch_load":
        patch_center = 0.25
        patch_width = 0.02
        patch_profile = torch.exp(-((px - patch_center) ** 2) / patch_width)
        patch_dx = (-2.0 * (px - patch_center) / patch_width) * patch_profile
        ux = torch.zeros_like(px)
        uy = load_scale * base_amp * py * patch_profile
        exx = torch.zeros_like(px)
        eyy = load_scale * base_amp * patch_profile
        exy = 0.5 * load_scale * base_amp * py * patch_dx
    else:
        raise ValueError(f"Unsupported load_mode: {load_mode}")
    lmbd, mu = exact_material_torch(x, case_config)
    trace = exx + eyy
    sxx = lmbd * trace + 2.0 * mu * exx
    syy = lmbd * trace + 2.0 * mu * eyy
    sxy = 2.0 * mu * exy
    return torch.cat((ux, uy, sxx, syy, sxy, lmbd, mu), dim=1)


def exact_body_force_torch(x, case_config, load_scale=1.0, load_mode="legacy"):
    exact_state = exact_state_torch(x, case_config, load_scale=load_scale, load_mode=load_mode)
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


def _sample_points_by_mask(count, rng, mask_fn, batch_size=4096):
    target = int(count)
    if target <= 0:
        return np.zeros((0, 2), dtype=float)
    samples = []
    total = 0
    attempts = 0
    while total < target:
        attempts += 1
        if attempts > 20000:
            raise RuntimeError("Failed to sample enough observation points for the requested region.")
        candidates = rng.random((int(batch_size), 2))
        accepted = candidates[np.asarray(mask_fn(candidates), dtype=bool)]
        if len(accepted) == 0:
            continue
        samples.append(accepted)
        total += len(accepted)
    return np.vstack(samples)[:target]


def _build_three_group_observation_splits(num_train, num_val, num_eval, seed, case_config):
    region_count = 3
    split_sizes = {
        "train": int(num_train),
        "val": int(num_val),
        "eval": int(num_eval),
    }
    split_regions = {}
    for split_name, total in split_sizes.items():
        if total % region_count != 0:
            raise ValueError(f"{split_name} observation count must be divisible by 3 for balanced three-group sampling.")
        per_region = total // region_count
        rng = np.random.default_rng(int(seed) + {"train": 0, "val": 101, "eval": 202}[split_name])
        if case_config.name == "layered":
            half_band = max(float(case_config.interface_width) * 2.0, 0.05)
            y0 = float(case_config.layer_y)

            def lower_mask(points):
                return points[:, 1] <= (y0 - half_band)

            def upper_mask(points):
                return points[:, 1] >= (y0 + half_band)

            def interface_mask(points):
                return np.abs(points[:, 1] - y0) < half_band

            lower_points = _sample_points_by_mask(per_region, rng, lower_mask)
            upper_points = _sample_points_by_mask(per_region, rng, upper_mask)
            interface_points = _sample_points_by_mask(per_region, rng, interface_mask)
            combined = np.vstack([lower_points, upper_points, interface_points])
        elif case_config.name == "single_inclusion":
            radius = float(case_config.circle_1_r)
            half_band = max(float(case_config.interface_width) * 2.0, 0.04)
            inner_radius = max(radius - half_band, 0.05)
            outer_radius = min(radius + half_band, 0.45)
            cx = float(case_config.circle_1_cx)
            cy = float(case_config.circle_1_cy)

            def radial_distance(points):
                return np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)

            def background_mask(points):
                return radial_distance(points) >= outer_radius

            def inclusion_mask(points):
                return radial_distance(points) <= inner_radius

            def interface_mask(points):
                dist = radial_distance(points)
                return (dist > inner_radius) & (dist < outer_radius)

            background_points = _sample_points_by_mask(per_region, rng, background_mask)
            inclusion_points = _sample_points_by_mask(per_region, rng, inclusion_mask)
            interface_points = _sample_points_by_mask(per_region, rng, interface_mask)
            combined = np.vstack([background_points, inclusion_points, interface_points])
        elif case_config.name == "weak_interlayer":
            half_band = max(float(case_config.interface_width) * 2.0, 0.05)
            y_bottom = float(case_config.weak_y_bottom)
            y_top = float(case_config.weak_y_top)

            def lower_mask(points):
                return points[:, 1] <= (y_bottom - half_band)

            def weak_band_mask(points):
                return (points[:, 1] >= y_bottom - half_band) & (points[:, 1] <= y_top + half_band)

            def upper_mask(points):
                return points[:, 1] >= (y_top + half_band)

            lower_points = _sample_points_by_mask(per_region, rng, lower_mask)
            weak_band_points = _sample_points_by_mask(per_region, rng, weak_band_mask)
            upper_points = _sample_points_by_mask(per_region, rng, upper_mask)
            combined = np.vstack([lower_points, weak_band_points, upper_points])
        elif case_config.name == "single_material":
            def left_mask(points):
                return points[:, 0] <= 1.0 / 3.0

            def middle_mask(points):
                return (points[:, 0] > 1.0 / 3.0) & (points[:, 0] < 2.0 / 3.0)

            def right_mask(points):
                return points[:, 0] >= 2.0 / 3.0

            left_points = _sample_points_by_mask(per_region, rng, left_mask)
            middle_points = _sample_points_by_mask(per_region, rng, middle_mask)
            right_points = _sample_points_by_mask(per_region, rng, right_mask)
            combined = np.vstack([left_points, middle_points, right_points])
        else:
            raise ValueError(f"Three-group observation sampling is not implemented for case: {case_config.name}")

        permutation = rng.permutation(len(combined))
        split_regions[split_name] = np.asarray(combined[permutation], dtype=float)
    return split_regions


def resolve_observation_split_cache_path(cache_dir, case_name, seed, num_train, num_val, num_eval, tag):
    filename = (
        f"{case_name}_seed{seed}_train{int(num_train)}_val{int(num_val)}_eval{int(num_eval)}_{tag}.npz"
    )
    return os.path.join(cache_dir, filename)


def build_observation_splits(num_train, num_val, num_eval, seed, case_name, case_config=None, cache_dir=None, tag="official_softbc_v1"):
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

    if case_config is not None and "balanced1200" in str(tag).lower():
        split_regions = _build_three_group_observation_splits(num_train, num_val, num_eval, seed, case_config)
        splits = {
            "train": split_regions["train"],
            "val": split_regions["val"],
            "eval": split_regions["eval"],
            "cache_path": cache_path,
        }
    else:
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


def select_boundary_edge(points, edge="top", tol=1e-8):
    points = np.asarray(points, dtype=float)
    edge_name = str(edge).strip().lower()
    if edge_name == "top":
        mask = np.isclose(points[:, 1], 1.0, atol=tol)
        normal = np.array([0.0, 1.0], dtype=float)
        edge_length = 1.0
    elif edge_name == "bottom":
        mask = np.isclose(points[:, 1], 0.0, atol=tol)
        normal = np.array([0.0, -1.0], dtype=float)
        edge_length = 1.0
    elif edge_name == "left":
        mask = np.isclose(points[:, 0], 0.0, atol=tol)
        normal = np.array([-1.0, 0.0], dtype=float)
        edge_length = 1.0
    elif edge_name == "right":
        mask = np.isclose(points[:, 0], 1.0, atol=tol)
        normal = np.array([1.0, 0.0], dtype=float)
        edge_length = 1.0
    else:
        raise ValueError(f"Unsupported boundary edge: {edge}")
    edge_points = np.asarray(points[mask], dtype=float)
    if len(edge_points) == 0:
        return edge_points, np.zeros((0, 2), dtype=float), np.zeros((0, 1), dtype=float)
    normals = np.tile(normal.reshape(1, 2), (len(edge_points), 1))
    weights = np.full((len(edge_points), 1), float(edge_length) / float(len(edge_points)), dtype=float)
    return edge_points, normals, weights


def compute_total_reaction_target(points, case_config, load_scale=1.0, load_mode="legacy", edge="top"):
    edge_points, normals, weights = select_boundary_edge(points, edge=edge)
    if len(edge_points) == 0:
        return {
            "points": edge_points,
            "normals": normals,
            "weights": weights,
            "target_fx": 0.0,
            "target_fy": 0.0,
            "edge": str(edge),
        }
    state = exact_state_numpy(edge_points, case_config, load_scale=load_scale, load_mode=load_mode)
    sxx = state[:, 2:3]
    syy = state[:, 3:4]
    sxy = state[:, 4:5]
    nx = normals[:, 0:1]
    ny = normals[:, 1:2]
    tx = sxx * nx + sxy * ny
    ty = sxy * nx + syy * ny
    target_fx = float(np.sum(tx * weights))
    target_fy = float(np.sum(ty * weights))
    return {
        "points": edge_points,
        "normals": normals,
        "weights": weights,
        "target_fx": target_fx,
        "target_fy": target_fy,
        "edge": str(edge),
    }


def build_reaction_operator(points, normals, weights, args, load_index, component="x"):
    points = np.asarray(points, dtype=float)
    normals = np.asarray(normals, dtype=float)
    weights = np.asarray(weights, dtype=float)
    rounded_points = np.round(points, decimals=8)
    point_lookup = {
        tuple(pt.tolist()): (
            float(normals[i, 0]),
            float(normals[i, 1]),
            float(weights[i, 0]),
        )
        for i, pt in enumerate(rounded_points)
    }
    component_name = str(component).strip().lower()
    if is_compact_material_method(args.method):
        raise ValueError("Total reaction consistency is only supported for methods with explicit stress outputs.")
    state_offset = 5 * int(load_index)
    sxx_index = state_offset + 2
    syy_index = state_offset + 3
    sxy_index = state_offset + 4

    def operator(inputs, outputs, X):
        device = outputs.device
        dtype = outputs.dtype
        if torch.is_tensor(inputs):
            x_np = inputs.detach().cpu().numpy()
        else:
            x_np = np.asarray(inputs, dtype=float)
        normals_full = np.zeros((x_np.shape[0], 2), dtype=float)
        weights_full = np.zeros((x_np.shape[0], 1), dtype=float)
        rounded_x = np.round(x_np[:, :2], decimals=8)
        for row_index, pt in enumerate(rounded_x):
            item = point_lookup.get(tuple(pt.tolist()))
            if item is None:
                continue
            normals_full[row_index, 0] = item[0]
            normals_full[row_index, 1] = item[1]
            weights_full[row_index, 0] = item[2]
        normals_t = torch.as_tensor(normals_full, dtype=dtype, device=device)
        weights_t = torch.as_tensor(weights_full, dtype=dtype, device=device)
        nx = normals_t[:, 0:1]
        ny = normals_t[:, 1:2]
        sxx = outputs[:, sxx_index:sxx_index + 1]
        syy = outputs[:, syy_index:syy_index + 1]
        sxy = outputs[:, sxy_index:sxy_index + 1]
        traction_x = sxx * nx + sxy * ny
        traction_y = sxy * nx + syy * ny
        if component_name == "x":
            resultant = torch.sum(traction_x * weights_t)
        elif component_name == "y":
            resultant = torch.sum(traction_y * weights_t)
        else:
            raise ValueError(f"Unsupported reaction component: {component}")
        return torch.ones((outputs.shape[0], 1), dtype=dtype, device=device) * resultant

    return operator


def reaction_consistency_enabled(args):
    return bool(getattr(args, "reaction_consistency", False)) and float(getattr(args, "reaction_weight", 0.0)) > 0.0


def build_bc_loss_names(args):
    num_loads = compact_num_loads(args)
    names = []
    for load_index in range(num_loads):
        names.extend(
            [
                f"boundary_ux_l{load_index}",
                f"boundary_uy_l{load_index}",
                f"obs_ux_l{load_index}",
                f"obs_uy_l{load_index}",
            ]
        )
        if reaction_consistency_enabled(args) and not is_compact_material_method(args.method):
            names.extend([f"reaction_fx_l{load_index}", f"reaction_fy_l{load_index}"])
    return names


def add_noise(values, noise_level, seed, scale_mode="component_std"):
    if noise_level <= 0:
        return values.copy()
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    mode = str(scale_mode).strip().lower()
    if mode == "component_std":
        scale = np.std(values, axis=0, keepdims=True)
        scale = np.where(scale < 1e-8, 1.0, scale)
    elif mode == "component_std_no_fallback":
        scale = np.std(values, axis=0, keepdims=True)
    elif mode == "global_rms":
        scale_value = float(np.sqrt(np.mean(values**2)))
        if scale_value < 1e-12:
            return values.copy()
        scale = np.full((1, values.shape[1]), scale_value, dtype=float)
    else:
        raise ValueError(f"Unsupported noise_scale_mode: {scale_mode}")
    noise = rng.normal(0.0, 1.0, size=values.shape) * scale * noise_level
    return values + noise


def validation_observation_target_name(args):
    return str(getattr(args, "validation_observation_target", "clean")).strip().lower()


def resolve_validation_observation_target(args, payload):
    target = validation_observation_target_name(args)
    if target == "clean":
        return np.asarray(payload["clean"], dtype=float)
    if target == "noisy":
        return np.asarray(payload["noisy"], dtype=float)
    raise ValueError(f"Unsupported validation_observation_target: {target}")


def report_checkpoint_preference(args):
    return str(getattr(args, "report_checkpoint", "best_model")).strip().lower()


def safe_mode_name(load_mode):
    return str(load_mode).replace(" ", "_").replace("-", "_").lower()


def load_npz_payload(path):
    payload = np.load(path)
    return {
        "points": np.asarray(payload["points"], dtype=float),
        "clean": np.asarray(payload["clean"], dtype=float),
        "noisy": np.asarray(payload["noisy"], dtype=float),
        "true_state": np.asarray(payload["true_state"], dtype=float),
    }


def load_fem_teacher_reference(run_dir):
    run_dir = os.path.abspath(run_dir)
    bundle_path = os.path.join(run_dir, "npz", "teacher_bundle.npz")
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Missing FEM teacher bundle: {bundle_path}")
    bundle = np.load(bundle_path)
    load_modes = [str(item) for item in bundle["load_modes"].tolist()]
    load_scales = [float(item) for item in bundle["load_scales"].tolist()]
    reference = {
        "source": "fem",
        "run_dir": run_dir,
        "bundle_path": bundle_path,
        "load_modes": load_modes,
        "load_scales": load_scales,
        "boundary_points": np.asarray(bundle["boundary_points"], dtype=float),
        "boundary_true_state_by_load": np.asarray(bundle["boundary_true_state"], dtype=float),
        "evaluation_grid_points": np.asarray(bundle["evaluation_grid_points"], dtype=float),
        "evaluation_grid_true_state_by_load": np.asarray(bundle["evaluation_grid_true_state"], dtype=float),
        "train_points": np.asarray(bundle["train_points"], dtype=float),
        "validation_points": np.asarray(bundle["validation_points"], dtype=float),
        "evaluation_points": np.asarray(bundle["evaluation_points"], dtype=float),
    }

    split_payloads = {"boundary": [], "train": [], "validation": [], "evaluation": []}
    for load_index, load_mode in enumerate(load_modes):
        mode_tag = safe_mode_name(load_mode)
        for split_name in split_payloads:
            path = os.path.join(run_dir, "npz", f"{split_name}_l{load_index}_{mode_tag}_set_data.npz")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing FEM teacher split payload: {path}")
            split_payloads[split_name].append(load_npz_payload(path))
    reference["boundary_payloads"] = split_payloads["boundary"]
    reference["train_payloads"] = split_payloads["train"]
    reference["validation_payloads"] = split_payloads["validation"]
    reference["evaluation_payloads"] = split_payloads["evaluation"]

    meta_path = os.path.join(run_dir, "json", "teacher_meta.json")
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
    reference["meta_path"] = meta_path if meta is not None else None
    reference["meta"] = meta
    if isinstance(meta, dict):
        splits = meta.get("splits", {})
        reference["observation_cache_path"] = splits.get("observation_cache_path")
        reference["observation_split_tag"] = splits.get("observation_split_tag")
        reference["noise_level"] = float(splits.get("noise_level", 0.0))
    else:
        reference["observation_cache_path"] = None
        reference["observation_split_tag"] = None
        reference["noise_level"] = 0.0
    return reference


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


def reorder_field_major_to_load_major(field_outputs, num_loads, num_fields=len(STATE_FIELD_NAMES)):
    num_loads = int(max(num_loads, 1))
    if num_loads == 1:
        return field_outputs
    expected_width = int(num_fields) * num_loads
    if field_outputs.shape[1] != expected_width:
        raise ValueError(
            f"Expected field-major outputs width {expected_width}, got {field_outputs.shape[1]}."
        )
    field_chunks = torch.chunk(field_outputs, int(num_fields), dim=1)
    interleaved = []
    for load_index in range(num_loads):
        for field_chunk in field_chunks:
            interleaved.append(field_chunk[:, load_index : load_index + 1])
    return torch.cat(interleaved, dim=1)


def decode_material_parameterization(raw_outputs, parameterization, lambda_floor, mu_floor, k_floor):
    parameterization = str(parameterization).lower()
    if parameterization == "bulkmu":
        bulk = float(k_floor) + F.softplus(raw_outputs[:, 0:1])
        mu = float(mu_floor) + F.softplus(raw_outputs[:, 1:2])
        lmbd = torch.clamp(bulk - (2.0 / 3.0) * mu, min=float(lambda_floor))
        diagnostics = {"bulk": bulk}
        return lmbd, mu, diagnostics
    if parameterization == "lamemu":
        lmbd = float(lambda_floor) + F.softplus(raw_outputs[:, 0:1])
        mu = float(mu_floor) + F.softplus(raw_outputs[:, 1:2])
        diagnostics = {}
        return lmbd, mu, diagnostics
    raise ValueError(f"Unsupported material parameterization: {parameterization}")


class TwoBranchCompactMaterialFieldNet(dde.nn.pytorch.nn.NN):
    def __init__(
        self,
        state_hidden_layers,
        material_hidden_layers,
        num_loads=1,
        activation="tanh",
        num_frequencies=0,
        lambda_floor=0.1,
        mu_floor=0.1,
        k_floor=0.2,
        parameterization="bulkmu",
        backbone_type="mlp",
    ):
        super().__init__()
        self.features = FourierFeatureMap(num_frequencies)
        self.num_loads = int(num_loads)
        self.lambda_floor = float(lambda_floor)
        self.mu_floor = float(mu_floor)
        self.k_floor = float(k_floor)
        self.parameterization = str(parameterization).lower()
        feature_dim = self.features.output_dim
        self.state_net = build_backbone(
            input_dim=feature_dim,
            hidden_layers=state_hidden_layers,
            output_dim=2 * self.num_loads,
            activation=activation,
            backbone_type=backbone_type,
        )
        self.material_net = build_backbone(
            input_dim=feature_dim,
            hidden_layers=material_hidden_layers,
            output_dim=2,
            activation=activation,
            backbone_type=backbone_type,
        )

    def _material_from_features(self, features):
        raw_outputs = self.material_net(features)
        return decode_material_parameterization(
            raw_outputs,
            parameterization=self.parameterization,
            lambda_floor=self.lambda_floor,
            mu_floor=self.mu_floor,
            k_floor=self.k_floor,
        )

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(inputs)
        features = self.features(x)
        state_outputs = self.state_net(features)
        lmbd, mu, _ = self._material_from_features(features)
        return torch.cat((state_outputs, lmbd, mu), dim=1)


class TwoBranchStressMaterialFieldNet(dde.nn.pytorch.nn.NN):
    def __init__(
        self,
        state_hidden_layers,
        material_hidden_layers,
        num_loads=1,
        activation="tanh",
        num_frequencies=0,
        lambda_floor=0.1,
        mu_floor=0.1,
        k_floor=0.2,
        parameterization="bulkmu",
        backbone_type="mlp",
    ):
        super().__init__()
        self.features = FourierFeatureMap(num_frequencies)
        self.num_loads = int(num_loads)
        self.lambda_floor = float(lambda_floor)
        self.mu_floor = float(mu_floor)
        self.k_floor = float(k_floor)
        self.parameterization = str(parameterization).lower()
        feature_dim = self.features.output_dim
        self.state_net = build_backbone(
            input_dim=feature_dim,
            hidden_layers=state_hidden_layers,
            output_dim=5 * self.num_loads,
            activation=activation,
            backbone_type=backbone_type,
        )
        self.material_net = build_backbone(
            input_dim=feature_dim,
            hidden_layers=material_hidden_layers,
            output_dim=2,
            activation=activation,
            backbone_type=backbone_type,
        )

    def _material_from_features(self, features):
        raw_outputs = self.material_net(features)
        return decode_material_parameterization(
            raw_outputs,
            parameterization=self.parameterization,
            lambda_floor=self.lambda_floor,
            mu_floor=self.mu_floor,
            k_floor=self.k_floor,
        )

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(inputs)
        features = self.features(x)
        state_outputs = self.state_net(features)
        lmbd, mu, _ = self._material_from_features(features)
        return torch.cat((state_outputs, lmbd, mu), dim=1)


class ThreeBranchStressMaterialFieldNet(dde.nn.pytorch.nn.NN):
    def __init__(
        self,
        state_hidden_layers,
        material_hidden_layers,
        interface_hidden_layers,
        num_loads=1,
        activation="tanh",
        num_frequencies=0,
        lambda_floor=0.1,
        mu_floor=0.1,
        k_floor=0.2,
        parameterization="bulkmu",
        interface_sharpness=10.0,
        backbone_type="mlp",
    ):
        super().__init__()
        self.features = FourierFeatureMap(num_frequencies)
        self.num_loads = int(num_loads)
        self.lambda_floor = float(lambda_floor)
        self.mu_floor = float(mu_floor)
        self.k_floor = float(k_floor)
        self.interface_sharpness = float(interface_sharpness)
        self.parameterization = str(parameterization).lower()
        feature_dim = self.features.output_dim
        self.state_net = build_backbone(
            input_dim=feature_dim,
            hidden_layers=state_hidden_layers,
            output_dim=5 * self.num_loads,
            activation=activation,
            backbone_type=backbone_type,
        )
        self.material_net = build_backbone(
            input_dim=feature_dim,
            hidden_layers=material_hidden_layers,
            output_dim=4,
            activation=activation,
            backbone_type=backbone_type,
        )
        self.interface_net = build_backbone(
            input_dim=feature_dim,
            hidden_layers=interface_hidden_layers,
            output_dim=1,
            activation=activation,
            backbone_type=backbone_type,
        )

    def _decode_region_materials(self, raw_outputs):
        raw_region_a = raw_outputs[:, 0:2]
        raw_region_b = raw_outputs[:, 2:4]
        lmbd_a, mu_a, diag_a = decode_material_parameterization(
            raw_region_a,
            parameterization=self.parameterization,
            lambda_floor=self.lambda_floor,
            mu_floor=self.mu_floor,
            k_floor=self.k_floor,
        )
        lmbd_b, mu_b, diag_b = decode_material_parameterization(
            raw_region_b,
            parameterization=self.parameterization,
            lambda_floor=self.lambda_floor,
            mu_floor=self.mu_floor,
            k_floor=self.k_floor,
        )
        bulk_a = diag_a.get("bulk") if isinstance(diag_a, dict) else None
        bulk_b = diag_b.get("bulk") if isinstance(diag_b, dict) else None
        if bulk_a is None:
            bulk_a = lmbd_a + (2.0 / 3.0) * mu_a
        if bulk_b is None:
            bulk_b = lmbd_b + (2.0 / 3.0) * mu_b
        return lmbd_a, mu_a, bulk_a, lmbd_b, mu_b, bulk_b

    def _material_from_features(self, features):
        raw_outputs = self.material_net(features)
        lmbd_a, mu_a, bulk_a, lmbd_b, mu_b, bulk_b = self._decode_region_materials(raw_outputs)
        raw_interface = self.interface_net(features)
        gate = torch.sigmoid(self.interface_sharpness * raw_interface)
        lmbd = (1.0 - gate) * lmbd_a + gate * lmbd_b
        mu = (1.0 - gate) * mu_a + gate * mu_b
        bulk = (1.0 - gate) * bulk_a + gate * bulk_b
        return lmbd, mu, bulk, gate, lmbd_a, mu_a, bulk_a, lmbd_b, mu_b, bulk_b

    def predict_material_diagnostics(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(inputs)
        features = self.features(x)
        (
            lmbd,
            mu,
            bulk,
            gate,
            lmbd_a,
            mu_a,
            bulk_a,
            lmbd_b,
            mu_b,
            bulk_b,
        ) = self._material_from_features(features)
        class_probs = torch.cat((1.0 - gate, gate), dim=1)
        return {
            "lambda": lmbd,
            "mu": mu,
            "bulk": bulk,
            "interface_indicator": gate,
            "class_probs": class_probs,
            "lambda_regions": torch.cat((lmbd_a.mean(dim=0), lmbd_b.mean(dim=0)), dim=0),
            "mu_regions": torch.cat((mu_a.mean(dim=0), mu_b.mean(dim=0)), dim=0),
            "bulk_regions": torch.cat((bulk_a.mean(dim=0), bulk_b.mean(dim=0)), dim=0),
        }

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(inputs)
        features = self.features(x)
        state_outputs = self.state_net(features)
        lmbd, mu, _, _, _, _, _, _, _, _ = self._material_from_features(features)
        return torch.cat((state_outputs, lmbd, mu), dim=1)


class FiveStateMLPStressMaterialFieldNet(dde.nn.pytorch.nn.NN):
    def __init__(
        self,
        state_hidden_layers,
        material_hidden_layers,
        num_loads=1,
        activation="tanh",
        num_frequencies=0,
        lambda_floor=0.1,
        mu_floor=0.1,
        k_floor=0.2,
        parameterization="bulkmu",
        backbone_type="mlp",
    ):
        super().__init__()
        self.features = FourierFeatureMap(num_frequencies)
        self.num_loads = int(num_loads)
        self.lambda_floor = float(lambda_floor)
        self.mu_floor = float(mu_floor)
        self.k_floor = float(k_floor)
        self.parameterization = str(parameterization).lower()
        feature_dim = self.features.output_dim
        self.state_nets = nn.ModuleDict(
            {
                field_name: build_backbone(
                    input_dim=feature_dim,
                    hidden_layers=state_hidden_layers,
                    output_dim=self.num_loads,
                    activation=activation,
                    backbone_type=backbone_type,
                )
                for field_name in STATE_FIELD_NAMES
            }
        )
        self.material_net = build_backbone(
            input_dim=feature_dim,
            hidden_layers=material_hidden_layers,
            output_dim=2,
            activation=activation,
            backbone_type=backbone_type,
        )

    def _material_from_features(self, features):
        raw_outputs = self.material_net(features)
        return decode_material_parameterization(
            raw_outputs,
            parameterization=self.parameterization,
            lambda_floor=self.lambda_floor,
            mu_floor=self.mu_floor,
            k_floor=self.k_floor,
        )

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(inputs)
        features = self.features(x)
        field_major_outputs = torch.cat(
            [self.state_nets[field_name](features) for field_name in STATE_FIELD_NAMES],
            dim=1,
        )
        state_outputs = reorder_field_major_to_load_major(field_major_outputs, self.num_loads)
        lmbd, mu, _ = self._material_from_features(features)
        return torch.cat((state_outputs, lmbd, mu), dim=1)


class PFNNScalarMaterialFieldNet(dde.nn.pytorch.nn.NN):
    def __init__(
        self,
        state_hidden_layers,
        num_loads=1,
        activation="tanh",
        num_frequencies=0,
        lambda_floor=0.1,
        mu_floor=0.1,
        k_floor=0.2,
        parameterization="bulkmu",
    ):
        super().__init__()
        self.features = FourierFeatureMap(num_frequencies)
        self.num_loads = int(num_loads)
        self.lambda_floor = float(lambda_floor)
        self.mu_floor = float(mu_floor)
        self.k_floor = float(k_floor)
        self.parameterization = str(parameterization).lower()
        feature_dim = self.features.output_dim
        pfnn_layers = [feature_dim]
        for hidden_dim in state_hidden_layers:
            pfnn_layers.append([int(hidden_dim)] * len(STATE_FIELD_NAMES))
        pfnn_layers.append([self.num_loads] * len(STATE_FIELD_NAMES))
        self.state_net = dde.nn.PFNN(pfnn_layers, activation, "Glorot uniform")
        if self.parameterization == "bulkmu":
            self.raw_material_params = nn.Parameter(torch.tensor([-0.05, 0.2], dtype=torch.float32))
        elif self.parameterization == "lamemu":
            self.raw_material_params = nn.Parameter(torch.tensor([-0.15, -0.05], dtype=torch.float32))
        else:
            raise ValueError(f"Unsupported material parameterization: {self.parameterization}")

    def _material_from_batch(self, batch_size, device, dtype):
        raw_outputs = self.raw_material_params.to(device=device, dtype=dtype).unsqueeze(0).expand(int(batch_size), -1)
        return decode_material_parameterization(
            raw_outputs,
            parameterization=self.parameterization,
            lambda_floor=self.lambda_floor,
            mu_floor=self.mu_floor,
            k_floor=self.k_floor,
        )

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(inputs)
        features = self.features(x)
        field_major_outputs = self.state_net(features)
        state_outputs = reorder_field_major_to_load_major(field_major_outputs, self.num_loads)
        lmbd, mu, _ = self._material_from_batch(features.shape[0], features.device, features.dtype)
        return torch.cat((state_outputs, lmbd, mu), dim=1)


class VanillaMaterialFieldNet(dde.nn.pytorch.nn.NN):
    def __init__(self, hidden_layers, activation="tanh", num_frequencies=0, backbone_type="mlp", output_dim=None):
        super().__init__()
        self.features = FourierFeatureMap(num_frequencies)
        if output_dim is None:
            output_dim = len(FIELD_NAMES)
        self.backbone = build_backbone(
            input_dim=self.features.output_dim,
            hidden_layers=hidden_layers,
            output_dim=int(output_dim),
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
    if case_name in {"layered", "single_inclusion", "weak_interlayer"}:
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
        num_loads=1,
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
        self.num_loads = int(num_loads)
        self.lambda_floor = float(lambda_floor)
        self.mu_floor = float(mu_floor)
        self.interface_sharpness = float(interface_sharpness)
        feature_dim = self.features.output_dim
        self.state_net = build_backbone(
            input_dim=feature_dim,
            hidden_layers=state_hidden_layers,
            output_dim=2 * self.num_loads,
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
        num_loads=1,
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
        self.num_loads = int(num_loads)
        self.num_regions = num_regions_for_case(case_name)
        self.lambda_floor = float(lambda_floor)
        self.mu_floor = float(mu_floor)
        self.interface_sharpness = float(interface_sharpness)
        self.state_net = build_backbone(
            input_dim=self.features.output_dim,
            hidden_layers=state_hidden_layers,
            output_dim=2 * self.num_loads,
            activation=activation,
            backbone_type=backbone_type,
        )
        lambda_init = torch.linspace(-0.35, 0.35, steps=self.num_regions + 1, dtype=torch.float32)
        mu_init = torch.linspace(-0.2, 0.2, steps=self.num_regions + 1, dtype=torch.float32)
        self.raw_lambda_params = nn.Parameter(lambda_init.clone())
        self.raw_mu_params = nn.Parameter(mu_init.clone())

        if case_name == "layered":
            self.raw_layer_y = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        elif case_name == "weak_interlayer":
            self.raw_weak_center = nn.Parameter(torch.tensor(-0.5, dtype=torch.float32))
            self.raw_weak_half_thickness = nn.Parameter(torch.tensor(-1.6, dtype=torch.float32))
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

        if self.case_name == "weak_interlayer":
            center = bounded_sigmoid(self.raw_weak_center, 0.2, 0.8)
            half_thickness = bounded_sigmoid(self.raw_weak_half_thickness, 0.03, 0.18)
            signed_distance = half_thickness - torch.abs(y - center)
            region_probability = torch.sigmoid(self.interface_sharpness * signed_distance)
            class_probs = torch.cat((1.0 - region_probability, region_probability), dim=1)
            geometry = {"weak_center": center, "weak_half_thickness": half_thickness}
            return signed_distance, class_probs, geometry

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


def make_output_transform(lambda_floor, mu_floor, k_floor=0.2, parameterization="bulkmu", method=None, num_loads=1):
    def output_transform(inputs, outputs):
        state_width = 5 * int(max(num_loads, 1))
        state_outputs = outputs[:, :state_width]
        if str(parameterization).lower() == "bulkmu":
            bulk = k_floor + F.softplus(outputs[:, state_width:state_width + 1])
            mu = mu_floor + F.softplus(outputs[:, state_width + 1:state_width + 2])
            lmbd = torch.clamp(bulk - (2.0 / 3.0) * mu, min=lambda_floor)
        else:
            lmbd = lambda_floor + F.softplus(outputs[:, state_width:state_width + 1])
            mu = mu_floor + F.softplus(outputs[:, state_width + 1:state_width + 2])
        return torch.cat((state_outputs, lmbd, mu), dim=1)

    return output_transform


def count_trainable_parameters(net):
    return int(sum(parameter.numel() for parameter in net.parameters() if parameter.requires_grad))


def is_compact_material_method(method):
    return method in {"iaminn_v2", "geoiaminn", "geoiaminn_v3", "twobranch_compact_kmu"}


def build_network(args):
    num_loads = compact_num_loads(args)
    material_hidden_layers = parse_hidden_layers(getattr(args, "material_layers", args.hidden_layers))
    if args.method == "iaminn_v2":
        net = InterfaceAwareMaterialNetV2(
            case_name=args.case,
            state_hidden_layers=parse_hidden_layers(args.state_layers),
            interface_hidden_layers=parse_hidden_layers(args.interface_layers),
            num_loads=num_loads,
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
            num_loads=num_loads,
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
    elif args.method == "twobranch_compact_kmu":
        net = TwoBranchCompactMaterialFieldNet(
            state_hidden_layers=parse_hidden_layers(args.state_layers),
            material_hidden_layers=material_hidden_layers,
            num_loads=num_loads,
            activation=args.activation,
            num_frequencies=args.num_frequencies,
            lambda_floor=args.lambda_floor,
            mu_floor=args.mu_floor,
            k_floor=args.k_floor,
            parameterization=args.material_parameterization,
            backbone_type=args.backbone_type,
        )
    elif args.method == "twobranch_stress_kmu":
        net = TwoBranchStressMaterialFieldNet(
            state_hidden_layers=parse_hidden_layers(args.state_layers),
            material_hidden_layers=material_hidden_layers,
            num_loads=num_loads,
            activation=args.activation,
            num_frequencies=args.num_frequencies,
            lambda_floor=args.lambda_floor,
            mu_floor=args.mu_floor,
            k_floor=args.k_floor,
            parameterization=args.material_parameterization,
            backbone_type=args.backbone_type,
        )
    elif args.method == "threebranch_stress_kmu":
        net = ThreeBranchStressMaterialFieldNet(
            state_hidden_layers=parse_hidden_layers(args.state_layers),
            material_hidden_layers=material_hidden_layers,
            interface_hidden_layers=parse_hidden_layers(args.interface_layers),
            num_loads=num_loads,
            activation=args.activation,
            num_frequencies=args.num_frequencies,
            lambda_floor=args.lambda_floor,
            mu_floor=args.mu_floor,
            k_floor=args.k_floor,
            parameterization=args.material_parameterization,
            interface_sharpness=args.interface_sharpness,
            backbone_type=args.backbone_type,
        )
    elif args.method == "fivestate_stress_kmu":
        net = FiveStateMLPStressMaterialFieldNet(
            state_hidden_layers=parse_hidden_layers(args.state_layers),
            material_hidden_layers=material_hidden_layers,
            num_loads=num_loads,
            activation=args.activation,
            num_frequencies=args.num_frequencies,
            lambda_floor=args.lambda_floor,
            mu_floor=args.mu_floor,
            k_floor=args.k_floor,
            parameterization=args.material_parameterization,
            backbone_type=args.backbone_type,
        )
    elif args.method == "pfnn_scalar_kmu":
        net = PFNNScalarMaterialFieldNet(
            state_hidden_layers=parse_hidden_layers(args.state_layers),
            num_loads=num_loads,
            activation=args.activation,
            num_frequencies=args.num_frequencies,
            lambda_floor=args.lambda_floor,
            mu_floor=args.mu_floor,
            k_floor=args.k_floor,
            parameterization=args.material_parameterization,
        )
    else:
        net = VanillaMaterialFieldNet(
            hidden_layers=parse_hidden_layers(args.hidden_layers),
            output_dim=5 * num_loads + 2,
            activation=args.activation,
            num_frequencies=args.num_frequencies,
            backbone_type=args.backbone_type,
        )
        net.apply_output_transform(
            make_output_transform(
                args.lambda_floor,
                args.mu_floor,
                k_floor=args.k_floor,
                parameterization=args.material_parameterization,
                method=args.method,
                num_loads=num_loads,
            )
        )
    return net



def build_pde(case_config, reg_weight, method=None, load_scales=None, load_modes=None):
    def pde(x, y):
        load_specs = resolve_load_specs(
            load_scales if load_scales is not None else "1.0",
            load_modes if load_modes is not None else "",
        )
        num_loads = len(load_specs)
        if is_compact_material_method(method):
            lambda_idx = 2 * num_loads
            mu_idx = lambda_idx + 1
            lmbd = y[:, lambda_idx:lambda_idx + 1]
            mu = y[:, mu_idx:mu_idx + 1]
            residuals = []
            for load_index, load_spec in enumerate(load_specs):
                load_scale = float(load_spec["scale"])
                load_mode = str(load_spec["mode"])
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
                fx, fy = exact_body_force_torch(x, case_config, load_scale=load_scale, load_mode=load_mode)
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

        lambda_idx, mu_idx = 5 * num_loads, 5 * num_loads + 1
        lmbd = y[:, lambda_idx:lambda_idx + 1]
        mu = y[:, mu_idx:mu_idx + 1]
        residuals = []
        for load_index, load_spec in enumerate(load_specs):
            load_scale = float(load_spec["scale"])
            load_mode = str(load_spec["mode"])
            offset = 5 * load_index
            ux = y[:, offset:offset + 1]
            uy = y[:, offset + 1:offset + 2]
            sxx = y[:, offset + 2:offset + 3]
            syy = y[:, offset + 3:offset + 4]
            sxy = y[:, offset + 4:offset + 5]

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

            sxx_x = dde.grad.jacobian(sxx, x, i=0, j=0)
            syy_y = dde.grad.jacobian(syy, x, i=0, j=1)
            sxy_x = dde.grad.jacobian(sxy, x, i=0, j=0)
            sxy_y = dde.grad.jacobian(sxy, x, i=0, j=1)
            fx, fy = exact_body_force_torch(x, case_config, load_scale=load_scale, load_mode=load_mode)
            residuals.extend(
                [
                    sxx_x + sxy_y + fx,
                    sxy_x + syy_y + fy,
                    sxx - constitutive_sxx,
                    syy - constitutive_syy,
                    sxy - constitutive_sxy,
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

    return pde


def pde_loss_names(reg_weight, method=None, load_scales=None, load_modes=None):
    load_specs = resolve_load_specs(
        load_scales if load_scales is not None else "1.0",
        load_modes if load_modes is not None else "",
    )
    if is_compact_material_method(method):
        names = []
        for load_index in range(len(load_specs)):
            names.extend([f"momentum_x_l{load_index}", f"momentum_y_l{load_index}"])
    else:
        if len(load_specs) == 1:
            names = ["momentum_x", "momentum_y", "constitutive_xx", "constitutive_yy", "constitutive_xy"]
        else:
            names = []
            for load_index in range(len(load_specs)):
                names.extend(
                    [
                        f"momentum_x_l{load_index}",
                        f"momentum_y_l{load_index}",
                        f"constitutive_xx_l{load_index}",
                        f"constitutive_yy_l{load_index}",
                        f"constitutive_xy_l{load_index}",
                    ]
                )
    if reg_weight > 0.0:
        names.extend(["lambda_x", "lambda_y", "mu_x", "mu_y"])
    return names


def pde_loss_weights(reg_weight, method=None, load_scales=None, load_modes=None):
    load_specs = resolve_load_specs(
        load_scales if load_scales is not None else "1.0",
        load_modes if load_modes is not None else "",
    )
    if is_compact_material_method(method):
        weights = []
        for _ in load_specs:
            weights.extend([1.0, 1.0])
    else:
        weights = []
        for _ in load_specs:
            weights.extend([1.0, 1.0, 1.0, 1.0, 1.0])
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
    load_specs = resolve_load_specs(
        getattr(args, "load_scales", "1.0"),
        getattr(args, "load_modes", ""),
    )
    load_balance = compute_load_balance_profile(args)
    weights = []
    if is_compact_material_method(args.method):
        for load_index in range(len(load_specs)):
            load_weight = float(load_balance["physics"][load_index])
            weights.extend([float(physics_scale) * load_weight, float(physics_scale) * load_weight])
    else:
        for load_index in range(len(load_specs)):
            load_weight = float(load_balance["physics"][load_index])
            weights.extend([float(physics_scale) * load_weight] * 5)
    if args.reg_weight > 0.0:
        weights.extend([float(args.reg_weight) * float(reg_scale)] * 4)
    num_loads = compact_num_loads(args)
    for load_index in range(num_loads):
        boundary_weight = float(args.boundary_weight) * float(boundary_scale) * float(load_balance["boundary"][load_index])
        data_weight = float(args.data_weight) * float(data_scale) * float(load_balance["data"][load_index])
        weights.extend(
            [
                boundary_weight,
                boundary_weight,
                data_weight,
                data_weight,
            ]
        )
        if reaction_consistency_enabled(args) and not is_compact_material_method(args.method):
            reaction_weight = (
                float(getattr(args, "reaction_weight", 0.0))
                * float(boundary_scale)
                * float(load_balance["boundary"][load_index])
            )
            weights.extend([reaction_weight, reaction_weight])
    return weights


def build_observation_payload(
    points,
    case_config,
    noise_level,
    seed,
    load_scale=1.0,
    load_mode="legacy",
    noise_scale_mode="component_std",
):
    if len(points) == 0:
        return {
            "points": np.zeros((0, 2), dtype=float),
            "clean": np.zeros((0, 2), dtype=float),
            "noisy": np.zeros((0, 2), dtype=float),
            "true_state": np.zeros((0, len(FIELD_NAMES)), dtype=float),
            "load_scale": float(load_scale),
            "load_mode": str(load_mode),
            "noise_scale_mode": str(noise_scale_mode),
        }
    exact_observation_state = exact_state_numpy(points, case_config, load_scale=load_scale, load_mode=load_mode)
    clean_observation = exact_observation_state[:, :2]
    noisy_observation = add_noise(clean_observation, noise_level, seed, scale_mode=noise_scale_mode)
    return {
        "points": points,
        "clean": clean_observation,
        "noisy": noisy_observation,
        "true_state": exact_observation_state,
        "load_scale": float(load_scale),
        "load_mode": str(load_mode),
        "noise_scale_mode": str(noise_scale_mode),
    }


def build_data(args, case_config):
    geom = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])
    load_specs = resolve_load_specs(getattr(args, "load_scales", "1.0"), getattr(args, "load_modes", ""))
    teacher_source = str(getattr(args, "teacher_source", "analytic")).lower()
    reference_data = None

    if teacher_source == "fem":
        reference_data = load_fem_teacher_reference(args.teacher_run_dir)
        teacher_specs = [
            {"scale": float(scale), "mode": str(mode)}
            for scale, mode in zip(reference_data["load_scales"], reference_data["load_modes"])
        ]
        if teacher_specs != load_specs:
            raise ValueError(
                "Configured load specs do not match FEM teacher bundle. "
                f"args={load_specs}, teacher={teacher_specs}"
            )
        boundary_points = np.asarray(reference_data["boundary_points"], dtype=float)
        boundary_observation_loads = [
            {
                "points": np.asarray(payload["points"], dtype=float),
                "clean": np.asarray(payload["clean"], dtype=float),
                "noisy": np.asarray(payload["noisy"], dtype=float),
                "true_state": np.asarray(payload["true_state"], dtype=float),
                "load_scale": float(load_specs[load_index]["scale"]),
                "load_mode": str(load_specs[load_index]["mode"]),
            }
            for load_index, payload in enumerate(reference_data["boundary_payloads"])
        ]
        train_observation_loads = [
            {
                "points": np.asarray(payload["points"], dtype=float),
                "clean": np.asarray(payload["clean"], dtype=float),
                "noisy": np.asarray(payload["noisy"], dtype=float),
                "true_state": np.asarray(payload["true_state"], dtype=float),
                "load_scale": float(load_specs[load_index]["scale"]),
                "load_mode": str(load_specs[load_index]["mode"]),
            }
            for load_index, payload in enumerate(reference_data["train_payloads"])
        ]
        val_observation_loads = [
            {
                "points": np.asarray(payload["points"], dtype=float),
                "clean": np.asarray(payload["clean"], dtype=float),
                "noisy": np.asarray(payload["noisy"], dtype=float),
                "true_state": np.asarray(payload["true_state"], dtype=float),
                "load_scale": float(load_specs[load_index]["scale"]),
                "load_mode": str(load_specs[load_index]["mode"]),
            }
            for load_index, payload in enumerate(reference_data["validation_payloads"])
        ]
        eval_observation_loads = [
            {
                "points": np.asarray(payload["points"], dtype=float),
                "clean": np.asarray(payload["clean"], dtype=float),
                "noisy": np.asarray(payload["noisy"], dtype=float),
                "true_state": np.asarray(payload["true_state"], dtype=float),
                "load_scale": float(load_specs[load_index]["scale"]),
                "load_mode": str(load_specs[load_index]["mode"]),
            }
            for load_index, payload in enumerate(reference_data["evaluation_payloads"])
        ]
        if len(boundary_points) != int(args.num_boundary):
            raise ValueError(
                f"num_boundary mismatch: args={args.num_boundary}, teacher={len(boundary_points)}"
            )
        for split_name, payloads, expected_count in [
            ("train", train_observation_loads, int(args.num_observe)),
            ("validation", val_observation_loads, int(args.num_val_observe)),
            ("evaluation", eval_observation_loads, int(args.num_eval_observe)),
        ]:
            for payload in payloads:
                if len(payload["points"]) != expected_count:
                    raise ValueError(
                        f"{split_name} observation count mismatch: args={expected_count}, teacher={len(payload['points'])}"
                    )
        reaction_targets = []
        if reaction_consistency_enabled(args) and not is_compact_material_method(args.method):
            for payload in boundary_observation_loads:
                reaction_payload = {
                    "points": np.asarray(payload["points"], dtype=float),
                }
                edge_points, normals, weights = select_boundary_edge(
                    reaction_payload["points"],
                    edge=getattr(args, "reaction_edge", "top"),
                )
                if len(edge_points) == 0:
                    reaction_targets.append(
                        {
                            "points": edge_points,
                            "normals": normals,
                            "weights": weights,
                            "target_fx": 0.0,
                            "target_fy": 0.0,
                            "edge": str(getattr(args, "reaction_edge", "top")),
                        }
                    )
                    continue
                point_lookup = {
                    tuple(np.round(point, 8).tolist()): idx for idx, point in enumerate(np.asarray(payload["points"], dtype=float))
                }
                indices = [point_lookup[tuple(np.round(point, 8).tolist())] for point in edge_points]
                state = np.asarray(payload["true_state"], dtype=float)[indices]
                sxx = state[:, 2:3]
                syy = state[:, 3:4]
                sxy = state[:, 4:5]
                nx = normals[:, 0:1]
                ny = normals[:, 1:2]
                tx = sxx * nx + sxy * ny
                ty = sxy * nx + syy * ny
                reaction_targets.append(
                    {
                        "points": edge_points,
                        "normals": normals,
                        "weights": weights,
                        "target_fx": float(np.sum(tx * weights)),
                        "target_fy": float(np.sum(ty * weights)),
                        "edge": str(getattr(args, "reaction_edge", "top")),
                    }
                )
    else:
        split_seed = getattr(args, "observation_split_seed", None)
        if split_seed is None:
            split_seed = args.seed + 17
        split_seed = int(split_seed)
        observation_splits = build_observation_splits(
            num_train=args.num_observe,
            num_val=args.num_val_observe,
            num_eval=args.num_eval_observe,
            seed=split_seed,
            case_name=args.case,
            case_config=case_config,
            cache_dir=args.observation_cache_dir,
            tag=args.observation_split_tag,
        )
        boundary_points = build_boundary_points(args.num_boundary)
        train_observation_loads = []
        val_observation_loads = []
        eval_observation_loads = []
        boundary_observation_loads = []
        reaction_targets = []
        for load_index, load_spec in enumerate(load_specs):
            load_scale = float(load_spec["scale"])
            load_mode = str(load_spec["mode"])
            train_observation_loads.append(
                build_observation_payload(
                    observation_splits["train"],
                    case_config,
                    args.noise_level,
                    args.seed + 123 + 1000 * load_index,
                    load_scale=load_scale,
                    load_mode=load_mode,
                    noise_scale_mode=getattr(args, "noise_scale_mode", "component_std"),
                )
            )
            val_observation_loads.append(
                build_observation_payload(
                    observation_splits["val"],
                    case_config,
                    args.noise_level,
                    args.seed + 223 + 1000 * load_index,
                    load_scale=load_scale,
                    load_mode=load_mode,
                    noise_scale_mode=getattr(args, "noise_scale_mode", "component_std"),
                )
            )
            eval_observation_loads.append(
                build_observation_payload(
                    observation_splits["eval"],
                    case_config,
                    args.noise_level,
                    args.seed + 323 + 1000 * load_index,
                    load_scale=load_scale,
                    load_mode=load_mode,
                    noise_scale_mode=getattr(args, "noise_scale_mode", "component_std"),
                )
            )
            boundary_observation_loads.append(
                build_observation_payload(
                    boundary_points,
                    case_config,
                    0.0,
                    args.seed + 423 + 1000 * load_index,
                    load_scale=load_scale,
                    load_mode=load_mode,
                    noise_scale_mode=getattr(args, "noise_scale_mode", "component_std"),
                )
            )
            if reaction_consistency_enabled(args) and not is_compact_material_method(args.method):
                reaction_targets.append(
                    compute_total_reaction_target(
                        boundary_points,
                        case_config,
                        load_scale=load_scale,
                        load_mode=load_mode,
                        edge=getattr(args, "reaction_edge", "top"),
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
        if reaction_consistency_enabled(args) and not is_compact_material_method(args.method):
            reaction_payload = reaction_targets[load_index]
            reaction_points = np.asarray(reaction_payload["points"], dtype=float)
            if len(reaction_points) > 0:
                fx_values = np.full((len(reaction_points), 1), float(reaction_payload["target_fx"]), dtype=float)
                fy_values = np.full((len(reaction_points), 1), float(reaction_payload["target_fy"]), dtype=float)
                bcs.extend(
                    [
                        dde.icbc.PointSetOperatorBC(
                            reaction_points,
                            fx_values,
                            build_reaction_operator(
                                reaction_payload["points"],
                                reaction_payload["normals"],
                                reaction_payload["weights"],
                                args=args,
                                load_index=load_index,
                                component="x",
                            ),
                        ),
                        dde.icbc.PointSetOperatorBC(
                            reaction_points,
                            fy_values,
                            build_reaction_operator(
                                reaction_payload["points"],
                                reaction_payload["normals"],
                                reaction_payload["weights"],
                                args=args,
                                load_index=load_index,
                                component="y",
                            ),
                        ),
                    ]
                )
    data = dde.data.PDE(
        geom,
        build_pde(
            case_config,
            args.reg_weight,
            args.method,
            load_scales=getattr(args, "load_scales", "1.0"),
            load_modes=getattr(args, "load_modes", ""),
        ),
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
        "load_scales": [float(spec["scale"]) for spec in load_specs],
        "load_modes": [str(spec["mode"]) for spec in load_specs],
        "observation_split_seed": (
            reference_data.get("observation_split_seed")
            if isinstance(reference_data, dict) and reference_data.get("source") == "fem"
            else split_seed
        ),
        "observation_split_tag": (
            reference_data.get("observation_split_tag")
            if isinstance(reference_data, dict) and reference_data.get("source") == "fem"
            else str(args.observation_split_tag)
        ),
        "observation_cache_path": (
            reference_data.get("observation_cache_path")
            if isinstance(reference_data, dict) and reference_data.get("source") == "fem"
            else observation_splits.get("cache_path")
        ),
        "reaction_targets": reaction_targets,
        "observation_points": [payload["points"] for payload in eval_observation_loads],
        "observation_clean": [payload["clean"] for payload in eval_observation_loads],
        "reference_data": reference_data,
        "teacher_source": teacher_source,
        "teacher_run_dir": reference_data.get("run_dir") if isinstance(reference_data, dict) else None,
    }
    args._reference_data = reference_data
    args._cached_load_balance_profile = None
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
        "twobranch_compact_kmu": "TwoBranch-Compact-KMu",
        "twobranch_stress_kmu": "TwoBranch-Stress-KMu",
        "threebranch_stress_kmu": "ThreeBranch-Stress-KMu",
        "fivestate_stress_kmu": "FiveState-Stress-KMu",
        "pfnn_scalar_kmu": "PFNN-Scalar-KMu",
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


def load_json_if_exists(path, default=None):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def _merge_loss_history_payloads(existing, current):
    merged_steps = []
    merged_train = []
    merged_test = []

    def add_row(step, train_row, test_row):
        step = int(step)
        train_row = list(train_row) if train_row is not None else []
        test_row = list(test_row) if test_row is not None else []
        while merged_steps and merged_steps[-1] > step:
            merged_steps.pop()
            merged_train.pop()
            merged_test.pop()
        if merged_steps and merged_steps[-1] == step:
            merged_train[-1] = train_row
            merged_test[-1] = test_row
            return
        merged_steps.append(step)
        merged_train.append(train_row)
        merged_test.append(test_row)

    existing_steps = existing.get("steps", []) if existing else []
    existing_train = existing.get("loss_train", []) if existing else []
    existing_test = existing.get("loss_test", []) if existing else []
    for idx, step in enumerate(existing_steps):
        add_row(
            step,
            existing_train[idx] if idx < len(existing_train) else [],
            existing_test[idx] if idx < len(existing_test) else [],
        )

    current_steps = list(getattr(current, "steps", []) or [])
    current_train = list(getattr(current, "loss_train", []) or [])
    current_test = list(getattr(current, "loss_test", []) or [])
    for idx, step in enumerate(current_steps):
        add_row(
            step,
            current_train[idx] if idx < len(current_train) else [],
            current_test[idx] if idx < len(current_test) else [],
        )

    return SimpleNamespace(
        steps=merged_steps,
        loss_train=merged_train,
        loss_test=merged_test,
    )


def _max_series_step(payload, step_key="steps"):
    if not isinstance(payload, dict):
        return None
    steps = payload.get(step_key)
    if not steps:
        return None
    try:
        return int(max(int(step) for step in steps))
    except Exception:
        return None


def _max_material_probe_step(payload):
    if not isinstance(payload, dict):
        return None
    history = payload.get("history")
    if not history:
        return None
    try:
        return int(max(int(row.get("step", 0)) for row in history))
    except Exception:
        return None


def infer_resume_artifact_step(save_dir):
    candidates = []
    loss_payload = load_json_if_exists(_get_save_path(save_dir, "json", "loss_history.json"))
    validation_payload = load_json_if_exists(_get_save_path(save_dir, "json", "validation_history.json"))
    material_payload = load_json_if_exists(_get_save_path(save_dir, "json", "material_parameter_evolution.json"))
    for value in (
        _max_series_step(loss_payload),
        _max_series_step(validation_payload),
        _max_material_probe_step(material_payload),
    ):
        if value is not None:
            candidates.append(int(value))
    return max(candidates) if candidates else 0


def _trim_series_payload(payload, max_step, value_keys):
    if not isinstance(payload, dict):
        return payload, False
    steps = [int(step) for step in payload.get("steps", [])]
    if not steps:
        return payload, False
    keep_count = 0
    for step in steps:
        if int(step) <= int(max_step):
            keep_count += 1
        else:
            break
    if keep_count == len(steps):
        return payload, False
    trimmed = dict(payload)
    trimmed["steps"] = steps[:keep_count]
    for key in value_keys:
        values = list(payload.get(key, []) or [])
        trimmed[key] = values[:keep_count]
    return trimmed, True


def _trim_material_probe_payload(payload, max_step):
    if not isinstance(payload, dict):
        return payload, False
    history = list(payload.get("history", []) or [])
    keep_history = [row for row in history if int(row.get("step", 0)) <= int(max_step)]
    changed = len(keep_history) != len(history)
    trimmed = dict(payload)
    trimmed["history"] = keep_history
    snapshots = dict(payload.get("snapshots", {}) or {})
    kept_snapshots = {}
    for tag, row in snapshots.items():
        try:
            step = int(row.get("step", 0))
        except Exception:
            step = 0
        if step <= int(max_step):
            kept_snapshots[tag] = row
    if len(kept_snapshots) != len(snapshots):
        changed = True
    trimmed["snapshots"] = kept_snapshots
    return trimmed, changed


def trim_resume_histories(save_dir, max_step):
    trimmed_any = False
    loss_path = _get_save_path(save_dir, "json", "loss_history.json")
    loss_payload = load_json_if_exists(loss_path)
    trimmed_loss, changed = _trim_series_payload(loss_payload, max_step, ("loss_train", "loss_test"))
    if changed:
        save_json(loss_path, trimmed_loss)
        trimmed_any = True

    validation_path = _get_save_path(save_dir, "json", "validation_history.json")
    validation_payload = load_json_if_exists(validation_path)
    trimmed_validation, changed = _trim_series_payload(
        validation_payload,
        max_step,
        ("validation_observation_mse",),
    )
    if changed:
        save_json(validation_path, trimmed_validation)
        trimmed_any = True

    best_info_path = _get_save_path(save_dir, "json", "best_validation_info.json")
    best_info = load_json_if_exists(best_info_path)
    if isinstance(best_info, dict) and int(best_info.get("step", 0) or 0) > int(max_step):
        if isinstance(trimmed_validation, dict) and trimmed_validation.get("steps") and trimmed_validation.get("validation_observation_mse"):
            steps = [int(step) for step in trimmed_validation["steps"]]
            values = [float(value) for value in trimmed_validation["validation_observation_mse"]]
            best_index = int(np.argmin(values))
            best_info["step"] = int(steps[best_index])
            best_info["best_value"] = float(values[best_index])
        else:
            best_info["step"] = 0
            best_info["best_value"] = float("inf")
            best_info["model_path"] = None
        save_json(best_info_path, best_info)
        trimmed_any = True

    material_path = _get_save_path(save_dir, "json", "material_parameter_evolution.json")
    material_payload = load_json_if_exists(material_path)
    trimmed_material, changed = _trim_material_probe_payload(material_payload, max_step)
    if changed:
        save_json(material_path, trimmed_material)
        trimmed_any = True
    return trimmed_any


def merge_losshistory_with_existing(losshistory, save_dir):
    existing = load_json_if_exists(_get_save_path(save_dir, "json", "loss_history.json"))
    if not existing or not existing.get("steps"):
        return losshistory
    return _merge_loss_history_payloads(existing, losshistory)

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
    parameterization="lamemu",
):
    grid_points, xx, yy = make_grid(200, 200)
    exact_fields = exact_state_numpy(grid_points, case_config)
    lambda_field = exact_fields[:, 5].reshape(xx.shape)
    mu_field = exact_fields[:, 6].reshape(xx.shape)
    parameterization = str(parameterization).lower()
    material_field_a = lambda_field
    material_field_b = mu_field
    material_label_a = r"True $\lambda(x,y)$"
    material_label_b = r"True $\mu(x,y)$"
    if parameterization == "bulkmu":
        material_field_a = bulk_from_lambda_mu(lambda_field, mu_field)
        material_label_a = r"True $K(x,y)$"
        material_label_b = r"True $\mu(x,y)$"

    plot_case_sampling_layout(
        save_dir=save_dir,
        xx=xx,
        yy=yy,
        material_field_a=material_field_a,
        material_field_b=material_field_b,
        material_label_a=material_label_a,
        material_label_b=material_label_b,
        lambda_field=lambda_field,
        mu_field=mu_field,
        domain_points=domain_points,
        boundary_points=boundary_points,
        train_observation_points=train_observation_points,
        val_observation_points=val_observation_points,
        eval_observation_points=eval_observation_points,
        title_suffix=title_suffix,
        filename=filename,
    )


def save_training_artifacts(

    args,
    save_dir,
    case_config,
    losshistory,
    metadata,
    net,
):
    region_payload = None
    save_run_config_artifacts(args, save_dir, case_config, net=net, metadata=metadata)
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
            parameterization=getattr(args, "material_parameterization", "lamemu"),
        )

    bc_loss_names = build_bc_loss_names(args)
    if losshistory is not None:
        effective_losshistory = (
            merge_losshistory_with_existing(losshistory, save_dir)
            if bool(getattr(args, "resume_training", False) or getattr(args, "model_path", None))
            else losshistory
        )
        plot_and_save_loss_history(
            effective_losshistory,
            save_dir,
            filename="loss_history.png",
            num_pde_losses=len(
                pde_loss_names(
                    args.reg_weight,
                    args.method,
                    getattr(args, "load_scales", "1.0"),
                    getattr(args, "load_modes", ""),
                )
            ),
            num_bc_losses=len(bc_loss_names),
            pde_label="Physics Loss",
            bc_label="Boundary Loss",
            bc_loss_names=bc_loss_names,
            data_loss_prefix="obs_",
            data_label="Observation Loss",
            show_total=False,
        )
        plot_all_loss_components(
            effective_losshistory,
            save_dir,
            filename="loss_components.png",
            num_pde_losses=len(
                pde_loss_names(
                    args.reg_weight,
                    args.method,
                    getattr(args, "load_scales", "1.0"),
                    getattr(args, "load_modes", ""),
                )
            ),
            num_bc_losses=len(bc_loss_names),
            pde_loss_names=pde_loss_names(
                args.reg_weight,
                args.method,
                getattr(args, "load_scales", "1.0"),
                getattr(args, "load_modes", ""),
            ),
            bc_loss_names=bc_loss_names,
        )
        save_loss_history_json(effective_losshistory, save_dir, filename="loss_history.json")
        save_best_test_loss_json(effective_losshistory, save_dir, filename="best_test_loss.json")
        save_loss_history_dat(effective_losshistory, save_dir)

    if args.method == "iaminn_v2" and hasattr(net, "predict_material_diagnostics"):
        with torch.no_grad():
            lambda_regions = net.lambda_floor + F.softplus(net.raw_lambda_params)
            mu_regions = net.mu_floor + F.softplus(net.raw_mu_params)
        region_payload = {
            "parameterization": "lamemu",
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
            "parameterization": str(getattr(args, "material_parameterization", "lamemu")),
            "lambda_regions": [float(v) for v in diagnostics["lambda_regions"].detach().cpu().numpy().tolist()],
            "mu_regions": [float(v) for v in diagnostics["mu_regions"].detach().cpu().numpy().tolist()],
            "num_regions": int(net.num_regions),
        }
        if "bulk_regions" in diagnostics:
            region_payload["bulk_regions"] = [float(v) for v in diagnostics["bulk_regions"].detach().cpu().numpy().tolist()]
        for key, value in diagnostics.items():
            if key in {"bulk", "lambda", "mu", "interface_indicator", "class_probs", "bulk_regions", "lambda_regions", "mu_regions"}:
                continue
            if torch.is_tensor(value) and value.numel() == 1:
                region_payload[key] = float(value.detach().cpu().item())
        save_json(_get_save_path(save_dir, "json", "material_region_parameters.json"), region_payload)
        save_metrics_text(_get_save_path(save_dir, "txt", "material_region_parameters.txt"), region_payload)

    if region_payload is not None:
        plot_region_parameter_evolution(
            save_dir=save_dir,
            case_config=case_config,
            total_iterations=int(getattr(args, "iterations", 0)),
            final_region_payload=region_payload,
        )


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


def compute_case_zone_metrics(points, prediction, truth, case_config, parameterization="lamemu"):
    if points is None or case_config is None:
        return {}

    x = points[:, 0]
    y = points[:, 1]
    bulk_prediction = bulk_from_lambda_mu(prediction[:, 5:6], prediction[:, 6:7])[:, 0]
    bulk_truth = bulk_from_lambda_mu(truth[:, 5:6], truth[:, 6:7])[:, 0]
    mu_prediction = prediction[:, 6]
    mu_truth = truth[:, 6]
    lambda_mu_error = np.linalg.norm(prediction[:, 5:7] - truth[:, 5:7], axis=1)
    bulk_mu_error = np.linalg.norm(
        np.stack([bulk_prediction - bulk_truth, mu_prediction - mu_truth], axis=1),
        axis=1,
    )

    if case_config.name == "single_inclusion":
        radius = float(case_config.circle_1_r)
        band = max(0.03, 0.2 * radius)
        distance = np.sqrt((x - float(case_config.circle_1_cx)) ** 2 + (y - float(case_config.circle_1_cy)) ** 2)
        masks = {
            "zone_background": distance >= radius + band,
            "zone_inclusion": distance <= max(radius - band, 0.02),
            "zone_interface": np.abs(distance - radius) < band,
        }
        score_weights = (0.5, 0.3, 0.2)
        feature_key = "zone_inclusion_material_err_bulk_mu"
    elif case_config.name == "layered":
        band = 0.04
        indicator = y - float(case_config.layer_y)
        masks = {
            "zone_background": indicator <= -band,
            "zone_layer": indicator >= band,
            "zone_interface": np.abs(indicator) < band,
        }
        score_weights = (0.6, 0.25, 0.15)
        feature_key = "zone_layer_material_err_bulk_mu"
    elif case_config.name == "weak_interlayer":
        band = max(0.03, float(case_config.interface_width) * 2.0)
        y_bottom = float(case_config.weak_y_bottom)
        y_top = float(case_config.weak_y_top)
        masks = {
            "zone_lower_soil": y <= y_bottom - band,
            "zone_weak_interlayer": (y >= y_bottom) & (y <= y_top),
            "zone_interface": ((y > y_bottom - band) & (y < y_bottom + band))
            | ((y > y_top - band) & (y < y_top + band)),
            "zone_upper_soil": y >= y_top + band,
        }
        score_weights = (0.45, 0.35, 0.2)
        feature_key = "zone_weak_interlayer_material_err_bulk_mu"
    elif case_config.name == "single_material":
        masks = {"zone_global": np.ones_like(x, dtype=bool)}
        score_weights = (1.0, 0.0, 0.0)
        feature_key = "zone_global_material_err_bulk_mu"
    else:
        return {}

    metrics = {}
    for zone_name, mask in masks.items():
        if not np.any(mask):
            continue
        metrics[f"{zone_name}_point_count"] = int(np.sum(mask))
        metrics[f"{zone_name}_bulk_mae"] = float(np.mean(np.abs(bulk_prediction[mask] - bulk_truth[mask])))
        metrics[f"{zone_name}_mu_mae"] = float(np.mean(np.abs(mu_prediction[mask] - mu_truth[mask])))
        metrics[f"{zone_name}_material_err_lambda_mu"] = float(np.mean(lambda_mu_error[mask]))
        metrics[f"{zone_name}_material_err_bulk_mu"] = float(np.mean(bulk_mu_error[mask]))

    global_error = float(np.mean(bulk_mu_error if str(parameterization).lower() == "bulkmu" else lambda_mu_error))
    feature_error = metrics.get(feature_key, global_error)
    interface_error = metrics.get("zone_interface_material_err_bulk_mu", global_error)
    metrics["protocol_search_score"] = float(
        score_weights[0] * global_error
        + score_weights[1] * feature_error
        + score_weights[2] * interface_error
    )
    return metrics


def compute_metrics(
    prediction,
    truth,
    observation_prediction,
    observation_truth,
    pde_residual,
    parameterization="lamemu",
    points=None,
    case_config=None,
):
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
    bulk_prediction = bulk_from_lambda_mu(prediction[:, 5:6], prediction[:, 6:7])
    bulk_truth = bulk_from_lambda_mu(truth[:, 5:6], truth[:, 6:7])
    metrics["field_relative_l2"]["bulk"] = relative_l2(bulk_prediction, bulk_truth)
    metrics["field_mae"]["bulk"] = mean_abs(bulk_prediction, bulk_truth)

    lambda_mu_error = np.linalg.norm(prediction[:, 5:7] - truth[:, 5:7], axis=1)
    bulk_mu_error = np.linalg.norm(
        np.concatenate([bulk_prediction, prediction[:, 6:7]], axis=1)
        - np.concatenate([bulk_truth, truth[:, 6:7]], axis=1),
        axis=1,
    )
    metrics["material_field_mean_abs_vector_error_lambda_mu"] = float(np.mean(lambda_mu_error))
    metrics["material_field_mean_abs_vector_error_bulk_mu"] = float(np.mean(bulk_mu_error))
    if str(parameterization).lower() == "bulkmu":
        metrics["material_field_mean_abs_vector_error"] = float(np.mean(bulk_mu_error))
    else:
        metrics["material_field_mean_abs_vector_error"] = float(np.mean(lambda_mu_error))
    metrics.update(
        compute_case_zone_metrics(
            points=points,
            prediction=prediction,
            truth=truth,
            case_config=case_config,
            parameterization=parameterization,
        )
    )
    return metrics


def split_residual_by_load(args, residual_matrix):
    residual_array = split_residual_prediction(residual_matrix)
    num_loads = compact_num_loads(args)
    block_width = 2 if is_compact_material_method(args.method) else 5
    load_blocks = []
    for load_index in range(num_loads):
        start = load_index * block_width
        end = start + block_width
        load_blocks.append(residual_array[:, start:end])
    reg_block = residual_array[:, block_width * num_loads :] if residual_array.shape[1] > block_width * num_loads else np.zeros((residual_array.shape[0], 0), dtype=float)
    return residual_array, load_blocks, reg_block


def aggregate_multiload_metrics(
    args,
    case_config,
    eval_points,
    residual_matrix,
    predictions_by_load,
    truths_by_load,
    observation_predictions_by_load,
    observation_truths_by_load,
):
    parameterization = getattr(args, "material_parameterization", "lamemu")
    residual_full, residual_by_load, residual_reg = split_residual_by_load(args, residual_matrix)
    metrics = {
        "field_relative_l2": {},
        "field_mae": {},
        "observation_mse": float(
            np.mean(
                [
                    np.mean((pred - truth) ** 2)
                    for pred, truth in zip(observation_predictions_by_load, observation_truths_by_load)
                ]
            )
        ),
        "pde_residual_mean_abs": float(np.mean(np.abs(residual_full))),
        "physics_residual_mean_abs": float(np.mean([np.mean(np.abs(block)) for block in residual_by_load])),
        "reg_residual_mean_abs": float(np.mean(np.abs(residual_reg))) if residual_reg.size else 0.0,
    }

    for field_index, field_name in enumerate(STATE_FIELD_NAMES):
        metrics["field_relative_l2"][field_name] = float(
            np.mean(
                [
                    relative_l2(pred[:, field_index : field_index + 1], truth[:, field_index : field_index + 1])
                    for pred, truth in zip(predictions_by_load, truths_by_load)
                ]
            )
        )
        metrics["field_mae"][field_name] = float(
            np.mean(
                [
                    mean_abs(pred[:, field_index : field_index + 1], truth[:, field_index : field_index + 1])
                    for pred, truth in zip(predictions_by_load, truths_by_load)
                ]
            )
        )

    material_prediction = predictions_by_load[0]
    material_truth = truths_by_load[0]
    for field_index, field_name in enumerate(MATERIAL_FIELD_NAMES, start=len(STATE_FIELD_NAMES)):
        metrics["field_relative_l2"][field_name] = relative_l2(
            material_prediction[:, field_index : field_index + 1],
            material_truth[:, field_index : field_index + 1],
        )
        metrics["field_mae"][field_name] = mean_abs(
            material_prediction[:, field_index : field_index + 1],
            material_truth[:, field_index : field_index + 1],
        )

    bulk_prediction = bulk_from_lambda_mu(material_prediction[:, 5:6], material_prediction[:, 6:7])
    bulk_truth = bulk_from_lambda_mu(material_truth[:, 5:6], material_truth[:, 6:7])
    metrics["field_relative_l2"]["bulk"] = relative_l2(bulk_prediction, bulk_truth)
    metrics["field_mae"]["bulk"] = mean_abs(bulk_prediction, bulk_truth)

    lambda_mu_error = np.linalg.norm(material_prediction[:, 5:7] - material_truth[:, 5:7], axis=1)
    bulk_mu_error = np.linalg.norm(
        np.concatenate([bulk_prediction, material_prediction[:, 6:7]], axis=1)
        - np.concatenate([bulk_truth, material_truth[:, 6:7]], axis=1),
        axis=1,
    )
    metrics["material_field_mean_abs_vector_error_lambda_mu"] = float(np.mean(lambda_mu_error))
    metrics["material_field_mean_abs_vector_error_bulk_mu"] = float(np.mean(bulk_mu_error))
    metrics["material_field_mean_abs_vector_error"] = (
        float(np.mean(bulk_mu_error))
        if str(parameterization).lower() == "bulkmu"
        else float(np.mean(lambda_mu_error))
    )
    metrics.update(
        compute_case_zone_metrics(
            points=eval_points,
            prediction=material_prediction,
            truth=material_truth,
            case_config=case_config,
            parameterization=parameterization,
        )
    )

    per_load = []
    load_specs = resolve_load_specs(getattr(args, "load_scales", "1.0"), getattr(args, "load_modes", ""))
    for load_index, (pred, truth, obs_pred, obs_truth, residual_block) in enumerate(
        zip(predictions_by_load, truths_by_load, observation_predictions_by_load, observation_truths_by_load, residual_by_load)
    ):
        per_load.append(
            {
                "load_index": int(load_index),
                "load_mode": str(load_specs[load_index]["mode"]),
                "load_scale": float(load_specs[load_index]["scale"]),
                "observation_mse": float(np.mean((obs_pred - obs_truth) ** 2)),
                "physics_residual_mean_abs": float(np.mean(np.abs(residual_block))),
                "field_relative_l2": {
                    field_name: relative_l2(pred[:, idx : idx + 1], truth[:, idx : idx + 1])
                    for idx, field_name in enumerate(STATE_FIELD_NAMES)
                },
                "field_mae": {
                    field_name: mean_abs(pred[:, idx : idx + 1], truth[:, idx : idx + 1])
                    for idx, field_name in enumerate(STATE_FIELD_NAMES)
                },
            }
        )
    metrics["per_load"] = per_load
    return metrics


def split_residual_prediction(residual_prediction):
    if isinstance(residual_prediction, list):
        return np.concatenate([np.asarray(item) for item in residual_prediction], axis=1)
    array = np.asarray(residual_prediction)
    if array.ndim == 1:
        return array[:, None]
    return array


def _forward_compact_raw_tensor(net, x, args):
    """Use the model native forward path for compact methods.

    Prediction-time forward must match training-time forward to avoid
    architecture-dependent feature-shape drift in staged training.
    """
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
    raw = np.asarray(model.predict(points))
    num_loads = compact_num_loads(args)
    if num_loads <= 1:
        return raw
    load_index = int(max(0, min(load_index, num_loads - 1)))
    offset = 5 * load_index
    lambda_idx, mu_idx = compact_material_indices(args)
    return np.concatenate(
        [
            raw[:, offset:offset + 5],
            raw[:, lambda_idx:lambda_idx + 1],
            raw[:, mu_idx:mu_idx + 1],
        ],
        axis=1,
    )


def predict_material_components(model, points, args, batch_size=4096):
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return {
            "lambda": np.zeros((0, 1), dtype=float),
            "mu": np.zeros((0, 1), dtype=float),
            "bulk": np.zeros((0, 1), dtype=float),
        }
    if is_compact_material_method(args.method):
        raw = _predict_compact_raw(model, points, args, batch_size=batch_size)
    else:
        raw = np.asarray(model.predict(points))
    lambda_idx, mu_idx = compact_material_indices(args)
    lmbd = np.asarray(raw[:, lambda_idx:lambda_idx + 1], dtype=float)
    mu = np.asarray(raw[:, mu_idx:mu_idx + 1], dtype=float)
    bulk = bulk_from_lambda_mu(lmbd, mu)
    return {"lambda": lmbd, "mu": mu, "bulk": bulk}


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
    plot_validation_history(history, save_dir)


class ValidationObservationCheckpoint(dde.callbacks.Callback):
    def __init__(
        self,
        filepath,
        save_dir,
        observation_points,
        observation_truth_clean,
        args,
        period=1,
        verbose=1,
        append_existing=False,
    ):
        super().__init__()
        self.filepath = filepath
        self.save_dir = save_dir
        if isinstance(observation_points, (list, tuple)):
            self.observation_points = [np.asarray(points, dtype=float) for points in observation_points]
            self.observation_truth_clean = [np.asarray(values, dtype=float) for values in observation_truth_clean]
        else:
            self.observation_points = np.asarray(observation_points, dtype=float)
            self.observation_truth_clean = np.asarray(observation_truth_clean, dtype=float)
        self.args = args
        self.period = period
        self.verbose = verbose
        self.append_existing = bool(append_existing)
        self.epochs_since_last_save = 0
        self.best = np.inf
        self.best_file = None
        self.best_step = 0
        self.validation_target = validation_observation_target_name(args)
        self.history = {
            "metric_name": "validation_observation_mse",
            "validation_target": self.validation_target,
            "steps": [],
            "validation_observation_mse": [],
        }
        if self.append_existing:
            history_path = _get_save_path(self.save_dir, "json", "validation_history.json")
            existing_history = load_json_if_exists(history_path)
            if isinstance(existing_history, dict) and existing_history.get("steps") is not None:
                self.history = existing_history
            best_info_path = _get_save_path(self.save_dir, "json", "best_validation_info.json")
            best_info = load_json_if_exists(best_info_path)
            if isinstance(best_info, dict):
                best_value = best_info.get("best_value")
                if best_value is not None and np.isfinite(best_value):
                    self.best = float(best_value)
                self.best_step = int(best_info.get("step", 0) or 0)
                model_name = best_info.get("model_path")
                if model_name:
                    candidate = os.path.join(os.path.dirname(self.filepath), str(model_name))
                    if os.path.exists(candidate):
                        self.best_file = candidate

    def on_train_begin(self):
        self._truncate_future(int(getattr(self.model.train_state, "iteration", 0)))
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
        self._truncate_future(step)
        if self.history["steps"] and int(self.history["steps"][-1]) == step:
            self.history["validation_observation_mse"][-1] = float(current)
        else:
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
                "target": self.validation_target,
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

    def _truncate_future(self, step):
        step = int(step)
        steps = [int(value) for value in self.history.get("steps", [])]
        values = list(self.history.get("validation_observation_mse", []) or [])
        while steps and steps[-1] > step:
            steps.pop()
            if values:
                values.pop()
        self.history["steps"] = steps
        self.history["validation_observation_mse"] = values
        if int(self.best_step) > step:
            self.best = np.inf
            self.best_step = 0
            self.best_file = None
            if steps and values:
                best_index = int(np.argmin(values))
                self.best = float(values[best_index])
                self.best_step = int(steps[best_index])


class MaterialProbeHistoryCallback(dde.callbacks.Callback):
    def __init__(self, save_dir, case_config, args, period=1, append_existing=False):
        super().__init__()
        self.save_dir = save_dir
        self.args = args
        self.period = max(int(period), 1)
        self.epochs_since_last_save = 0
        self.probe_specs, self.history_payload = init_material_probe_history_payload(case_config, args)
        self.append_existing = bool(append_existing)
        if self.append_existing:
            existing_payload = load_json_if_exists(_get_save_path(self.save_dir, "json", "material_parameter_evolution.json"))
            if isinstance(existing_payload, dict) and existing_payload.get("history") is not None:
                self.history_payload = existing_payload
        self.last_step = None
        if self.history_payload.get("history"):
            self.last_step = int(self.history_payload["history"][-1].get("step", 0))

    def on_train_begin(self):
        self._record(force=True)

    def on_epoch_end(self):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save < self.period:
            return
        self.epochs_since_last_save = 0
        self._record(force=False)

    def on_batch_end(self):
        current_step = self._current_step()
        if current_step <= 0:
            return
        if current_step % self.period != 0:
            return
        self._record(force=False)

    def on_train_end(self):
        self._record(force=True)

    def _current_step(self):
        train_state = getattr(self.model, "train_state", None)
        if train_state is None:
            return 0
        step = int(getattr(train_state, "step", 0) or 0)
        iteration = int(getattr(train_state, "iteration", 0) or 0)
        return max(step, iteration)

    def _record(self, force=False):
        step = self._current_step()
        if not force and self.last_step == step:
            return
        points = np.asarray([spec["point"] for spec in self.probe_specs], dtype=float)
        predictions = predict_material_components(self.model, points, self.args, batch_size=max(len(points), 1))
        row = {"step": step, "probes": {}}
        for index, spec in enumerate(self.probe_specs):
            row["probes"][spec["name"]] = {
                "lambda": float(predictions["lambda"][index, 0]),
                "mu": float(predictions["mu"][index, 0]),
                "bulk": float(predictions["bulk"][index, 0]),
            }
        history_rows = self.history_payload.setdefault("history", [])
        while history_rows and int(history_rows[-1].get("step", -1)) > step:
            history_rows.pop()
        snapshots = self.history_payload.setdefault("snapshots", {})
        for tag, snapshot_row in list(snapshots.items()):
            try:
                snapshot_step = int(snapshot_row.get("step", 0))
            except Exception:
                snapshot_step = 0
            if snapshot_step > step:
                snapshots.pop(tag, None)
        if history_rows and int(history_rows[-1].get("step", -1)) == step:
            history_rows[-1] = row
        else:
            history_rows.append(row)
        self.last_step = step
        self._save_history()

    def _save_history(self):
        save_json(_get_save_path(self.save_dir, "json", "material_parameter_evolution.json"), self.history_payload)
        plot_material_probe_evolution(self.save_dir, self.history_payload)


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
    reference_data = getattr(args, "_reference_data", None)
    grid_nx = int(args.eval_nx)
    grid_ny = int(args.eval_ny)
    if isinstance(reference_data, dict) and reference_data.get("source") == "fem":
        eval_points = np.asarray(reference_data["evaluation_grid_points"], dtype=float)
        xs = np.unique(np.round(eval_points[:, 0], decimals=12))
        ys = np.unique(np.round(eval_points[:, 1], decimals=12))
        if len(xs) * len(ys) == len(eval_points):
            grid_nx = int(len(xs))
            grid_ny = int(len(ys))
            xx, yy = np.meshgrid(xs, ys)
        else:
            eval_points, xx, yy = make_grid(args.eval_nx, args.eval_ny)
        truths_by_load = [
            np.asarray(state, dtype=float) for state in reference_data["evaluation_grid_true_state_by_load"]
        ]
    else:
        eval_points, xx, yy = make_grid(args.eval_nx, args.eval_ny)
        truths_by_load = []
    primary_load_index = int(getattr(args, "primary_load_index", 0))
    load_specs = resolve_load_specs(getattr(args, "load_scales", "1.0"), getattr(args, "load_modes", ""))
    if isinstance(observation_points, (list, tuple)):
        observation_points_by_load = [np.asarray(points, dtype=float) for points in observation_points]
        observation_truth_by_load = [np.asarray(values, dtype=float) for values in observation_truth_clean]
    else:
        observation_points_by_load = [np.asarray(observation_points, dtype=float)]
        observation_truth_by_load = [np.asarray(observation_truth_clean, dtype=float)]

    primary_spec = load_specs[min(primary_load_index, len(load_specs) - 1)]
    truth = (
        truths_by_load[min(primary_load_index, len(truths_by_load) - 1)]
        if truths_by_load
        else exact_state_numpy(
            eval_points,
            case_config,
            load_scale=float(primary_spec["scale"]),
            load_mode=str(primary_spec["mode"]),
        )
    )
    prediction = predict_full_fields(model, eval_points, args, load_index=primary_load_index)
    residual = split_residual_prediction(
        model.predict(
            eval_points,
            operator=build_pde(
                case_config,
                args.reg_weight,
                args.method,
                load_scales=getattr(args, "load_scales", "1.0"),
                load_modes=getattr(args, "load_modes", ""),
            ),
        )
    )
    predictions_by_load = []
    observation_predictions_by_load = []
    for load_index, load_spec in enumerate(load_specs):
        if load_index >= len(truths_by_load):
            truths_by_load.append(
                exact_state_numpy(
                    eval_points,
                    case_config,
                    load_scale=float(load_spec["scale"]),
                    load_mode=str(load_spec["mode"]),
                )
            )
        predictions_by_load.append(predict_full_fields(model, eval_points, args, load_index=load_index))
        if load_index < len(observation_points_by_load):
            observation_predictions_by_load.append(
                predict_full_fields(model, observation_points_by_load[load_index], args, load_index=load_index)[:, :2]
            )
        else:
            observation_predictions_by_load.append(np.zeros((0, 2), dtype=float))
    primary_observation_points = observation_points_by_load[min(primary_load_index, len(observation_points_by_load) - 1)]
    primary_observation_truth = observation_truth_by_load[min(primary_load_index, len(observation_truth_by_load) - 1)]
    primary_observation_prediction = observation_predictions_by_load[min(primary_load_index, len(observation_predictions_by_load) - 1)]
    material_diagnostics = predict_material_diagnostics(model, eval_points, args)

    metrics = aggregate_multiload_metrics(
        args=args,
        case_config=case_config,
        eval_points=eval_points,
        residual_matrix=residual,
        predictions_by_load=predictions_by_load,
        truths_by_load=truths_by_load,
        observation_predictions_by_load=observation_predictions_by_load,
        observation_truths_by_load=observation_truth_by_load,
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
        for load_index, load_spec in enumerate(load_specs):
            mode_name = str(load_spec.get("mode", f"load_{load_index}"))
            safe_mode_name = mode_name.replace(" ", "_").replace("-", "_")
            np.savez(
                _get_save_path(save_dir, "npz", f"{observation_split_name}_l{load_index}_{safe_mode_name}_grid.npz"),
                points=eval_points,
                prediction=predictions_by_load[load_index],
                truth=truths_by_load[load_index],
                xx=xx,
                yy=yy,
                load_mode=mode_name,
                load_scale=float(load_spec.get("scale", 1.0)),
            )
        save_json(_get_save_path(save_dir, "metrics", f"{observation_split_name}_metrics.json"), metrics)
        save_metrics_text(_get_save_path(save_dir, "metrics", f"{observation_split_name}_metrics.txt"), metrics)
        np.savez(
            _get_save_path(save_dir, "npz", f"{observation_split_name}_observation_fit.npz"),
            points=primary_observation_points,
            prediction=primary_observation_prediction,
            truth=primary_observation_truth,
        )
        save_array_txt(
            _get_save_path(save_dir, "txt", f"{observation_split_name}_observation_fit.txt"),
            np.concatenate(
                [
                    primary_observation_points,
                    primary_observation_truth,
                    primary_observation_prediction,
                ],
                axis=1,
            ),
            "x y true_ux true_uy pred_ux pred_uy",
        )

        primary_mode = str(primary_spec.get("mode", f"load_{primary_load_index}"))
        for index, field_name in enumerate(STATE_FIELD_NAMES):
            truth_grid = reshape_grid(truth[:, index], grid_ny, grid_nx)
            pred_grid = reshape_grid(prediction[:, index], grid_ny, grid_nx)
            plot_field_triplet(
                save_dir,
                xx,
                yy,
                truth_grid,
                pred_grid,
                f"{observation_split_name}_{field_name}",
                title_name=f"{observation_split_name}_{field_name} ({primary_mode})",
                shared_scale=True,
            )

        if len(load_specs) > 1:
            for load_index, load_spec in enumerate(load_specs):
                mode_name = str(load_spec.get("mode", f"load_{load_index}"))
                safe_mode_name = mode_name.replace(" ", "_").replace("-", "_")
                load_truth = truths_by_load[load_index]
                load_prediction = predictions_by_load[load_index]
                for state_index, field_name in enumerate(STATE_FIELD_NAMES):
                    truth_grid = reshape_grid(load_truth[:, state_index], grid_ny, grid_nx)
                    pred_grid = reshape_grid(load_prediction[:, state_index], grid_ny, grid_nx)
                    plot_field_triplet(
                        save_dir,
                        xx,
                        yy,
                        truth_grid,
                        pred_grid,
                        f"{observation_split_name}_l{load_index}_{safe_mode_name}_{field_name}",
                        title_name=f"{observation_split_name}_{field_name} ({mode_name})",
                        shared_scale=True,
                    )

        parameterization = str(getattr(args, "material_parameterization", "lamemu")).lower()
        lambda_truth_grid = reshape_grid(truth[:, 5], grid_ny, grid_nx)
        mu_truth_grid = reshape_grid(truth[:, 6], grid_ny, grid_nx)
        lambda_pred_grid = reshape_grid(prediction[:, 5], grid_ny, grid_nx)
        mu_pred_grid = reshape_grid(prediction[:, 6], grid_ny, grid_nx)
        bulk_truth_grid = bulk_from_lambda_mu(lambda_truth_grid, mu_truth_grid)
        bulk_pred_grid = bulk_from_lambda_mu(lambda_pred_grid, mu_pred_grid)
        material_grids = {
            "lambda": (lambda_truth_grid, lambda_pred_grid, "Lambda"),
            "mu": (mu_truth_grid, mu_pred_grid, "Shear modulus Mu"),
            "bulk": (bulk_truth_grid, bulk_pred_grid, "Bulk modulus K"),
        }
        for field_key, field_title in primary_material_specs(parameterization):
            truth_grid, pred_grid, title_name = material_grids[field_key]
            file_stem = f"{observation_split_name}_{field_key}"
            plot_field_triplet(
                save_dir,
                xx,
                yy,
                truth_grid,
                pred_grid,
                field_key,
                title_name=field_title if field_key != "bulk" else title_name,
                file_stem=file_stem,
            )
            plot_material_overlay(
                save_dir,
                xx,
                yy,
                truth_grid,
                pred_grid,
                field_key,
                title_name=field_title if field_key != "bulk" else title_name,
                file_stem=file_stem,
            )

        plot_k_mu_truth_prediction_comparison(
            save_dir=save_dir,
            xx=xx,
            yy=yy,
            bulk_truth_grid=bulk_truth_grid,
            bulk_pred_grid=bulk_pred_grid,
            mu_truth_grid=mu_truth_grid,
            mu_pred_grid=mu_pred_grid,
            filename=f"{observation_split_name}_K与μ真值预测对比图.png",
        )

        if material_diagnostics is not None:
            interface_indicator_grid = reshape_grid(material_diagnostics["interface_indicator"][:, 0], grid_ny, grid_nx)
            class_probability_grid = reshape_grid(material_diagnostics["class_probs"][:, -1], grid_ny, grid_nx)
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
            primary_observation_truth,
            primary_observation_prediction,
            filename=f"{observation_split_name}_observation_fit.png",
        )
    return metrics


def save_runtime_status(save_dir, phase, step=None, extra=None):
    payload = {
        "phase": str(phase),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pid": int(os.getpid()),
    }
    if step is not None:
        payload["step"] = int(step)
    if extra:
        payload.update(extra)
    save_json(os.path.join(save_dir, "json", "runtime_status.json"), payload)


def configure_torch_runtime(thread_limit=1):
    payload = {
        "requested_thread_limit": int(thread_limit),
        "env": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }
    limit = max(int(thread_limit), 1)
    try:
        torch.set_num_threads(limit)
        payload["torch_num_threads"] = int(torch.get_num_threads())
    except Exception as exc:
        payload["torch_num_threads_error"] = repr(exc)
    try:
        torch.set_num_interop_threads(limit)
        payload["torch_num_interop_threads"] = int(torch.get_num_interop_threads())
    except Exception as exc:
        payload["torch_num_interop_threads_error"] = repr(exc)
    return payload


def inspect_runtime_device(args, net):
    device_index = int(getattr(args, "device_index", 0))
    payload = {
        "backend": dde.backend.backend_name,
        "torch_cuda_is_available": bool(torch.cuda.is_available()),
        "requested_device_index": device_index,
        "visible_devices": {
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")
        },
    }
    parameters = [parameter for parameter in net.parameters() if parameter.requires_grad]
    parameter_device = str(parameters[0].device) if parameters else "no_trainable_parameters"
    payload["parameter_device"] = parameter_device

    if dde.backend.backend_name != "pytorch":
        return payload

    if not torch.cuda.is_available():
        payload["probe_status"] = "cpu_only"
        return payload

    expected_device = torch.device(f"cuda:{device_index}")
    payload["expected_device"] = str(expected_device)
    payload["current_device_index"] = int(torch.cuda.current_device())
    payload["current_device_name"] = str(torch.cuda.get_device_name(expected_device))

    if parameter_device != str(expected_device):
        raise RuntimeError(
            f"Network parameter device mismatch: expected {expected_device}, got {parameter_device}"
        )

    net.eval()
    probe_input = torch.zeros((2, 2), dtype=torch.float32, device=expected_device)
    with torch.no_grad():
        probe_output = net(probe_input)
    payload["probe_input_device"] = str(probe_input.device)
    payload["probe_output_device"] = str(probe_output.device)
    payload["probe_output_shape"] = list(probe_output.shape)
    if str(probe_output.device) != str(expected_device):
        raise RuntimeError(
            f"Network forward device mismatch: expected {expected_device}, got {probe_output.device}"
        )
    payload["probe_status"] = "ok"
    return payload


def build_run_config_payload(args, case_config, net=None, metadata=None):
    bc_loss_names = build_bc_loss_names(args)
    return {
        "args": {key: value for key, value in vars(args).items() if not key.startswith("_")},
        "case": asdict(case_config),
        "parameter_count": count_trainable_parameters(net) if net is not None else None,
        "field_names": FIELD_NAMES,
        "pde_loss_names": pde_loss_names(
            args.reg_weight,
            args.method,
            getattr(args, "load_scales", "1.0"),
            getattr(args, "load_modes", ""),
        ),
        "bc_loss_names": bc_loss_names,
        "selection_metric": "validation_observation_mse",
        "selection_metric_target": validation_observation_target_name(args),
        "report_checkpoint": report_checkpoint_preference(args),
        "observation_cache_path": metadata.get("observation_cache_path") if isinstance(metadata, dict) else None,
    }


def save_run_config_artifacts(args, save_dir, case_config, net=None, metadata=None):
    config_payload = build_run_config_payload(args, case_config, net=net, metadata=metadata)
    save_json(_get_save_path(save_dir, "json", "run_config.json"), config_payload)
    save_json_as_text(os.path.join(save_dir, "配置.txt"), config_payload)
    return config_payload


def get_material_branch_parameters(net):
    if hasattr(net, "material_net"):
        params = [parameter for parameter in net.material_net.parameters() if parameter.requires_grad]
        if params:
            return params, "material_net"
    if hasattr(net, "raw_material_params"):
        parameter = getattr(net, "raw_material_params")
        if parameter.requires_grad:
            return [parameter], "raw_material_params"
    params = [parameter for parameter in net.parameters() if parameter.requires_grad]
    return params, "full_network_fallback"


def build_loss_group_indices(args):
    num_pde = len(
        pde_loss_names(
            args.reg_weight,
            args.method,
            getattr(args, "load_scales", "1.0"),
            getattr(args, "load_modes", ""),
        )
    )
    bc_loss_names = build_bc_loss_names(args)
    observation_indices = [num_pde + index for index, name in enumerate(bc_loss_names) if name.startswith("obs_")]
    boundary_indices = [num_pde + index for index, name in enumerate(bc_loss_names) if not name.startswith("obs_")]
    return {
        "num_pde": int(num_pde),
        "boundary_indices": boundary_indices,
        "observation_indices": observation_indices,
    }


def clamp_float(value, lower, upper):
    return float(max(float(lower), min(float(upper), float(value))))


def propose_scales_from_signals(signals, base_scales, min_scales, max_scales):
    signal_array = np.asarray(signals, dtype=np.float64)
    signal_array = np.maximum(signal_array, 1e-12)
    inv_signal = 1.0 / signal_array
    inv_sum = float(np.sum(inv_signal))
    if not np.isfinite(inv_sum) or inv_sum <= 0:
        return np.asarray(base_scales, dtype=np.float64)
    shares = inv_signal / inv_sum
    total_base = float(np.sum(base_scales))
    proposed = shares * total_base
    return np.clip(proposed, np.asarray(min_scales, dtype=np.float64), np.asarray(max_scales, dtype=np.float64))


def gradient_l2_norm(loss_tensor, parameters, retain_graph, eps=1e-12):
    if not parameters:
        return float("nan")
    grads = torch.autograd.grad(
        loss_tensor,
        parameters,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    total = None
    for grad in grads:
        if grad is None:
            continue
        grad_sq = torch.sum(grad.detach() ** 2)
        total = grad_sq if total is None else (total + grad_sq)
    if total is None:
        return float("nan")
    return float(torch.sqrt(total + float(eps)).detach().cpu().item())


class MaterialBranchGradNormCallback(dde.callbacks.Callback):
    def __init__(
        self,
        args,
        save_dir,
        period=1000,
        reg_scale=1.0,
        ema=0.5,
        min_physics_scale=0.25,
        max_physics_scale=4.0,
        min_observation_scale=0.25,
        max_observation_scale=4.0,
        min_boundary_scale=0.25,
        max_boundary_scale=4.0,
        eps=1e-12,
    ):
        super().__init__()
        self.args = args
        self.save_dir = save_dir
        self.period = max(int(period), 1)
        self.reg_scale = float(reg_scale)
        self.ema = clamp_float(float(ema), 0.0, 1.0)
        self.min_scales = np.array(
            [float(min_physics_scale), float(min_observation_scale), float(min_boundary_scale)],
            dtype=np.float64,
        )
        self.max_scales = np.array(
            [float(max_physics_scale), float(max_observation_scale), float(max_boundary_scale)],
            dtype=np.float64,
        )
        self.base_scales = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        self.current_scales = self.base_scales.copy()
        self.eps = float(eps)
        self.group_indices = build_loss_group_indices(args)
        self.history = []
        self.parameter_source = "unknown"

    def on_train_begin(self):
        self._apply_loss_weights()

    def on_epoch_end(self):
        step = int(getattr(self.model.train_state, "step", 0))
        if step <= 0 or step % self.period != 0:
            return

        parameters, parameter_source = get_material_branch_parameters(self.model.net)
        self.parameter_source = parameter_source
        if not parameters:
            return

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

        physics_loss = torch.sum(raw_losses[: self.group_indices["num_pde"]])
        if self.group_indices["boundary_indices"]:
            boundary_loss = torch.sum(raw_losses[self.group_indices["boundary_indices"]])
        else:
            boundary_loss = torch.zeros((), dtype=raw_losses.dtype, device=raw_losses.device)
        if self.group_indices["observation_indices"]:
            observation_loss = torch.sum(raw_losses[self.group_indices["observation_indices"]])
        else:
            observation_loss = torch.zeros((), dtype=raw_losses.dtype, device=raw_losses.device)

        physics_grad_norm = gradient_l2_norm(physics_loss, parameters, retain_graph=True, eps=self.eps)
        observation_grad_norm = gradient_l2_norm(observation_loss, parameters, retain_graph=True, eps=self.eps)
        boundary_grad_norm = gradient_l2_norm(boundary_loss, parameters, retain_graph=False, eps=self.eps)

        grad_vector = np.array(
            [physics_grad_norm, observation_grad_norm, boundary_grad_norm],
            dtype=np.float64,
        )
        if np.any(~np.isfinite(grad_vector)):
            return

        proposed_scales = propose_scales_from_signals(
            grad_vector,
            self.base_scales,
            self.min_scales,
            self.max_scales,
        )
        self.current_scales = (1.0 - self.ema) * self.current_scales + self.ema * proposed_scales
        self.current_scales = np.clip(self.current_scales, self.min_scales, self.max_scales)
        self._apply_loss_weights()

        row = {
            "step": int(step),
            "parameter_source": self.parameter_source,
            "physics_loss": float(physics_loss.detach().cpu().item()),
            "observation_loss": float(observation_loss.detach().cpu().item()),
            "boundary_loss": float(boundary_loss.detach().cpu().item()),
            "physics_grad_norm": float(physics_grad_norm),
            "observation_grad_norm": float(observation_grad_norm),
            "boundary_grad_norm": float(boundary_grad_norm),
            "physics_scale": float(self.current_scales[0]),
            "observation_scale": float(self.current_scales[1]),
            "boundary_scale": float(self.current_scales[2]),
        }
        self.history.append(row)
        save_json(os.path.join(self.save_dir, "json", "材料分支梯度归一化历史.json"), self.history)

    def on_train_end(self):
        save_json(os.path.join(self.save_dir, "json", "材料分支梯度归一化历史.json"), self.history)

    def _apply_loss_weights(self):
        self.model.loss_weights = resolve_loss_weights(
            self.args,
            physics_scale=float(self.current_scales[0]),
            reg_scale=self.reg_scale,
            boundary_scale=float(self.current_scales[2]),
            data_scale=float(self.current_scales[1]),
        )


class RuntimeStatusCallback(dde.callbacks.Callback):
    def __init__(self, save_dir, period):
        super().__init__()
        self.save_dir = save_dir
        self.period = max(int(period), 1)

    def on_train_begin(self):
        save_runtime_status(
            self.save_dir,
            "training",
            step=int(getattr(self.model.train_state, "step", 0)),
        )
        self._flush_streams()

    def on_epoch_end(self):
        step = int(getattr(self.model.train_state, "step", 0))
        if step <= 0 or step % self.period != 0:
            return
        loss_train = getattr(self.model.train_state, "loss_train", None)
        loss_test = getattr(self.model.train_state, "loss_test", None)
        payload = {}
        if loss_train is not None and len(loss_train) > 0:
            payload["loss_train_sum"] = float(np.sum(loss_train))
        if loss_test is not None and len(loss_test) > 0:
            payload["loss_test_sum"] = float(np.sum(loss_test))
        save_runtime_status(self.save_dir, "training", step=step, extra=payload)
        self._flush_streams()

    def on_train_end(self):
        save_runtime_status(
            self.save_dir,
            "train_returned",
            step=int(getattr(self.model.train_state, "step", 0)),
        )
        self._flush_streams()

    @staticmethod
    def _flush_streams():
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass


def build_model(args, data):
    net = build_network(args)
    if dde.backend.backend_name == "pytorch":
        try:
            import torch

            if torch.cuda.is_available():
                target_device = torch.device(f"cuda:{int(getattr(args, 'device_index', 0))}")
                net = net.to(target_device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to place network on cuda:{int(getattr(args, 'device_index', 0))}"
            ) from exc
    model = dde.Model(data, net)
    loss_weights = resolve_loss_weights(args)
    model.compile("adam", lr=args.lr, loss_weights=loss_weights)
    return model, net


def make_callbacks(args, save_dir, metadata, include_history=True, include_checkpoint=True):
    append_existing = bool(
        getattr(args, "resume_training", False)
        or getattr(args, "model_path", None)
        or int(getattr(args, "_resume_initial_step", 0)) > 0
        or int(getattr(args, "_progress_initial_step", 0)) > 0
    )
    num_pde = len(
        pde_loss_names(
            args.reg_weight,
            args.method,
            getattr(args, "load_scales", "1.0"),
            getattr(args, "load_modes", ""),
        )
    )
    bc_loss_names = build_bc_loss_names(args)
    model_dir = ensure_dir(os.path.join(save_dir, "model"))
    callbacks = []
    if include_history:
        callbacks.append(
            LossHistoryCallback(
            save_dir=save_dir,
            period=args.display_every,
            filename="loss_history.png",
            num_pde_losses=num_pde,
            num_bc_losses=len(bc_loss_names),
            pde_loss_names=pde_loss_names(
                args.reg_weight,
                args.method,
                getattr(args, "load_scales", "1.0"),
                getattr(args, "load_modes", ""),
            ),
            bc_loss_names=bc_loss_names,
            pde_label="Physics Loss",
            bc_label="Boundary Loss",
            data_loss_prefix="obs_",
            data_label="Observation Loss",
            show_total=False,
            save_all_components=True,
            append_existing=append_existing,
        )
        )
    if include_checkpoint:
        val_payloads = metadata.get("val_observation_loads", [metadata["val_observation"]])
        callbacks.append(
            ValidationObservationCheckpoint(
            filepath=os.path.join(model_dir, "best_model"),
            save_dir=save_dir,
            observation_points=[payload["points"] for payload in val_payloads],
            observation_truth_clean=[resolve_validation_observation_target(args, payload) for payload in val_payloads],
            args=args,
            period=args.display_every,
            verbose=1,
            append_existing=append_existing,
        )
        )
    callbacks.append(
        MaterialProbeHistoryCallback(
            save_dir=save_dir,
            case_config=get_case_config(args.case),
            args=args,
            period=args.display_every,
            append_existing=append_existing,
        )
    )
    callbacks.append(
        RuntimeStatusCallback(
            save_dir=save_dir,
            period=max(1, args.display_every),
        )
    )
    if str(getattr(args, "dynamic_loss_balance", "none")).lower() == "material_gradnorm":
        callbacks.append(
            MaterialBranchGradNormCallback(
                args=args,
                save_dir=save_dir,
                period=args.dynamic_balance_period,
                reg_scale=args.dynamic_balance_reg_scale,
                ema=args.dynamic_balance_ema,
                min_physics_scale=args.dynamic_balance_min_physics_scale,
                max_physics_scale=args.dynamic_balance_max_physics_scale,
                min_observation_scale=args.dynamic_balance_min_observation_scale,
                max_observation_scale=args.dynamic_balance_max_observation_scale,
                min_boundary_scale=args.dynamic_balance_min_boundary_scale,
                max_boundary_scale=args.dynamic_balance_max_boundary_scale,
                eps=args.dynamic_balance_grad_eps,
            )
        )
    callbacks.append(
        TqdmProgressCallback(
            total_steps=int(getattr(args, "_resume_initial_step", 0) + args.iterations),
            display_every=args.display_every,
            metric_name="relL2",
            initial_step=getattr(args, "_progress_initial_step", 0),
            step_offset=getattr(args, "_progress_step_offset", 0),
        )
    )
    return callbacks


def save_last_model(model, save_dir):
    model_dir = ensure_dir(os.path.join(save_dir, "model"))
    model.save(os.path.join(model_dir, "last_model"))


def build_common_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--method",
        choices=[
            "pinn",
            "iaminn_v2",
            "geoiaminn",
            "geoiaminn_v3",
            "twobranch_compact_kmu",
            "twobranch_stress_kmu",
            "threebranch_stress_kmu",
            "fivestate_stress_kmu",
            "pfnn_scalar_kmu",
        ],
        default="pinn",
    )
    parser.add_argument(
        "--case",
        choices=["layered", "single_material", "single_inclusion", "double_inclusion", "weak_interlayer"],
        default="single_inclusion",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--display_every", type=int, default=1000)
    parser.add_argument("--num_domain", type=int, default=4000)
    parser.add_argument("--num_test", type=int, default=2000)
    parser.add_argument("--num_boundary", type=int, default=400)
    parser.add_argument("--num_observe", type=int, default=1200)
    parser.add_argument("--num_val_observe", type=int, default=1200)
    parser.add_argument("--num_eval_observe", type=int, default=1200)
    parser.add_argument("--noise_level", type=float, default=0.0)
    parser.add_argument(
        "--noise_scale_mode",
        choices=["component_std", "component_std_no_fallback", "global_rms"],
        default="component_std",
    )
    parser.add_argument("--reg_weight", type=float, default=1e-4)
    parser.add_argument("--data_weight", type=float, default=20.0)
    parser.add_argument("--boundary_weight", type=float, default=20.0)
    parser.add_argument("--lambda_floor", type=float, default=0.1)
    parser.add_argument("--mu_floor", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument("--backbone_type", choices=["mlp", "resmlp"], default="mlp")
    parser.add_argument("--hidden_layers", type=str, default="128,128,128,128")
    parser.add_argument("--state_layers", type=str, default="128,128,128,128")
    parser.add_argument("--material_layers", type=str, default="128,128,128,128")
    parser.add_argument("--geometry_layers", type=str, default="64,64,64")
    parser.add_argument("--interface_layers", type=str, default="128,128,128,128")
    parser.add_argument("--interface_sharpness", type=float, default=10.0)
    parser.add_argument("--num_frequencies", type=int, default=4)
    parser.add_argument("--material_parameterization", choices=["bulkmu", "lamemu"], default="bulkmu")
    parser.add_argument("--k_floor", type=float, default=0.2)
    parser.add_argument("--load_scales", type=str, default="1.0")
    parser.add_argument("--load_modes", type=str, default="")
    parser.add_argument("--load_balance_mode", type=str, default="reference_rms", choices=["none", "reference_rms"])
    parser.add_argument("--load_balance_nx", type=int, default=48)
    parser.add_argument("--load_balance_ny", type=int, default=48)
    parser.add_argument("--load_balance_epsilon", type=float, default=1e-6)
    parser.add_argument(
        "--dynamic_loss_balance",
        type=str,
        default="none",
        choices=["none", "material_gradnorm"],
    )
    parser.add_argument("--dynamic_balance_period", type=int, default=1000)
    parser.add_argument("--dynamic_balance_ema", type=float, default=0.5)
    parser.add_argument("--dynamic_balance_reg_scale", type=float, default=1.0)
    parser.add_argument("--dynamic_balance_min_physics_scale", type=float, default=0.25)
    parser.add_argument("--dynamic_balance_max_physics_scale", type=float, default=4.0)
    parser.add_argument("--dynamic_balance_min_observation_scale", type=float, default=0.25)
    parser.add_argument("--dynamic_balance_max_observation_scale", type=float, default=4.0)
    parser.add_argument("--dynamic_balance_min_boundary_scale", type=float, default=0.25)
    parser.add_argument("--dynamic_balance_max_boundary_scale", type=float, default=4.0)
    parser.add_argument("--dynamic_balance_grad_eps", type=float, default=1e-12)
    parser.add_argument("--primary_load_index", type=int, default=0)
    parser.add_argument("--observation_split_tag", type=str, default="official_softbc_v1")
    parser.add_argument("--observation_split_seed", type=int, default=None)
    parser.add_argument("--observation_cache_dir", type=str, default=DEFAULT_OBSERVATION_CACHE_DIR)
    parser.add_argument("--validation_observation_target", choices=["clean", "noisy"], default="clean")
    parser.add_argument("--report_checkpoint", choices=["best_model", "last_model"], default="best_model")
    parser.add_argument("--reaction_consistency", action="store_true")
    parser.add_argument("--reaction_weight", type=float, default=0.0)
    parser.add_argument("--reaction_edge", type=str, default="top", choices=["top", "bottom", "left", "right"])
    parser.add_argument("--teacher_source", choices=["analytic", "fem"], default="analytic")
    parser.add_argument("--teacher_run_dir", type=str, default=None)
    parser.add_argument("--exp_root", type=str, default=DEFAULT_EXP_ROOT)
    parser.add_argument("--experiment_group", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--eval_nx", type=int, default=141)
    parser.add_argument("--eval_ny", type=int, default=141)
    parser.add_argument("--device_index", type=int, default=0)
    parser.add_argument("--device_debug", action="store_true")
    return parser


def prepare_run(args):
    if args.num_val_observe <= 0:
        raise ValueError("num_val_observe must be positive for fair model selection.")
    if args.num_eval_observe <= 0:
        raise ValueError("num_eval_observe must be positive for held-out evaluation.")
    if "balanced1200" in str(getattr(args, "observation_split_tag", "")).lower():
        for split_name, value in {
            "num_observe": args.num_observe,
            "num_val_observe": args.num_val_observe,
            "num_eval_observe": args.num_eval_observe,
        }.items():
            if int(value) % 3 != 0:
                raise ValueError(f"{split_name} must be divisible by 3 for balanced three-group observation sampling.")
    if args.num_boundary <= 0:
        raise ValueError("num_boundary must be positive when using unified soft boundary constraints.")
    load_specs = resolve_load_specs(getattr(args, "load_scales", "1.0"), getattr(args, "load_modes", ""))
    if int(getattr(args, "primary_load_index", 0)) < 0 or int(getattr(args, "primary_load_index", 0)) >= len(load_specs):
        raise ValueError("primary_load_index is out of range for configured load cases.")
    if reaction_consistency_enabled(args) and is_compact_material_method(args.method):
        raise ValueError("reaction_consistency is not supported for compact material methods without explicit stress outputs.")
    if str(getattr(args, "teacher_source", "analytic")).lower() == "fem":
        if not getattr(args, "teacher_run_dir", None):
            raise ValueError("teacher_run_dir is required when teacher_source=fem.")
        args.teacher_run_dir = os.path.abspath(args.teacher_run_dir)
        if not os.path.isdir(args.teacher_run_dir):
            raise FileNotFoundError(f"FEM teacher run directory does not exist: {args.teacher_run_dir}")
    set_random_seed(args.seed)
    if args.device_debug:
        print_gpu_info()
    case_config = get_case_config(args.case)
    if args.save_dir is None:
        args.save_dir = resolve_save_dir(args)
    ensure_dir(args.save_dir)
    return case_config
