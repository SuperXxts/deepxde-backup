import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np

from fem_visualization import render_teacher_run_visuals


PI = math.pi
FIELD_NAMES = ["ux", "uy", "sxx", "syy", "sxy", "lambda", "mu"]
DEFAULT_EXP_ROOT = os.path.join("exp", "04.FEM")
DEFAULT_OBSERVATION_CACHE_DIR = os.path.join(
    "exp",
    "_shared_observation_splits",
    "spatial_material_inverse",
)


def resolve_repo_root():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.abspath(os.path.join(current_dir, "../..")),
        os.path.abspath(os.path.join(current_dir, "..", "..", "..", "deepxde_work", "deepxde-master")),
    ]
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "utils")):
            return candidate
    raise FileNotFoundError("Cannot resolve repository root for FEM teacher generation.")


ROOT_DIR = resolve_repo_root()
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


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


def _get_save_path(save_dir, subfolder, filename):
    ensure_dir(save_dir)
    subfolder_path = os.path.join(save_dir, subfolder)
    ensure_dir(subfolder_path)
    return os.path.join(subfolder_path, filename)


def make_grid(nx, ny):
    xs = np.linspace(0.0, 1.0, int(nx))
    ys = np.linspace(0.0, 1.0, int(ny))
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    return points, xx, yy


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


def build_observation_points(num_observe, seed):
    rng = np.random.default_rng(int(seed))
    return rng.random((int(num_observe), 2))


def resolve_observation_split_cache_path(cache_dir, case_name, seed, num_train, num_val, num_eval, tag):
    filename = f"{case_name}_seed{int(seed)}_train{int(num_train)}_val{int(num_val)}_eval{int(num_eval)}_{tag}.npz"
    return os.path.join(cache_dir, filename)


def build_observation_splits(num_train, num_val, num_eval, seed, case_name, cache_dir=None, tag="official_softbc_v1"):
    total = int(num_train) + int(num_val) + int(num_eval)
    if total <= 0:
        raise ValueError("At least one observation point is required.")

    cache_path = None
    if cache_dir:
        cache_dir = cache_dir if os.path.isabs(cache_dir) else os.path.join(ROOT_DIR, cache_dir)
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


def add_noise(values, noise_level, seed):
    values = np.asarray(values, dtype=float)
    if float(noise_level) <= 0.0 or values.size == 0:
        return values.copy()
    rng = np.random.default_rng(int(seed))
    scale = np.std(values, axis=0, keepdims=True)
    scale = np.where(scale > 1e-12, scale, 1.0)
    noisy = values + float(noise_level) * scale * rng.standard_normal(values.shape)
    return np.asarray(noisy, dtype=float)


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
            raise ValueError(f"Unsupported load_mode: {mode}. Valid modes: {sorted(valid_modes)}")
    return modes


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


def bulk_from_lambda_mu(lmbd, mu):
    return np.asarray(lmbd, dtype=float) + (2.0 / 3.0) * np.asarray(mu, dtype=float)


def lambda_from_bulk_mu(bulk, mu):
    return np.asarray(bulk, dtype=float) - (2.0 / 3.0) * np.asarray(mu, dtype=float)


def smooth_step_numpy(value):
    return 0.5 * (1.0 + np.tanh(value))


def smooth_disk_numpy(x, y, cx, cy, radius, width):
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return smooth_step_numpy((radius - dist) / width)


def smooth_horizontal_band_numpy(y, y_bottom, y_top, width):
    lower = smooth_step_numpy((y - y_bottom) / width)
    upper = smooth_step_numpy((y_top - y) / width)
    return lower * upper


def exact_material_xy_numpy(x, y, case_config):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    width = float(case_config.interface_width)

    if case_config.name == "single_material":
        lmbd = np.full_like(x, float(case_config.lambda_bg), dtype=float)
        mu = np.full_like(x, float(case_config.mu_bg), dtype=float)
        return lmbd, mu

    if case_config.name == "layered":
        mask = smooth_step_numpy((y - float(case_config.layer_y)) / width)
        lmbd = float(case_config.lambda_bg) + float(case_config.lambda_ctr_1) * mask
        mu = float(case_config.mu_bg) + float(case_config.mu_ctr_1) * mask
        return lmbd, mu

    if case_config.name == "weak_interlayer":
        mask = smooth_horizontal_band_numpy(y, float(case_config.weak_y_bottom), float(case_config.weak_y_top), width)
        lmbd = float(case_config.lambda_bg) + float(case_config.lambda_ctr_1) * mask
        mu = float(case_config.mu_bg) + float(case_config.mu_ctr_1) * mask
        return lmbd, mu

    mask_1 = smooth_disk_numpy(
        x,
        y,
        float(case_config.circle_1_cx),
        float(case_config.circle_1_cy),
        float(case_config.circle_1_r),
        width,
    )
    lmbd = float(case_config.lambda_bg) + float(case_config.lambda_ctr_1) * mask_1
    mu = float(case_config.mu_bg) + float(case_config.mu_ctr_1) * mask_1

    if case_config.name == "double_inclusion":
        mask_2 = smooth_disk_numpy(
            x,
            y,
            float(case_config.circle_2_cx),
            float(case_config.circle_2_cy),
            float(case_config.circle_2_r),
            width,
        )
        lmbd = lmbd + float(case_config.lambda_ctr_2) * mask_2
        mu = mu + float(case_config.mu_ctr_2) * mask_2
    return lmbd, mu


def exact_material_numpy(points, case_config):
    points = np.asarray(points, dtype=float)
    return exact_material_xy_numpy(points[:, 0:1], points[:, 1:2], case_config)


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


def exact_strain_gradient_numpy(points, case_config, load_scale=1.0, load_mode="legacy"):
    x = points[:, 0:1]
    y = points[:, 1:2]
    mode = str(load_mode).strip().lower()
    base_amp = max(abs(case_config.amplitude_u), abs(case_config.amplitude_v), 1e-8)
    if mode in {"", "legacy"}:
        exx_x = -load_scale * case_config.amplitude_u * (PI ** 2) * np.sin(PI * x) * np.sin(PI * y)
        exx_y = load_scale * case_config.amplitude_u * (PI ** 2) * np.cos(PI * x) * np.cos(PI * y)
        eyy_x = load_scale * case_config.amplitude_v * (2.0 * PI ** 2) * np.cos(2.0 * PI * x) * np.cos(PI * y)
        eyy_y = -load_scale * case_config.amplitude_v * (PI ** 2) * np.sin(2.0 * PI * x) * np.sin(PI * y)
        exy_x = 0.5 * (
            load_scale * case_config.amplitude_u * (PI ** 2) * np.cos(PI * x) * np.cos(PI * y)
            - load_scale * case_config.amplitude_v * (4.0 * PI ** 2) * np.sin(2.0 * PI * x) * np.sin(PI * y)
        )
        exy_y = 0.5 * (
            -load_scale * case_config.amplitude_u * (PI ** 2) * np.sin(PI * x) * np.sin(PI * y)
            + load_scale * case_config.amplitude_v * (2.0 * PI ** 2) * np.cos(2.0 * PI * x) * np.cos(PI * y)
        )
    elif mode == "x_tension":
        exx_x = -load_scale * base_amp * (PI ** 2) * np.sin(PI * x) * np.sin(PI * y)
        exx_y = load_scale * base_amp * (PI ** 2) * np.cos(PI * x) * np.cos(PI * y)
        eyy_x = np.zeros_like(exx_x)
        eyy_y = np.zeros_like(exx_x)
        exy_x = 0.5 * load_scale * base_amp * (PI ** 2) * np.cos(PI * x) * np.cos(PI * y)
        exy_y = -0.5 * load_scale * base_amp * (PI ** 2) * np.sin(PI * x) * np.sin(PI * y)
    elif mode == "y_tension":
        exx_x = np.zeros_like(x)
        exx_y = np.zeros_like(x)
        eyy_x = load_scale * base_amp * (PI ** 2) * np.cos(PI * x) * np.cos(PI * y)
        eyy_y = -load_scale * base_amp * (PI ** 2) * np.sin(PI * x) * np.sin(PI * y)
        exy_x = -0.5 * load_scale * base_amp * (PI ** 2) * np.sin(PI * x) * np.sin(PI * y)
        exy_y = 0.5 * load_scale * base_amp * (PI ** 2) * np.cos(PI * x) * np.cos(PI * y)
    elif mode == "shear":
        exx_x = -load_scale * base_amp * (PI ** 2) * np.sin(PI * x) * np.sin(2.0 * PI * y)
        exx_y = load_scale * base_amp * (2.0 * PI ** 2) * np.cos(PI * x) * np.cos(2.0 * PI * y)
        eyy_x = load_scale * base_amp * (2.0 * PI ** 2) * np.cos(2.0 * PI * x) * np.cos(PI * y)
        eyy_y = -load_scale * base_amp * (PI ** 2) * np.sin(2.0 * PI * x) * np.sin(PI * y)
        exy_x = load_scale * base_amp * (PI ** 2) * (
            np.cos(PI * x) * np.cos(2.0 * PI * y) - 2.0 * np.sin(2.0 * PI * x) * np.sin(PI * y)
        )
        exy_y = load_scale * base_amp * (PI ** 2) * (
            -2.0 * np.sin(PI * x) * np.sin(2.0 * PI * y) + np.cos(2.0 * PI * x) * np.cos(PI * y)
        )
    elif mode == "biaxial_bulk":
        exx_x = -load_scale * base_amp * (PI ** 2) * np.sin(PI * x)
        exx_y = np.zeros_like(exx_x)
        eyy_x = np.zeros_like(exx_x)
        eyy_y = -load_scale * base_amp * (PI ** 2) * np.sin(PI * y)
        exy_x = np.zeros_like(exx_x)
        exy_y = np.zeros_like(exx_x)
    elif mode == "uniaxial_x":
        exx_x = -load_scale * base_amp * (PI ** 2) * np.sin(PI * x)
        exx_y = np.zeros_like(exx_x)
        eyy_x = np.zeros_like(exx_x)
        eyy_y = np.zeros_like(exx_x)
        exy_x = np.zeros_like(exx_x)
        exy_y = np.zeros_like(exx_x)
    elif mode in {"uniaxial_y", "normal_to_layer"}:
        exx_x = np.zeros_like(x)
        exx_y = np.zeros_like(x)
        eyy_x = np.zeros_like(x)
        eyy_y = -load_scale * base_amp * (PI ** 2) * np.sin(PI * y)
        exy_x = np.zeros_like(x)
        exy_y = np.zeros_like(x)
    elif mode == "cross_layer_shear":
        exx_x = np.zeros_like(x)
        exx_y = np.zeros_like(x)
        eyy_x = np.zeros_like(x)
        eyy_y = np.zeros_like(x)
        exy_x = np.zeros_like(x)
        exy_y = -0.5 * load_scale * base_amp * (PI ** 2) * np.sin(PI * y)
    elif mode == "pure_shear":
        exx_x = np.zeros_like(x)
        exx_y = np.zeros_like(x)
        eyy_x = np.zeros_like(x)
        eyy_y = np.zeros_like(x)
        exy_x = -0.5 * load_scale * base_amp * (PI ** 2) * np.sin(PI * x)
        exy_y = -0.5 * load_scale * base_amp * (PI ** 2) * np.sin(PI * y)
    elif mode == "bending_y":
        exx_x = load_scale * base_amp * (y - 0.5) * (PI ** 2) * np.cos(PI * x)
        exx_y = load_scale * base_amp * PI * np.sin(PI * x)
        eyy_x = np.zeros_like(exx_x)
        eyy_y = np.zeros_like(exx_x)
        exy_x = -0.5 * load_scale * base_amp * PI * np.sin(PI * x) * (0.25 * PI - 1.0)
        exy_y = np.zeros_like(exx_x)
    elif mode == "top_nonuniform_compression":
        exx_x = np.zeros_like(x)
        exx_y = np.zeros_like(x)
        eyy_x = load_scale * base_amp * PI * np.cos(PI * x)
        eyy_y = np.zeros_like(x)
        exy_x = -0.5 * load_scale * base_amp * y * (PI ** 2) * np.sin(PI * x)
        exy_y = 0.5 * load_scale * base_amp * PI * np.cos(PI * x)
    elif mode == "edge_patch_load":
        patch_center = 0.25
        patch_width = 0.02
        patch_profile = np.exp(-((x - patch_center) ** 2) / patch_width)
        patch_dx = (-2.0 * (x - patch_center) / patch_width) * patch_profile
        patch_dxx = ((4.0 * (x - patch_center) ** 2) / (patch_width ** 2) - 2.0 / patch_width) * patch_profile
        exx_x = np.zeros_like(x)
        exx_y = np.zeros_like(x)
        eyy_x = load_scale * base_amp * patch_dx
        eyy_y = np.zeros_like(x)
        exy_x = 0.5 * load_scale * base_amp * y * patch_dxx
        exy_y = 0.5 * load_scale * base_amp * patch_dx
    else:
        raise ValueError(f"Unsupported load_mode: {load_mode}")
    return exx_x, exx_y, eyy_x, eyy_y, exy_x, exy_y


def exact_material_gradient_xy_numpy(x, y, case_config):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    width = float(case_config.interface_width)

    if case_config.name == "single_material":
        zeros = np.zeros_like(x)
        return zeros, zeros, zeros, zeros

    if case_config.name == "layered":
        arg = (y - float(case_config.layer_y)) / width
        sech2 = 1.0 - np.tanh(arg) ** 2
        mask_y = 0.5 * sech2 / width
        zeros = np.zeros_like(x)
        return (
            zeros,
            float(case_config.lambda_ctr_1) * mask_y,
            zeros,
            float(case_config.mu_ctr_1) * mask_y,
        )

    if case_config.name == "weak_interlayer":
        y_bottom = float(case_config.weak_y_bottom)
        y_top = float(case_config.weak_y_top)
        arg_lower = (y - y_bottom) / width
        arg_upper = (y_top - y) / width
        lower = smooth_step_numpy(arg_lower)
        upper = smooth_step_numpy(arg_upper)
        lower_y = 0.5 * (1.0 - np.tanh(arg_lower) ** 2) / width
        upper_y = -0.5 * (1.0 - np.tanh(arg_upper) ** 2) / width
        mask_y = lower_y * upper + lower * upper_y
        zeros = np.zeros_like(x)
        return (
            zeros,
            float(case_config.lambda_ctr_1) * mask_y,
            zeros,
            float(case_config.mu_ctr_1) * mask_y,
        )

    def disk_grad(cx, cy, radius):
        dx = x - float(cx)
        dy = y - float(cy)
        dist = np.sqrt(dx ** 2 + dy ** 2)
        safe_dist = np.where(dist > 1e-12, dist, 1.0)
        arg = (float(radius) - dist) / width
        sech2 = 1.0 - np.tanh(arg) ** 2
        common = -0.5 * sech2 / width
        dist_x = np.where(dist > 1e-12, dx / safe_dist, 0.0)
        dist_y = np.where(dist > 1e-12, dy / safe_dist, 0.0)
        return common * dist_x, common * dist_y

    mask1_x, mask1_y = disk_grad(case_config.circle_1_cx, case_config.circle_1_cy, case_config.circle_1_r)
    lmbd_x = float(case_config.lambda_ctr_1) * mask1_x
    lmbd_y = float(case_config.lambda_ctr_1) * mask1_y
    mu_x = float(case_config.mu_ctr_1) * mask1_x
    mu_y = float(case_config.mu_ctr_1) * mask1_y

    if case_config.name == "double_inclusion":
        mask2_x, mask2_y = disk_grad(case_config.circle_2_cx, case_config.circle_2_cy, case_config.circle_2_r)
        lmbd_x = lmbd_x + float(case_config.lambda_ctr_2) * mask2_x
        lmbd_y = lmbd_y + float(case_config.lambda_ctr_2) * mask2_y
        mu_x = mu_x + float(case_config.mu_ctr_2) * mask2_x
        mu_y = mu_y + float(case_config.mu_ctr_2) * mask2_y
    return lmbd_x, lmbd_y, mu_x, mu_y


def exact_state_numpy(points, case_config, load_scale=1.0, load_mode="legacy"):
    ux, uy = exact_displacement_numpy(points, case_config, load_scale=load_scale, load_mode=load_mode)
    exx, eyy, exy = exact_strain_numpy(points, case_config, load_scale=load_scale, load_mode=load_mode)
    lmbd, mu = exact_material_numpy(points, case_config)
    trace = exx + eyy
    sxx = lmbd * trace + 2.0 * mu * exx
    syy = lmbd * trace + 2.0 * mu * eyy
    sxy = 2.0 * mu * exy
    return np.hstack((ux, uy, sxx, syy, sxy, lmbd, mu))


def smooth_step_torch(value):
    import torch

    return 0.5 * (1.0 + torch.tanh(value))


def smooth_disk_torch(x, y, cx, cy, radius, width):
    import torch

    dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return smooth_step_torch((radius - dist) / width)


def smooth_horizontal_band_torch(y, y_bottom, y_top, width):
    lower = smooth_step_torch((y - y_bottom) / width)
    upper = smooth_step_torch((y_top - y) / width)
    return lower * upper


def exact_material_torch(points, case_config):
    import torch

    x = points[:, 0:1]
    y = points[:, 1:2]
    width = float(case_config.interface_width)

    if case_config.name == "single_material":
        lmbd = torch.full_like(x, float(case_config.lambda_bg))
        mu = torch.full_like(x, float(case_config.mu_bg))
        return lmbd, mu

    if case_config.name == "layered":
        mask = smooth_step_torch((y - float(case_config.layer_y)) / width)
        lmbd = float(case_config.lambda_bg) + float(case_config.lambda_ctr_1) * mask
        mu = float(case_config.mu_bg) + float(case_config.mu_ctr_1) * mask
        return lmbd, mu

    if case_config.name == "weak_interlayer":
        mask = smooth_horizontal_band_torch(y, float(case_config.weak_y_bottom), float(case_config.weak_y_top), width)
        lmbd = float(case_config.lambda_bg) + float(case_config.lambda_ctr_1) * mask
        mu = float(case_config.mu_bg) + float(case_config.mu_ctr_1) * mask
        return lmbd, mu

    mask_1 = smooth_disk_torch(
        x,
        y,
        float(case_config.circle_1_cx),
        float(case_config.circle_1_cy),
        float(case_config.circle_1_r),
        width,
    )
    lmbd = float(case_config.lambda_bg) + float(case_config.lambda_ctr_1) * mask_1
    mu = float(case_config.mu_bg) + float(case_config.mu_ctr_1) * mask_1

    if case_config.name == "double_inclusion":
        mask_2 = smooth_disk_torch(
            x,
            y,
            float(case_config.circle_2_cx),
            float(case_config.circle_2_cy),
            float(case_config.circle_2_r),
            width,
        )
        lmbd = lmbd + float(case_config.lambda_ctr_2) * mask_2
        mu = mu + float(case_config.mu_ctr_2) * mask_2
    return lmbd, mu


def exact_state_torch(points, case_config, load_scale=1.0, load_mode="legacy"):
    import torch

    px = points[:, 0:1]
    py = points[:, 1:2]
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
    lmbd, mu = exact_material_torch(points, case_config)
    trace = exx + eyy
    sxx = lmbd * trace + 2.0 * mu * exx
    syy = lmbd * trace + 2.0 * mu * eyy
    sxy = 2.0 * mu * exy
    return torch.cat((ux, uy, sxx, syy, sxy, lmbd, mu), dim=1)


def exact_body_force_numpy(points, case_config, load_scale=1.0, load_mode="legacy", chunk_size=65536):
    sample_points = np.asarray(points, dtype=float)
    if len(sample_points) == 0:
        zeros = np.zeros((0, 1), dtype=float)
        return zeros, zeros

    x = sample_points[:, 0:1]
    y = sample_points[:, 1:2]
    lmbd, mu = exact_material_numpy(sample_points, case_config)
    lmbd_x, lmbd_y, mu_x, mu_y = exact_material_gradient_xy_numpy(x, y, case_config)
    exx, eyy, exy = exact_strain_numpy(sample_points, case_config, load_scale=load_scale, load_mode=load_mode)
    exx_x, exx_y, eyy_x, eyy_y, exy_x, exy_y = exact_strain_gradient_numpy(
        sample_points,
        case_config,
        load_scale=load_scale,
        load_mode=load_mode,
    )
    trace = exx + eyy
    trace_x = exx_x + eyy_x
    trace_y = exx_y + eyy_y

    sxx_x = lmbd_x * trace + lmbd * trace_x + 2.0 * mu_x * exx + 2.0 * mu * exx_x
    syy_y = lmbd_y * trace + lmbd * trace_y + 2.0 * mu_y * eyy + 2.0 * mu * eyy_y
    sxy_x = 2.0 * mu_x * exy + 2.0 * mu * exy_x
    sxy_y = 2.0 * mu_y * exy + 2.0 * mu * exy_y

    fx = -(sxx_x + sxy_y)
    fy = -(sxy_x + syy_y)
    return np.asarray(fx, dtype=float), np.asarray(fy, dtype=float)


def exact_body_force_xy_numpy(x, y, case_config, load_scale=1.0, load_mode="legacy", chunk_size=65536):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    shape = x.shape
    points = np.column_stack((x.reshape(-1), y.reshape(-1)))
    fx, fy = exact_body_force_numpy(
        points,
        case_config,
        load_scale=load_scale,
        load_mode=load_mode,
        chunk_size=chunk_size,
    )
    return fx.reshape(shape), fy.reshape(shape)


def build_named_mesh(nx, ny):
    from skfem import MeshTri

    xs = np.linspace(0.0, 1.0, int(nx))
    ys = np.linspace(0.0, 1.0, int(ny))
    return MeshTri.init_tensor(xs, ys).with_boundaries(
        {
            "left": lambda x: np.isclose(x[0], 0.0),
            "right": lambda x: np.isclose(x[0], 1.0),
            "bottom": lambda x: np.isclose(x[1], 0.0),
            "top": lambda x: np.isclose(x[1], 1.0),
        }
    )


def build_scalar_element(order):
    from skfem.element import ElementTriP1, ElementTriP2

    if int(order) == 1:
        return ElementTriP1()
    if int(order) == 2:
        return ElementTriP2()
    raise ValueError(f"Unsupported element_order: {order}")


def boundary_component_dofs(dofs_view, component_name):
    indices = []
    for attr_name in ("nodal", "facet"):
        group = getattr(dofs_view, attr_name, None)
        if group is not None and component_name in group:
            indices.append(np.asarray(group[component_name], dtype=int).reshape(-1))
    if not indices:
        return np.zeros((0,), dtype=int)
    return np.unique(np.concatenate(indices))


def evaluate_vector_solution_with_gradients(vector_basis, solution, points):
    sample_points = np.asarray(points, dtype=float)
    if len(sample_points) == 0:
        return np.zeros((2, 0), dtype=float), np.zeros((2, 2, 0), dtype=float)

    coords = sample_points.T
    cells = vector_basis.mesh.element_finder(mapping=vector_basis.mapping)(*coords)
    pts = vector_basis.mapping.invF(coords[:, :, np.newaxis], tind=cells)

    values = np.zeros((2, len(sample_points)), dtype=float)
    gradients = np.zeros((2, 2, len(sample_points)), dtype=float)
    for local_index in range(vector_basis.Nbfun):
        discrete_field = vector_basis.elem.gbasis(
            vector_basis.mapping,
            pts,
            local_index,
            tind=cells,
        )[0]
        dofs = vector_basis.element_dofs[local_index, cells]
        coefficients = np.asarray(solution[dofs], dtype=float).reshape(1, 1, -1)
        values = values + np.asarray(discrete_field.value, dtype=float)[..., 0] * coefficients.reshape(1, -1)
        gradients = gradients + np.asarray(discrete_field.grad, dtype=float)[..., 0] * coefficients
    return values, gradients


def solve_case_fem(case_config, mesh_nx, mesh_ny, element_order, load_scale, load_mode):
    from skfem import Basis, BilinearForm, LinearForm, asm, condense, solve
    from skfem.element import ElementVector
    from skfem.helpers import ddot, sym_grad, trace

    mesh = build_named_mesh(mesh_nx, mesh_ny)
    scalar_element = build_scalar_element(element_order)
    vector_element = ElementVector(scalar_element)
    integration_order = max(10, 2 * int(element_order) + 6)
    vector_basis = Basis(mesh, vector_element, intorder=integration_order)
    scalar_basis = Basis(mesh, scalar_element, intorder=integration_order)

    @BilinearForm
    def variable_elasticity(u, v, w):
        lmbd_q, mu_q = exact_material_xy_numpy(w.x[0], w.x[1], case_config)
        return lmbd_q * trace(sym_grad(u)) * trace(sym_grad(v)) + 2.0 * mu_q * ddot(sym_grad(u), sym_grad(v))

    @LinearForm
    def body_force(v, w):
        fx_q, fy_q = exact_body_force_xy_numpy(
            w.x[0],
            w.x[1],
            case_config,
            load_scale=load_scale,
            load_mode=load_mode,
        )
        return fx_q * v[0] + fy_q * v[1]

    stiffness = asm(variable_elasticity, vector_basis)
    rhs = asm(body_force, vector_basis)
    prescribed = vector_basis.zeros()

    boundary_dofs = vector_basis.get_dofs()
    boundary_flat = boundary_dofs.flatten()
    dof_locations = vector_basis.doflocs

    ux_dofs = boundary_component_dofs(boundary_dofs, "u^1")
    uy_dofs = boundary_component_dofs(boundary_dofs, "u^2")
    ux_points = np.column_stack((dof_locations[0, ux_dofs], dof_locations[1, ux_dofs]))
    uy_points = np.column_stack((dof_locations[0, uy_dofs], dof_locations[1, uy_dofs]))
    prescribed_ux, _ = exact_displacement_numpy(ux_points, case_config, load_scale=load_scale, load_mode=load_mode)
    _, prescribed_uy = exact_displacement_numpy(uy_points, case_config, load_scale=load_scale, load_mode=load_mode)
    prescribed[ux_dofs] = prescribed_ux[:, 0]
    prescribed[uy_dofs] = prescribed_uy[:, 0]

    solution = solve(*condense(stiffness, rhs, x=prescribed, D=boundary_flat))

    discrete_field = vector_basis.interpolate(solution)
    lambda_eval, mu_eval = exact_material_xy_numpy(vector_basis.X[0], vector_basis.X[1], case_config)
    exx = discrete_field.grad[0, 0]
    eyy = discrete_field.grad[1, 1]
    exy = 0.5 * (discrete_field.grad[0, 1] + discrete_field.grad[1, 0])
    trace = exx + eyy
    sxx = lambda_eval * trace + 2.0 * mu_eval * exx
    syy = lambda_eval * trace + 2.0 * mu_eval * eyy
    sxy = 2.0 * mu_eval * exy
    sxx_projection = scalar_basis.project(sxx)
    syy_projection = scalar_basis.project(syy)
    sxy_projection = scalar_basis.project(sxy)

    probe_points = np.asarray(
        [
            [0.1, 0.1],
            [0.5, 0.5],
            [0.8, 0.8],
        ],
        dtype=float,
    )
    probe_lambda, probe_mu = exact_material_numpy(probe_points, case_config)

    return {
        "mesh": mesh,
        "vector_basis": vector_basis,
        "scalar_basis": scalar_basis,
        "solution": solution,
        "probe_points": probe_points,
        "probe_lambda": probe_lambda[:, 0],
        "probe_mu": probe_mu[:, 0],
        "probe_bulk": bulk_from_lambda_mu(probe_lambda[:, 0], probe_mu[:, 0]),
        "sxx_projection": sxx_projection,
        "syy_projection": syy_projection,
        "sxy_projection": sxy_projection,
    }


def evaluate_state_from_solution(solution_bundle, case_config, points):
    sample_points = np.asarray(points, dtype=float)
    if len(sample_points) == 0:
        return np.zeros((0, len(FIELD_NAMES)), dtype=float)
    vector_basis = solution_bundle["vector_basis"]
    displacement, gradients = evaluate_vector_solution_with_gradients(
        vector_basis,
        solution_bundle["solution"],
        sample_points,
    )
    exx = gradients[0, 0, :]
    eyy = gradients[1, 1, :]
    exy = 0.5 * (gradients[0, 1, :] + gradients[1, 0, :])
    lmbd, mu = exact_material_numpy(sample_points, case_config)
    trace = exx + eyy
    sxx = lmbd[:, 0] * trace + 2.0 * mu[:, 0] * exx
    syy = lmbd[:, 0] * trace + 2.0 * mu[:, 0] * eyy
    sxy = 2.0 * mu[:, 0] * exy
    return np.column_stack(
        (
            displacement[0],
            displacement[1],
            np.asarray(sxx, dtype=float),
            np.asarray(syy, dtype=float),
            np.asarray(sxy, dtype=float),
            lmbd[:, 0],
            mu[:, 0],
        )
    )


def build_split_payload(points, true_state, noise_level, seed):
    clean = np.asarray(true_state[:, :2], dtype=float)
    noisy = add_noise(clean, noise_level, seed)
    return {
        "points": np.asarray(points, dtype=float),
        "clean": clean,
        "noisy": noisy,
        "true_state": np.asarray(true_state, dtype=float),
    }


def save_split_payload(save_dir, split_name, payload):
    np.savez(
        _get_save_path(save_dir, "npz", f"{split_name}_set_data.npz"),
        points=payload["points"],
        clean=payload["clean"],
        noisy=payload["noisy"],
        true_state=payload["true_state"],
    )
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
    np.savetxt(
        _get_save_path(save_dir, "txt", f"{split_name}_set_data.txt"),
        combined,
        fmt="%.8e",
        header=header,
    )
    save_json(
        _get_save_path(save_dir, "json", f"{split_name}_set_metadata.json"),
        {
            "split": split_name,
            "num_points": int(len(payload["points"])),
            "field_names": FIELD_NAMES,
        },
    )


def safe_mode_name(load_mode):
    return str(load_mode).replace(" ", "_").replace("-", "_").lower()


def default_run_name(args):
    load_tag = "_".join(safe_mode_name(spec["mode"]) for spec in resolve_load_specs(args.load_scales, args.load_modes))
    return (
        f"{args.case}_fem_smoke_"
        f"{load_tag}_"
        f"p{int(args.element_order)}_"
        f"{int(args.mesh_nx)}x{int(args.mesh_ny)}_"
        f"seed{int(args.seed)}"
    )


def default_save_dir(args):
    relative_root = args.exp_root if os.path.isabs(args.exp_root) else os.path.join(ROOT_DIR, args.exp_root)
    run_name = args.run_name or default_run_name(args)
    return os.path.join(relative_root, "smoke", args.case, run_name)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate FEM teacher data for spatial material inverse smoke experiments.")
    parser.add_argument("--case", choices=["single_material", "layered", "single_inclusion", "double_inclusion", "weak_interlayer"], default="single_material")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mesh_nx", type=int, default=41)
    parser.add_argument("--mesh_ny", type=int, default=41)
    parser.add_argument("--element_order", type=int, choices=[1, 2], default=2)
    parser.add_argument("--load_scales", type=str, default="1.0,1.0,1.0")
    parser.add_argument("--load_modes", type=str, default="biaxial_bulk,pure_shear,uniaxial_y")
    parser.add_argument("--num_boundary", type=int, default=400)
    parser.add_argument("--num_observe", type=int, default=1200)
    parser.add_argument("--num_val_observe", type=int, default=1200)
    parser.add_argument("--num_eval_observe", type=int, default=1200)
    parser.add_argument("--eval_nx", type=int, default=141)
    parser.add_argument("--eval_ny", type=int, default=141)
    parser.add_argument("--noise_level", type=float, default=0.0)
    parser.add_argument("--observation_split_tag", type=str, default="official_softbc_v1")
    parser.add_argument("--observation_cache_dir", type=str, default=DEFAULT_OBSERVATION_CACHE_DIR)
    parser.add_argument("--exp_root", type=str, default=DEFAULT_EXP_ROOT)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    case_config = get_case_config(args.case)
    load_specs = resolve_load_specs(args.load_scales, args.load_modes)

    save_dir = args.save_dir or default_save_dir(args)
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(ROOT_DIR, save_dir)
    if os.path.exists(save_dir) and not args.overwrite:
        raise FileExistsError(f"Save directory already exists: {save_dir}. Use --overwrite only for explicit reruns.")
    ensure_dir(save_dir)

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
    evaluation_grid_points, xx, yy = make_grid(args.eval_nx, args.eval_ny)

    bundle_payload = {
        "field_names": np.asarray(FIELD_NAMES),
        "load_modes": np.asarray([spec["mode"] for spec in load_specs]),
        "load_scales": np.asarray([spec["scale"] for spec in load_specs], dtype=float),
        "boundary_points": boundary_points,
        "train_points": observation_splits["train"],
        "validation_points": observation_splits["val"],
        "evaluation_points": observation_splits["eval"],
        "evaluation_grid_points": evaluation_grid_points,
    }

    per_load_metadata = []
    boundary_true_states = []
    train_true_states = []
    validation_true_states = []
    evaluation_true_states = []
    grid_true_states = []

    for load_index, load_spec in enumerate(load_specs):
        solution_bundle = solve_case_fem(
            case_config=case_config,
            mesh_nx=args.mesh_nx,
            mesh_ny=args.mesh_ny,
            element_order=args.element_order,
            load_scale=float(load_spec["scale"]),
            load_mode=str(load_spec["mode"]),
        )
        mode_name = str(load_spec["mode"])
        mode_tag = safe_mode_name(mode_name)

        boundary_state = evaluate_state_from_solution(solution_bundle, case_config, boundary_points)
        train_state = evaluate_state_from_solution(solution_bundle, case_config, observation_splits["train"])
        validation_state = evaluate_state_from_solution(solution_bundle, case_config, observation_splits["val"])
        evaluation_state = evaluate_state_from_solution(solution_bundle, case_config, observation_splits["eval"])
        grid_state = evaluate_state_from_solution(solution_bundle, case_config, evaluation_grid_points)

        boundary_true_states.append(boundary_state)
        train_true_states.append(train_state)
        validation_true_states.append(validation_state)
        evaluation_true_states.append(evaluation_state)
        grid_true_states.append(grid_state)

        boundary_payload = build_split_payload(boundary_points, boundary_state, noise_level=0.0, seed=args.seed + 4000 + load_index)
        train_payload = build_split_payload(
            observation_splits["train"],
            train_state,
            noise_level=args.noise_level,
            seed=args.seed + 1000 * load_index + 123,
        )
        validation_payload = build_split_payload(
            observation_splits["val"],
            validation_state,
            noise_level=args.noise_level,
            seed=args.seed + 1000 * load_index + 223,
        )
        evaluation_payload = build_split_payload(
            observation_splits["eval"],
            evaluation_state,
            noise_level=args.noise_level,
            seed=args.seed + 1000 * load_index + 323,
        )

        save_split_payload(save_dir, f"boundary_l{load_index}_{mode_tag}", boundary_payload)
        save_split_payload(save_dir, f"train_l{load_index}_{mode_tag}", train_payload)
        save_split_payload(save_dir, f"validation_l{load_index}_{mode_tag}", validation_payload)
        save_split_payload(save_dir, f"evaluation_l{load_index}_{mode_tag}", evaluation_payload)

        np.savez(
            _get_save_path(save_dir, "npz", f"evaluation_grid_l{load_index}_{mode_tag}.npz"),
            points=evaluation_grid_points,
            truth=grid_state,
            xx=xx,
            yy=yy,
            field_names=np.asarray(FIELD_NAMES),
        )

        np.savez(
            _get_save_path(save_dir, "npz", f"mesh_solution_l{load_index}_{mode_tag}.npz"),
            mesh_points=solution_bundle["mesh"].p.T,
            mesh_triangles=solution_bundle["mesh"].t.T,
            vector_dof_locations=solution_bundle["vector_basis"].doflocs.T,
            vector_solution=solution_bundle["solution"],
            scalar_sxx_dofs=solution_bundle["sxx_projection"],
            scalar_syy_dofs=solution_bundle["syy_projection"],
            scalar_sxy_dofs=solution_bundle["sxy_projection"],
            probe_points=solution_bundle["probe_points"],
            probe_lambda=solution_bundle["probe_lambda"],
            probe_mu=solution_bundle["probe_mu"],
            probe_bulk=solution_bundle["probe_bulk"],
            element_order=int(args.element_order),
            load_scale=float(load_spec["scale"]),
            load_mode=mode_name,
        )

        material_probes = []
        for probe_index, point in enumerate(solution_bundle["probe_points"]):
            material_probes.append(
                {
                    "index": int(probe_index),
                    "point": [float(point[0]), float(point[1])],
                    "lambda": float(solution_bundle["probe_lambda"][probe_index]),
                    "mu": float(solution_bundle["probe_mu"][probe_index]),
                    "bulk": float(solution_bundle["probe_bulk"][probe_index]),
                }
            )
        per_load_metadata.append(
            {
                "load_index": int(load_index),
                "load_mode": mode_name,
                "load_scale": float(load_spec["scale"]),
                "material_probes": material_probes,
            }
        )

    boundary_true_states = np.asarray(boundary_true_states, dtype=float)
    train_true_states = np.asarray(train_true_states, dtype=float)
    validation_true_states = np.asarray(validation_true_states, dtype=float)
    evaluation_true_states = np.asarray(evaluation_true_states, dtype=float)
    grid_true_states = np.asarray(grid_true_states, dtype=float)

    np.savez(
        _get_save_path(save_dir, "npz", "teacher_bundle.npz"),
        field_names=np.asarray(FIELD_NAMES),
        load_modes=np.asarray([spec["mode"] for spec in load_specs]),
        load_scales=np.asarray([spec["scale"] for spec in load_specs], dtype=float),
        boundary_points=boundary_points,
        boundary_true_state=boundary_true_states,
        train_points=observation_splits["train"],
        train_true_state=train_true_states,
        validation_points=observation_splits["val"],
        validation_true_state=validation_true_states,
        evaluation_points=observation_splits["eval"],
        evaluation_true_state=evaluation_true_states,
        evaluation_grid_points=evaluation_grid_points,
        evaluation_grid_true_state=grid_true_states,
    )

    meta = {
        "save_dir": save_dir,
        "case": asdict(case_config),
        "field_names": FIELD_NAMES,
        "fem_model": {
            "solver_family": "scikit-fem",
            "model": "small-strain isotropic linear elasticity",
            "kinematics": "plane_strain",
            "boundary_condition_type": "full_dirichlet_from_existing_load_mode",
            "body_force": "exact_body_force_matching_analytic_teacher",
            "element_order": int(args.element_order),
            "mesh_nx": int(args.mesh_nx),
            "mesh_ny": int(args.mesh_ny),
            "estimated_triangles": int(2 * (int(args.mesh_nx) - 1) * (int(args.mesh_ny) - 1)),
        },
        "loads": per_load_metadata,
        "splits": {
            "num_boundary": int(len(boundary_points)),
            "num_train": int(len(observation_splits["train"])),
            "num_validation": int(len(observation_splits["val"])),
            "num_evaluation": int(len(observation_splits["eval"])),
            "observation_cache_path": observation_splits.get("cache_path"),
            "observation_split_tag": str(args.observation_split_tag),
            "noise_level": float(args.noise_level),
        },
        "status": "completed",
    }
    save_json(_get_save_path(save_dir, "json", "teacher_meta.json"), meta)
    save_text(
        _get_save_path(save_dir, "txt", "teacher_summary.txt"),
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
    )
    try:
        render_teacher_run_visuals(save_dir)
    except Exception as exc:
        print(f"[warn] failed to render FEM teacher visuals: {exc}")

    print("=" * 80)
    print("FEM teacher generation finished.")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
