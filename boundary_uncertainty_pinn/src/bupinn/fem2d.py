"""Reproducible 2-D plane-strain FEM benchmark for geotechnical inversion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .io import write_json


@dataclass(frozen=True)
class FEMConfig:
    width: float = 6.0
    depth: float = 3.0
    plate_width: float = 1.0
    nx: int = 96
    ny: int = 48
    settlement: float = -0.05
    seed: int = 42
    k_ref: float = 1.0
    mu_ref: float = 0.45

    @property
    def plate_center(self) -> float:
        return 0.5 * self.width

    @property
    def plate_left(self) -> float:
        return self.plate_center - 0.5 * self.plate_width

    @property
    def plate_right(self) -> float:
        return self.plate_center + 0.5 * self.plate_width


@dataclass
class FEMResult:
    config: FEMConfig
    nodes: np.ndarray
    elements: np.ndarray
    element_centers: np.ndarray
    element_fields: np.ndarray
    node_fields: np.ndarray
    displacements: np.ndarray
    reactions: np.ndarray
    plate_nodes: np.ndarray
    total_reaction_y: float


FIELD_NAMES = ("ux", "uy", "sxx", "syy", "sxy", "K", "mu", "E", "nu")


def k_mu_to_e_nu(k: np.ndarray, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    e = 9.0 * k * mu / (3.0 * k + mu)
    nu = (3.0 * k - 2.0 * mu) / (2.0 * (3.0 * k + mu))
    return e, nu


def lambda_from_k_mu(k: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return k - 2.0 * mu / 3.0


def material_field(x: np.ndarray, y: np.ndarray, config: FEMConfig) -> tuple[np.ndarray, np.ndarray]:
    """Independent K and mu fields with layered soil, weak seam and soft lens."""

    width = float(config.width)
    depth = float(config.depth)
    xr = np.asarray(x, dtype=float) / width
    yr = np.asarray(y, dtype=float) / depth

    layer = 1.0 + 0.22 * np.tanh((0.55 - yr) / 0.075)
    weak_layer = 1.0 - 0.32 * np.exp(-((yr - 0.46 - 0.04 * np.sin(2.0 * np.pi * xr)) ** 2) / 0.0038)
    lens = 1.0 - 0.38 * np.exp(-(((xr - 0.62) ** 2) / 0.010 + ((yr - 0.64) ** 2) / 0.012))
    ripple = 1.0 + 0.035 * np.sin(2.0 * np.pi * xr) * np.sin(np.pi * yr)

    k_factor = layer * weak_layer * lens * ripple

    mu_layer = 1.0 + 0.16 * np.tanh((0.58 - yr) / 0.090)
    mu_weak = 1.0 - 0.26 * np.exp(-((yr - 0.43 - 0.05 * xr) ** 2) / 0.0045)
    mu_lens = 1.0 - 0.30 * np.exp(-(((xr - 0.38) ** 2) / 0.012 + ((yr - 0.70) ** 2) / 0.018))
    mu_ripple = 1.0 + 0.025 * np.cos(2.0 * np.pi * xr) * np.sin(np.pi * yr)

    mu_factor = mu_layer * mu_weak * mu_lens * mu_ripple

    k = config.k_ref * np.clip(k_factor, 0.45, 1.55)
    mu = config.mu_ref * np.clip(mu_factor, 0.48, 1.50)

    # Keep Poisson's ratio in an engineering-reasonable range.
    nu = (3.0 * k - 2.0 * mu) / (2.0 * (3.0 * k + mu))
    too_low = nu < 0.18
    too_high = nu > 0.42
    if np.any(too_low) or np.any(too_high):
        ratio_low = 2.0 * mu[too_low] * (1.0 + 0.18) / (3.0 * (1.0 - 2.0 * 0.18))
        k[too_low] = np.maximum(k[too_low], ratio_low)
        ratio_high = 2.0 * mu[too_high] * (1.0 + 0.42) / (3.0 * (1.0 - 2.0 * 0.42))
        k[too_high] = np.minimum(k[too_high], ratio_high)

    return k.astype(np.float64), mu.astype(np.float64)


def structured_grid_points(nx: int, ny: int, width: float, depth: float) -> np.ndarray:
    xs = np.linspace(0.0, float(width), int(nx))
    ys = np.linspace(0.0, float(depth), int(ny))
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)


def build_mesh(config: FEMConfig) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(0.0, config.width, config.nx + 1)
    ys = np.linspace(0.0, config.depth, config.ny + 1)
    xx, yy = np.meshgrid(xs, ys)
    nodes = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)

    elements = []
    row = config.nx + 1
    for j in range(config.ny):
        for i in range(config.nx):
            n0 = j * row + i
            n1 = n0 + 1
            n2 = n0 + row + 1
            n3 = n0 + row
            elements.append([n0, n1, n2, n3])
    return nodes, np.asarray(elements, dtype=np.int64)


def _quad4_stiffness(coords: np.ndarray, k: float, mu: float) -> np.ndarray:
    lam = float(lambda_from_k_mu(np.asarray(k), np.asarray(mu)))
    dmat = np.asarray(
        [
            [lam + 2.0 * mu, lam, 0.0],
            [lam, lam + 2.0 * mu, 0.0],
            [0.0, 0.0, mu],
        ],
        dtype=np.float64,
    )
    ke = np.zeros((8, 8), dtype=np.float64)
    points = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))
    for xi in points:
        for eta in points:
            dndxi = 0.25 * np.asarray(
                [
                    [-(1.0 - eta), -(1.0 - xi)],
                    [1.0 - eta, -(1.0 + xi)],
                    [1.0 + eta, 1.0 + xi],
                    [-(1.0 + eta), 1.0 - xi],
                ],
                dtype=np.float64,
            )
            jac = dndxi.T @ coords
            det_j = float(np.linalg.det(jac))
            inv_j = np.linalg.inv(jac)
            dndxy = dndxi @ inv_j
            bmat = np.zeros((3, 8), dtype=np.float64)
            for a in range(4):
                bmat[0, 2 * a] = dndxy[a, 0]
                bmat[1, 2 * a + 1] = dndxy[a, 1]
                bmat[2, 2 * a] = dndxy[a, 1]
                bmat[2, 2 * a + 1] = dndxy[a, 0]
            ke += bmat.T @ dmat @ bmat * det_j
    return ke


def solve_fem(config: FEMConfig) -> FEMResult:
    nodes, elements = build_mesh(config)
    centers = np.mean(nodes[elements], axis=1)
    k_elem, mu_elem = material_field(centers[:, 0], centers[:, 1], config)
    e_elem, nu_elem = k_mu_to_e_nu(k_elem, mu_elem)

    ndof = nodes.shape[0] * 2
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for elem, k_val, mu_val in zip(elements, k_elem, mu_elem):
        coords = nodes[elem]
        ke = _quad4_stiffness(coords, float(k_val), float(mu_val))
        dofs = np.ravel([[2 * n, 2 * n + 1] for n in elem])
        rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
        rows.extend(rr.ravel().tolist())
        cols.extend(cc.ravel().tolist())
        vals.extend(ke.ravel().tolist())

    k_global = coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
    force = np.zeros(ndof, dtype=np.float64)

    x = nodes[:, 0]
    y = nodes[:, 1]
    tol = max(config.width / config.nx, config.depth / config.ny) * 1.0e-6
    bottom = np.where(np.isclose(y, 0.0, atol=tol))[0]
    left = np.where(np.isclose(x, 0.0, atol=tol))[0]
    right = np.where(np.isclose(x, config.width, atol=tol))[0]
    plate = np.where(
        np.isclose(y, config.depth, atol=tol)
        & (x >= config.plate_left - tol)
        & (x <= config.plate_right + tol)
    )[0]

    prescribed: dict[int, float] = {}
    for node in bottom:
        prescribed[2 * int(node)] = 0.0
        prescribed[2 * int(node) + 1] = 0.0
    for node in np.union1d(left, right):
        prescribed[2 * int(node)] = 0.0
    for node in plate:
        prescribed[2 * int(node)] = 0.0
        prescribed[2 * int(node) + 1] = float(config.settlement)

    fixed_dofs = np.asarray(sorted(prescribed), dtype=np.int64)
    fixed_vals = np.asarray([prescribed[int(d)] for d in fixed_dofs], dtype=np.float64)
    all_dofs = np.arange(ndof, dtype=np.int64)
    free = np.setdiff1d(all_dofs, fixed_dofs)

    rhs = force[free] - k_global[free][:, fixed_dofs] @ fixed_vals
    u = np.zeros(ndof, dtype=np.float64)
    u[fixed_dofs] = fixed_vals
    u[free] = spsolve(k_global[free][:, free], rhs)

    reactions = k_global @ u - force
    total_ry = float(np.sum(reactions[2 * plate + 1]))

    element_fields = np.column_stack([k_elem, mu_elem, e_elem, nu_elem])
    node_fields = _nodal_average(nodes.shape[0], elements, element_fields)
    displacements = u.reshape(-1, 2)

    return FEMResult(
        config=config,
        nodes=nodes.astype(np.float32),
        elements=elements,
        element_centers=centers.astype(np.float32),
        element_fields=element_fields.astype(np.float32),
        node_fields=node_fields.astype(np.float32),
        displacements=displacements.astype(np.float32),
        reactions=reactions.reshape(-1, 2).astype(np.float32),
        plate_nodes=plate.astype(np.int64),
        total_reaction_y=total_ry,
    )


def _nodal_average(num_nodes: int, elements: np.ndarray, elem_values: np.ndarray) -> np.ndarray:
    out = np.zeros((num_nodes, elem_values.shape[1]), dtype=np.float64)
    count = np.zeros((num_nodes, 1), dtype=np.float64)
    for elem, values in zip(elements, elem_values):
        out[elem] += values
        count[elem] += 1.0
    return out / np.maximum(count, 1.0)


class FEMInterpolator:
    def __init__(self, result: FEMResult):
        self.result = result
        fields = nodal_all_fields(result)
        self._linear = [LinearNDInterpolator(result.nodes, fields[:, i]) for i in range(fields.shape[1])]
        self._nearest = [NearestNDInterpolator(result.nodes, fields[:, i]) for i in range(fields.shape[1])]

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        cols = []
        for lin, near in zip(self._linear, self._nearest):
            values = lin(points)
            missing = ~np.isfinite(values)
            if np.any(missing):
                values = np.asarray(values)
                values[missing] = near(points[missing])
            cols.append(values.astype(np.float32))
        return np.column_stack(cols).astype(np.float32)


def nodal_all_fields(result: FEMResult) -> np.ndarray:
    k = result.node_fields[:, 0:1]
    mu = result.node_fields[:, 1:2]
    e = result.node_fields[:, 2:3]
    nu = result.node_fields[:, 3:4]
    stress = recover_nodal_stress(result)
    return np.column_stack([result.displacements, stress, k, mu, e, nu]).astype(np.float32)


def recover_nodal_stress(result: FEMResult) -> np.ndarray:
    elem_stress = np.zeros((len(result.elements), 3), dtype=np.float64)
    for idx, elem in enumerate(result.elements):
        coords = result.nodes[elem].astype(np.float64)
        u_elem = result.displacements[elem].reshape(8).astype(np.float64)
        k_val = float(result.element_fields[idx, 0])
        mu_val = float(result.element_fields[idx, 1])
        lam = float(lambda_from_k_mu(np.asarray(k_val), np.asarray(mu_val)))
        dmat = np.asarray(
            [
                [lam + 2.0 * mu_val, lam, 0.0],
                [lam, lam + 2.0 * mu_val, 0.0],
                [0.0, 0.0, mu_val],
            ],
            dtype=np.float64,
        )
        dndxi = 0.25 * np.asarray(
            [
                [-1.0, -1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [-1.0, 1.0],
            ],
            dtype=np.float64,
        )
        jac = dndxi.T @ coords
        dndxy = dndxi @ np.linalg.inv(jac)
        bmat = np.zeros((3, 8), dtype=np.float64)
        for a in range(4):
            bmat[0, 2 * a] = dndxy[a, 0]
            bmat[1, 2 * a + 1] = dndxy[a, 1]
            bmat[2, 2 * a] = dndxy[a, 1]
            bmat[2, 2 * a + 1] = dndxy[a, 0]
        elem_stress[idx] = dmat @ (bmat @ u_elem)
    return _nodal_average(result.nodes.shape[0], result.elements, elem_stress).astype(np.float32)


def save_fem_result(result: FEMResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "fem_solution.npz",
        nodes=result.nodes,
        elements=result.elements,
        element_centers=result.element_centers,
        element_fields=result.element_fields,
        node_fields=result.node_fields,
        displacements=result.displacements,
        reactions=result.reactions,
        plate_nodes=result.plate_nodes,
        total_reaction_y=np.asarray(result.total_reaction_y, dtype=np.float64),
        field_names=np.asarray(FIELD_NAMES),
    )
    write_json(out_dir / "fem_config.json", asdict(result.config))
    write_json(
        out_dir / "fem_summary.json",
        {
            "num_nodes": int(result.nodes.shape[0]),
            "num_elements": int(result.elements.shape[0]),
            "num_plate_nodes": int(result.plate_nodes.shape[0]),
            "total_reaction_y": float(result.total_reaction_y),
            "ux_min": float(np.min(result.displacements[:, 0])),
            "ux_max": float(np.max(result.displacements[:, 0])),
            "uy_min": float(np.min(result.displacements[:, 1])),
            "uy_max": float(np.max(result.displacements[:, 1])),
            "K_min": float(np.min(result.node_fields[:, 0])),
            "K_max": float(np.max(result.node_fields[:, 0])),
            "mu_min": float(np.min(result.node_fields[:, 1])),
            "mu_max": float(np.max(result.node_fields[:, 1])),
            "nu_min": float(np.min(result.node_fields[:, 3])),
            "nu_max": float(np.max(result.node_fields[:, 3])),
        },
    )


def load_fem_result(path: Path) -> FEMResult:
    data = np.load(path, allow_pickle=False)
    config_path = path.with_name("fem_config.json")
    if config_path.exists():
        import json

        cfg = FEMConfig(**json.loads(config_path.read_text(encoding="utf-8")))
    else:
        cfg = FEMConfig()
    return FEMResult(
        config=cfg,
        nodes=data["nodes"],
        elements=data["elements"],
        element_centers=data["element_centers"],
        element_fields=data["element_fields"],
        node_fields=data["node_fields"],
        displacements=data["displacements"],
        reactions=data["reactions"],
        plate_nodes=data["plate_nodes"],
        total_reaction_y=float(np.asarray(data["total_reaction_y"]).item()),
    )


def make_observation_points(config: FEMConfig, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    margin_x = 0.08 * config.width
    margin_bottom = 0.08 * config.depth
    margin_top = 0.10 * config.depth
    x = rng.uniform(margin_x, config.width - margin_x, int(count))
    y = rng.uniform(margin_bottom, config.depth - margin_top, int(count))
    return np.column_stack([x, y]).astype(np.float32)


def make_validation_points(config: FEMConfig, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + 7919)
    x = rng.uniform(0.02 * config.width, 0.98 * config.width, int(count))
    y = rng.uniform(0.02 * config.depth, 0.98 * config.depth, int(count))
    return np.column_stack([x, y]).astype(np.float32)


def make_anchor_points(config: FEMConfig, count: int) -> np.ndarray:
    count = int(count)
    if count <= 0:
        return np.empty((0, 2), dtype=np.float32)
    x = np.linspace(config.plate_left, config.plate_right, count)
    y = np.full_like(x, config.depth)
    return np.column_stack([x, y]).astype(np.float32)


def make_reaction_points(config: FEMConfig, count: int) -> tuple[np.ndarray, np.ndarray]:
    count = max(2, int(count))
    x = np.linspace(config.plate_left, config.plate_right, count)
    y = np.full_like(x, config.depth)
    weights = np.ones((count, 1), dtype=np.float32) * (config.plate_width / float(count - 1))
    weights[0, 0] *= 0.5
    weights[-1, 0] *= 0.5
    return np.column_stack([x, y]).astype(np.float32), weights


def plate_mode_values_np(x: np.ndarray, num_modes: int, config: FEMConfig) -> np.ndarray:
    if num_modes <= 0:
        return np.empty((len(x), 0), dtype=np.float32)
    local = (x[:, 0:1] - config.plate_center) / (0.5 * config.plate_width)
    modes = [np.ones_like(local)]
    if num_modes >= 2:
        modes.append(local)
    if num_modes >= 3:
        modes.append(2.0 * local**2 - 1.0)
    if num_modes >= 4:
        modes.append(np.sin(np.pi * (local + 1.0) / 2.0))
    if num_modes >= 5:
        modes.append(np.sin(np.pi * local))
    if num_modes > 5:
        raise ValueError(f"Unsupported num_modes={num_modes}")
    return np.hstack(modes[:num_modes]).astype(np.float32)


def fem_observation_design_np(x: np.ndarray, num_modes: int, config: FEMConfig) -> np.ndarray:
    if num_modes <= 0:
        return np.empty((2 * len(x), 0), dtype=np.float32)
    y_lift = ((x[:, 1:2] - 0.0) / max(config.depth, 1.0e-12)).astype(np.float32)
    top_modes = plate_mode_values_np(x, num_modes, config)
    u_modes = np.zeros_like(top_modes, dtype=np.float32)
    v_modes = y_lift * top_modes
    return np.vstack([u_modes, v_modes]).astype(np.float32)


def add_displacement_noise(values: np.ndarray, level: float, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    level = float(level)
    if level <= 0.0:
        return values
    rng = np.random.default_rng(int(seed))
    scale = np.sqrt(np.mean(np.sum(values**2, axis=1))) + 1.0e-12
    return (values + rng.normal(0.0, level * scale, size=values.shape)).astype(np.float32)
