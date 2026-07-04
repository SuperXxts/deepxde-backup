"""Analytical benchmark definitions for boundary-uncertainty inversion."""

from __future__ import annotations

import numpy as np


def make_grid(n: int, include_boundary: bool = True) -> np.ndarray:
    """Return an n by n grid on the unit square."""
    if include_boundary:
        xs = np.linspace(0.0, 1.0, n)
        ys = np.linspace(0.0, 1.0, n)
    else:
        xs = np.linspace(0.0, 1.0, n + 2)[1:-1]
        ys = np.linspace(0.0, 1.0, n + 2)[1:-1]
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)


def material_E(x: np.ndarray) -> np.ndarray:
    """Synthetic heterogeneous Young's modulus field."""
    xp = x[:, 0:1]
    yp = x[:, 1:2]
    layer = 1.0 + 0.22 * np.tanh((yp - 0.55) / 0.045)
    lens_soft = -0.18 * np.exp(-((xp - 0.68) ** 2 / 0.018 + (yp - 0.35) ** 2 / 0.012))
    lens_stiff = 0.16 * np.exp(-((xp - 0.32) ** 2 / 0.012 + (yp - 0.72) ** 2 / 0.018))
    ripple = 0.04 * np.sin(2.0 * np.pi * xp) * np.sin(np.pi * yp)
    return np.clip(layer + lens_soft + lens_stiff + ripple, 0.45, 1.65).astype(np.float32)


def material_K_mu_factors(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic independent bulk/shear modulus factors for the formal method tests."""
    xp = x[:, 0:1]
    yp = x[:, 1:2]
    k_layer = 1.0 + 0.20 * np.tanh((yp - 0.55) / 0.055)
    k_lens = -0.16 * np.exp(-((xp - 0.68) ** 2 / 0.020 + (yp - 0.35) ** 2 / 0.014))
    k_ripple = 0.035 * np.sin(2.0 * np.pi * xp) * np.sin(np.pi * yp)
    mu_layer = 1.0 + 0.13 * np.tanh((yp - 0.48) / 0.070)
    mu_lens = 0.18 * np.exp(-((xp - 0.32) ** 2 / 0.014 + (yp - 0.72) ** 2 / 0.020))
    mu_shear_band = -0.12 * np.exp(-((yp - 0.40 - 0.15 * xp) ** 2 / 0.010))
    mu_ripple = 0.025 * np.cos(2.0 * np.pi * xp) * np.sin(np.pi * yp)
    k_factor = np.clip(k_layer + k_lens + k_ripple, 0.45, 1.65)
    mu_factor = np.clip(mu_layer + mu_lens + mu_shear_band + mu_ripple, 0.45, 1.65)
    return k_factor.astype(np.float32), mu_factor.astype(np.float32)


def reference_k_mu(nu: float = 0.30) -> tuple[float, float]:
    """Reference K and mu for unit Young's modulus."""
    k0 = 1.0 / (3.0 * (1.0 - 2.0 * nu))
    mu0 = 1.0 / (2.0 * (1.0 + nu))
    return float(k0), float(mu0)


def lame_from_E(E: np.ndarray, nu: float = 0.30) -> tuple[np.ndarray, np.ndarray]:
    """Plane-strain Lame parameters from Young's modulus and Poisson's ratio."""
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return lam.astype(np.float32), mu.astype(np.float32)


def lambda_from_k_mu(k: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return (k - 2.0 * mu / 3.0).astype(np.float32)


def base_displacement(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Manufactured displacement shape before boundary/load amplitude scaling."""
    xp = x[:, 0:1]
    yp = x[:, 1:2]
    ux = 0.05 * np.sin(np.pi * xp) * np.sin(np.pi * yp) * (1.0 - yp)
    uy = yp * (0.15 + 0.04 * np.sin(np.pi * xp) + 0.02 * np.sin(2.0 * np.pi * xp) * yp)
    return ux.astype(np.float32), uy.astype(np.float32)


def displacement(x: np.ndarray, amplitude: float = 1.0) -> np.ndarray:
    """Return [ux, uy] displacement observations."""
    ux, uy = base_displacement(x)
    return (amplitude * np.hstack([ux, uy])).astype(np.float32)


def strain_np(x: np.ndarray, amplitude: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytical strain components for the manufactured displacement."""
    xp = x[:, 0:1]
    yp = x[:, 1:2]
    dux_dx = 0.05 * np.pi * np.cos(np.pi * xp) * np.sin(np.pi * yp) * (1.0 - yp)
    dux_dy = 0.05 * np.sin(np.pi * xp) * (
        np.pi * np.cos(np.pi * yp) * (1.0 - yp) - np.sin(np.pi * yp)
    )
    duy_dx = yp * (
        0.04 * np.pi * np.cos(np.pi * xp)
        + 0.04 * np.pi * np.cos(2.0 * np.pi * xp) * yp
    )
    duy_dy = (
        0.15
        + 0.04 * np.sin(np.pi * xp)
        + 0.04 * np.sin(2.0 * np.pi * xp) * yp
    )
    exx = amplitude * dux_dx
    eyy = amplitude * duy_dy
    exy = 0.5 * amplitude * (dux_dy + duy_dx)
    return exx.astype(np.float32), eyy.astype(np.float32), exy.astype(np.float32)


def stress_np(x: np.ndarray, amplitude: float = 1.0, nu: float = 0.30) -> np.ndarray:
    """Return [sxx, syy, sxy] from the synthetic material and displacement."""
    E = material_E(x)
    lam, mu = lame_from_E(E, nu)
    exx, eyy, exy = strain_np(x, amplitude)
    sxx = (2.0 * mu + lam) * exx + lam * eyy
    syy = lam * exx + (2.0 * mu + lam) * eyy
    sxy = 2.0 * mu * exy
    return np.hstack([sxx, syy, sxy]).astype(np.float32)


def stress_k_mu_np(x: np.ndarray, amplitude: float = 1.0, nu: float = 0.30) -> np.ndarray:
    """Return stress from independent K/mu synthetic fields."""
    k0, mu0 = reference_k_mu(nu)
    k_factor, mu_factor = material_K_mu_factors(x)
    k = k0 * k_factor
    mu = mu0 * mu_factor
    lam = lambda_from_k_mu(k, mu)
    exx, eyy, exy = strain_np(x, amplitude)
    sxx = (2.0 * mu + lam) * exx + lam * eyy
    syy = lam * exx + (2.0 * mu + lam) * eyy
    sxy = 2.0 * mu * exy
    return np.hstack([sxx, syy, sxy]).astype(np.float32)


def all_fields(x: np.ndarray, amplitude: float = 1.0, nu: float = 0.30) -> np.ndarray:
    """Return [ux, uy, sxx, syy, sxy, E]."""
    return np.hstack([displacement(x, amplitude), stress_np(x, amplitude, nu), material_E(x)]).astype(
        np.float32
    )


def all_fields_k_mu(x: np.ndarray, amplitude: float = 1.0, nu: float = 0.30) -> np.ndarray:
    """Return [ux, uy, sxx, syy, sxy, K, mu, E, nu]."""
    k0, mu0 = reference_k_mu(nu)
    k_factor, mu_factor = material_K_mu_factors(x)
    k = k0 * k_factor
    mu = mu0 * mu_factor
    e = 9.0 * k * mu / (3.0 * k + mu)
    nu_field = (3.0 * k - 2.0 * mu) / (2.0 * (3.0 * k + mu))
    return np.hstack(
        [displacement(x, amplitude), stress_k_mu_np(x, amplitude, nu), k, mu, e, nu_field]
    ).astype(np.float32)


def add_noise(values: np.ndarray, noise_level: float, seed: int) -> np.ndarray:
    """Add Gaussian noise scaled by component-wise standard deviation."""
    if noise_level <= 0.0:
        return values.astype(np.float32)
    rng = np.random.default_rng(seed)
    scale = np.std(values, axis=0, keepdims=True)
    scale = np.maximum(scale, 1e-8)
    noisy = values + noise_level * scale * rng.standard_normal(values.shape)
    return noisy.astype(np.float32)


def boundary_anchor_points(count: int) -> np.ndarray:
    """Sparse, experimentally plausible boundary displacement anchor locations."""
    if count <= 0:
        return np.empty((0, 2), dtype=np.float32)
    pools: list[tuple[float, float]] = []
    top_x = np.linspace(0.10, 0.90, max(2, count // 2 + 2))
    bottom_x = np.linspace(0.10, 0.90, max(2, count // 4 + 2))
    side_y = np.linspace(0.20, 0.80, max(2, count // 4 + 2))
    pools.extend([(float(x), 1.0) for x in top_x])
    pools.extend([(float(x), 0.0) for x in bottom_x])
    pools.extend([(0.0, float(y)) for y in side_y])
    pools.extend([(1.0, float(y)) for y in side_y])
    pts = np.array(pools[:count], dtype=np.float32)
    return pts
