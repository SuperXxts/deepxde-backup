"""Low-dimensional boundary modes for boundary-material decoupling."""

from __future__ import annotations

import numpy as np


def top_shape_np(x: np.ndarray) -> np.ndarray:
    """Reference vertical displacement shape on the top boundary."""
    xp = x[:, 0:1]
    return 0.15 + 0.04 * np.sin(np.pi * xp) + 0.02 * np.sin(2.0 * np.pi * xp)


def top_mode_values_np(x: np.ndarray, num_modes: int) -> np.ndarray:
    """Return top-boundary shape modes evaluated at points x.

    The first mode is the reference loading shape. Additional modes describe
    low-dimensional boundary-shape errors.
    """
    if num_modes <= 0:
        return np.empty((len(x), 0), dtype=np.float32)
    xp = x[:, 0:1]
    modes = [top_shape_np(x)]
    if num_modes >= 2:
        modes.append(0.10 * (2.0 * xp - 1.0))
    if num_modes >= 3:
        modes.append(0.10 * np.sin(np.pi * xp))
    if num_modes >= 4:
        modes.append(0.10 * np.sin(2.0 * np.pi * xp))
    if num_modes >= 5:
        modes.append(0.10 * np.cos(np.pi * xp))
    if num_modes > 5:
        raise ValueError(f"Unsupported num_modes={num_modes}; use 0..5.")
    return np.hstack(modes[:num_modes]).astype(np.float32)


def domain_mode_values_np(x: np.ndarray, num_modes: int) -> tuple[np.ndarray, np.ndarray]:
    """Return low-dimensional displacement influence modes in the domain.

    The current analytical benchmark uses kinematic lifting:
    u_beta = 0, v_beta = y * phi_j(x). It exactly satisfies zero vertical
    boundary contribution at the bottom and phi_j on the top boundary.
    """
    if num_modes <= 0:
        zeros = np.empty((len(x), 0), dtype=np.float32)
        return zeros, zeros
    y = x[:, 1:2]
    top_modes = top_mode_values_np(x, num_modes)
    u_modes = np.zeros_like(top_modes, dtype=np.float32)
    v_modes = (y * top_modes).astype(np.float32)
    return u_modes, v_modes


def observation_design_np(x: np.ndarray, num_modes: int) -> np.ndarray:
    """Build the flattened displacement design matrix for observation points.

    The flattened residual order is [ux_1..ux_N, uy_1..uy_N].
    """
    u_modes, v_modes = domain_mode_values_np(x, num_modes)
    if num_modes <= 0:
        return np.empty((2 * len(x), 0), dtype=np.float32)
    return np.vstack([u_modes, v_modes]).astype(np.float32)


def projection_matrix_np(design: np.ndarray, ridge: float = 1.0e-8) -> np.ndarray:
    """Return the ridge-regularized projection onto the boundary-mode subspace."""
    if design.size == 0 or design.shape[1] == 0:
        return np.zeros((design.shape[0], design.shape[0]), dtype=np.float32)
    gram = design.T @ design
    gram = gram + float(ridge) * np.eye(gram.shape[0], dtype=np.float32)
    return (design @ np.linalg.solve(gram, design.T)).astype(np.float32)


def trapezoid_weights(n: int) -> np.ndarray:
    """Trapezoidal integration weights on [0, 1]."""
    n = max(2, int(n))
    weights = np.ones((n, 1), dtype=np.float32) / float(n - 1)
    weights[0, 0] *= 0.5
    weights[-1, 0] *= 0.5
    return weights
