"""Residual projection helpers for boundary-material decoupling."""

from __future__ import annotations

import numpy as np
import torch


class ObservationProjector:
    """Project displacement observation residuals into boundary/material parts.

    DeepXDE calls PointSetOperatorBC functions with all training points and then
    slices the returned tensor by the current boundary-condition block. We use
    the NumPy all-point array passed by DeepXDE to cache row indices once, while
    avoiding any GPU-to-CPU synchronization from the network outputs. The
    projection is evaluated in low-rank form:

        P r = D (D^T D + ridge I)^-1 D^T r

    where D contains low-dimensional boundary-induced displacement modes.
    """

    def __init__(self, obs_x: np.ndarray, obs_u: np.ndarray, design: np.ndarray, ridge: float):
        self.obs_x = np.asarray(obs_x, dtype=np.float32)
        self.obs_u = np.asarray(obs_u, dtype=np.float32)
        self.design_np = np.asarray(design, dtype=np.float32)
        self.ridge = float(ridge)
        self._cache: dict[tuple[torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._index_cache: dict[tuple[int, int, int], np.ndarray] = {}
        self.lookup = {
            tuple(np.round(pt, decimals=8).tolist()): i
            for i, pt in enumerate(self.obs_x[:, :2])
        }

    def _tensors(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (device, dtype)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        target = torch.as_tensor(self.obs_u, dtype=dtype, device=device)
        design = torch.as_tensor(self.design_np, dtype=dtype, device=device)
        if design.numel() == 0 or design.shape[1] == 0:
            gram_inv = torch.empty((0, 0), dtype=dtype, device=device)
        else:
            eye = torch.eye(design.shape[1], dtype=dtype, device=device)
            gram = design.T @ design + self.ridge * eye
            gram_inv = torch.linalg.inv(gram)
        cached = (target, design, gram_inv)
        self._cache[key] = cached
        return cached

    def _row_index(self, X, occurrence: int = 0) -> np.ndarray:
        x_np = np.asarray(X, dtype=np.float32)
        occurrence = int(occurrence)
        key = (id(X), len(x_np), occurrence)
        cached = self._index_cache.get(key)
        if cached is not None:
            return cached
        matches: list[list[int]] = [[] for _ in range(len(self.obs_x))]
        rounded = np.round(x_np[:, :2], decimals=8)
        for row, pt in enumerate(rounded):
            obs_idx = self.lookup.get(tuple(pt.tolist()), -1)
            if obs_idx >= 0:
                matches[obs_idx].append(row)
        row_for_obs = np.full(len(self.obs_x), -1, dtype=np.int64)
        for obs_idx, rows in enumerate(matches):
            if len(rows) > occurrence:
                row_for_obs[obs_idx] = rows[occurrence]
        if np.any(row_for_obs < 0):
            missing = int(np.sum(row_for_obs < 0))
            counts = sorted({len(rows) for rows in matches})
            raise RuntimeError(
                "Observation projector missing "
                f"{missing} observation rows for occurrence={occurrence}; "
                f"available duplicate counts={counts}."
            )
        self._index_cache[key] = row_for_obs
        return row_for_obs

    def projected_component(
        self,
        inputs,
        outputs,
        X,
        component: int,
        part: str,
        update_path: str = "joint",
        occurrence: int = 0,
    ):
        n_obs = len(self.obs_x)
        row_index_np = self._row_index(X, occurrence=occurrence)
        row_index = torch.as_tensor(row_index_np, dtype=torch.long, device=outputs.device)
        point_outputs = outputs[row_index]
        if update_path == "joint" or outputs.shape[1] < 11:
            pred = point_outputs[:, 0:2]
        else:
            core = point_outputs[:, 7:9]
            boundary = point_outputs[:, 9:11]
            if update_path == "material":
                pred = core + boundary.detach()
            elif update_path == "boundary":
                pred = core.detach() + boundary
            else:
                raise ValueError(f"Unknown update_path: {update_path}")
        target, design, gram_inv = self._tensors(outputs.device, outputs.dtype)
        flat = torch.cat([pred[:, 0] - target[:, 0], pred[:, 1] - target[:, 1]], dim=0)
        if design.numel() == 0 or design.shape[1] == 0:
            boundary_flat = torch.zeros_like(flat)
        else:
            coeff = gram_inv @ (design.T @ flat)
            boundary_flat = design @ coeff
        if part == "boundary":
            projected = boundary_flat
        elif part == "material":
            projected = flat - boundary_flat
        else:
            raise ValueError(f"Unknown projection part: {part}")
        values = projected[int(component) * n_obs : (int(component) + 1) * n_obs].reshape(-1, 1)
        result = torch.zeros((outputs.shape[0], 1), dtype=outputs.dtype, device=outputs.device)
        result[row_index, 0:1] = values
        return result


class ReactionIntegral:
    """Fast top-boundary resultant reaction operator."""

    def __init__(self, weights: np.ndarray):
        self.weights_np = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
        self._cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self._index_cache: dict[tuple[int, int], np.ndarray] = {}

    def _weights(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device, dtype)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        weights = torch.as_tensor(self.weights_np, dtype=dtype, device=device)
        self._cache[key] = weights
        return weights

    def _row_index(self, X) -> np.ndarray:
        key = (id(X), len(X))
        cached = self._index_cache.get(key)
        if cached is not None:
            return cached
        n = len(self.weights_np)
        x_np = np.asarray(X, dtype=np.float32)
        top_rows = np.where(np.isclose(x_np[:, 1], 1.0, atol=1.0e-7))[0]
        if len(top_rows) < n:
            raise RuntimeError(f"Reaction integral expected {n} top-boundary rows, got {len(top_rows)}.")
        # PointSetOperatorBC places its own point set as a contiguous block. If
        # other top-boundary BC points also exist, use the first block with
        # matching length and monotone x coordinates; this is stable for the
        # deterministic DeepXDE BC-point ordering used here.
        best = None
        for start in range(0, len(top_rows) - n + 1):
            rows = top_rows[start : start + n]
            xs = x_np[rows, 0]
            if np.all(np.diff(xs) >= -1.0e-7):
                best = rows
                break
        if best is None:
            best = top_rows[:n]
        self._index_cache[key] = best.astype(np.int64)
        return self._index_cache[key]

    def __call__(self, inputs, outputs, X):
        weights = self._weights(outputs.device, outputs.dtype)
        row_index_np = self._row_index(X)
        row_index = torch.as_tensor(row_index_np, dtype=torch.long, device=outputs.device)
        syy = outputs[row_index, 3:4]
        resultant = torch.sum(syy * weights)
        result = torch.zeros((outputs.shape[0], 1), dtype=outputs.dtype, device=outputs.device)
        result[row_index, 0:1] = resultant
        return result
