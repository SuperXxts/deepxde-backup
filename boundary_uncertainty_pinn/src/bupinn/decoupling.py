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

    def __init__(
        self,
        obs_x: np.ndarray,
        obs_u: np.ndarray,
        design: np.ndarray,
        ridge: float,
        core_start: int | None = 7,
        boundary_start: int | None = 9,
        prediction_start: int = 0,
        cache_components: int = 1,
    ):
        self.obs_x = np.asarray(obs_x, dtype=np.float32)
        self.obs_u = np.asarray(obs_u, dtype=np.float32)
        self.design_np = np.asarray(design, dtype=np.float32)
        self.ridge = float(ridge)
        self.core_start = None if core_start is None else int(core_start)
        self.boundary_start = None if boundary_start is None else int(boundary_start)
        self.prediction_start = int(prediction_start)
        self.cache_components = int(cache_components)
        self._cache: dict[tuple[torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._index_cache: dict[tuple[int, int, int], np.ndarray] = {}
        self._index_tensor_cache: dict[tuple[int, int, int, torch.device], torch.Tensor] = {}
        self._projection_cache: dict[str, object] = {}
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

    def _row_index_tensor(self, X, occurrence: int, device: torch.device) -> torch.Tensor:
        x_len = len(X)
        key = (id(X), x_len, int(occurrence), device)
        cached = self._index_tensor_cache.get(key)
        if cached is not None:
            return cached
        row_index = torch.as_tensor(self._row_index(X, occurrence=occurrence), dtype=torch.long, device=device)
        self._index_tensor_cache[key] = row_index
        return row_index

    def _has_split_outputs(self, outputs) -> bool:
        if self.core_start is None or self.boundary_start is None:
            return False
        return outputs.shape[1] >= self.boundary_start + 2

    def _split_prediction(self, point_outputs, update_path: str):
        if update_path == "joint" or not self._has_split_outputs(point_outputs):
            start = self.prediction_start
            return point_outputs[:, start : start + 2]
        core = point_outputs[:, self.core_start : self.core_start + 2]
        boundary = point_outputs[:, self.boundary_start : self.boundary_start + 2]
        if update_path == "material":
            return core + boundary.detach()
        if update_path == "boundary":
            return core.detach() + boundary
        raise ValueError(f"Unknown update_path: {update_path}")

    def _projection_values(self, outputs, X, source_occurrence: int, update_path: str) -> dict[tuple[str, int], torch.Tensor]:
        key = (
            id(outputs),
            id(X),
            outputs.shape[0],
            outputs.shape[1],
            str(outputs.device),
            str(outputs.dtype),
            self.core_start,
            self.boundary_start,
            self.prediction_start,
        )
        cached = self._projection_cache.get("entry")
        if isinstance(cached, dict) and cached.get("key") == key:
            return cached["values"]  # type: ignore[return-value]

        row_index = self._row_index_tensor(X, source_occurrence, outputs.device)
        point_outputs = outputs[row_index]
        target, design, gram_inv = self._tensors(outputs.device, outputs.dtype)

        pred_material = self._split_prediction(point_outputs, update_path="material" if update_path != "joint" else "joint")
        pred_boundary = self._split_prediction(point_outputs, update_path="boundary" if update_path != "joint" else "joint")
        flat_material = torch.cat(
            [pred_material[:, 0] - target[:, 0], pred_material[:, 1] - target[:, 1]],
            dim=0,
        )
        flat_boundary_path = torch.cat(
            [pred_boundary[:, 0] - target[:, 0], pred_boundary[:, 1] - target[:, 1]],
            dim=0,
        )
        if design.numel() == 0 or design.shape[1] == 0:
            material_boundary_flat = torch.zeros_like(flat_material)
            boundary_flat = torch.zeros_like(flat_boundary_path)
        else:
            coeff_material = gram_inv @ (design.T @ flat_material)
            coeff_boundary = gram_inv @ (design.T @ flat_boundary_path)
            material_boundary_flat = design @ coeff_material
            boundary_flat = design @ coeff_boundary
        material_projected = flat_material - material_boundary_flat
        n_obs = len(self.obs_x)
        values = {
            ("material", 0): material_projected[:n_obs].reshape(-1, 1),
            ("material", 1): material_projected[n_obs : 2 * n_obs].reshape(-1, 1),
            ("boundary", 0): boundary_flat[:n_obs].reshape(-1, 1),
            ("boundary", 1): boundary_flat[n_obs : 2 * n_obs].reshape(-1, 1),
        }
        self._projection_cache["entry"] = {"key": key, "values": values, "uses": 0}
        return values

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
        if part not in {"boundary", "material"}:
            raise ValueError(f"Unknown projection part: {part}")
        values = self._projection_values(outputs, X, occurrence, update_path)[(part, int(component))]
        row_index = self._row_index_tensor(X, occurrence, outputs.device)
        result = torch.zeros((outputs.shape[0], 1), dtype=outputs.dtype, device=outputs.device)
        result[row_index, 0:1] = values
        cached = self._projection_cache.get("entry")
        if isinstance(cached, dict):
            cached["uses"] = int(cached.get("uses", 0)) + 1
            if int(cached["uses"]) >= max(1, self.cache_components):
                self._projection_cache.clear()
        return result


class ReactionIntegral:
    """Fast resultant reaction operator with cached point and tensor indices."""

    def __init__(
        self,
        weights: np.ndarray,
        boundary_y: float = 1.0,
        points: np.ndarray | None = None,
        stress_index: int = 3,
        scale: float = 1.0,
        occurrence: int = 0,
        fill_all: bool = False,
    ):
        self.weights_np = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
        self.boundary_y = float(boundary_y)
        self.points = None if points is None else np.asarray(points, dtype=np.float32)
        self.stress_index = int(stress_index)
        self.scale = float(scale)
        self.occurrence = int(occurrence)
        self.fill_all = bool(fill_all)
        self._cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self._index_cache: dict[tuple[int, int, int], np.ndarray] = {}
        self._index_tensor_cache: dict[tuple[int, int, int, torch.device], torch.Tensor] = {}
        self.lookup = None
        if self.points is not None:
            self.lookup = {
                tuple(np.round(pt, decimals=8).tolist()): i
                for i, pt in enumerate(self.points[:, :2])
            }

    def _weights(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device, dtype)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        weights = torch.as_tensor(self.weights_np, dtype=dtype, device=device)
        self._cache[key] = weights
        return weights

    def _row_index(self, X) -> np.ndarray:
        key = (id(X), len(X), self.occurrence)
        cached = self._index_cache.get(key)
        if cached is not None:
            return cached
        x_np = np.asarray(X, dtype=np.float32)
        n = len(self.weights_np)
        if self.points is not None:
            matches: list[list[int]] = [[] for _ in range(len(self.points))]
            rounded = np.round(x_np[:, :2], decimals=8)
            if self.lookup is None:
                raise RuntimeError("Reaction point lookup is not initialized.")
            for row, pt in enumerate(rounded):
                idx = self.lookup.get(tuple(pt.tolist()), -1)
                if idx >= 0:
                    matches[idx].append(row)
            rows = np.full(len(self.points), -1, dtype=np.int64)
            for idx, row_list in enumerate(matches):
                if len(row_list) > self.occurrence:
                    rows[idx] = row_list[self.occurrence]
                elif row_list:
                    rows[idx] = row_list[-1]
            if np.any(rows < 0):
                missing = int(np.sum(rows < 0))
                counts = sorted({len(row_list) for row_list in matches})
                raise RuntimeError(
                    "Reaction integral missing "
                    f"{missing} rows for occurrence={self.occurrence}; "
                    f"available duplicate counts={counts}."
                )
            self._index_cache[key] = rows
            return rows

        top_rows = np.where(np.isclose(x_np[:, 1], self.boundary_y, atol=1.0e-7))[0]
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

    def _row_index_tensor(self, X, device: torch.device) -> torch.Tensor:
        key = (id(X), len(X), self.occurrence, device)
        cached = self._index_tensor_cache.get(key)
        if cached is not None:
            return cached
        row_index = torch.as_tensor(self._row_index(X), dtype=torch.long, device=device)
        self._index_tensor_cache[key] = row_index
        return row_index

    def __call__(self, inputs, outputs, X):
        weights = self._weights(outputs.device, outputs.dtype)
        row_index = self._row_index_tensor(X, outputs.device)
        density = outputs[row_index, self.stress_index : self.stress_index + 1]
        resultant = torch.sum(density * weights) / self.scale
        if self.fill_all:
            return torch.ones((outputs.shape[0], 1), dtype=outputs.dtype, device=outputs.device) * resultant
        result = torch.zeros((outputs.shape[0], 1), dtype=outputs.dtype, device=outputs.device)
        result[row_index, 0:1] = resultant
        return result
