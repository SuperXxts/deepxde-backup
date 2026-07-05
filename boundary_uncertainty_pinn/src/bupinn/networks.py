"""DeepXDE-compatible networks for boundary-uncertainty inversion."""

from __future__ import annotations

import torch


class BoundaryMaterialPFNN(torch.nn.Module):
    """State/material network with an optional low-dimensional boundary layer.

    Output columns are:
    [ux, uy, sxx, syy, sxy, zK, zmu, ux_core, uy_core, ux_boundary, uy_boundary]

    The first seven columns are used by the PDE and boundary conditions. The
    remaining columns are diagnostics and make the split explicit in saved data.
    """

    def __init__(
        self,
        dde,
        state_width: int = 64,
        state_depth: int = 4,
        material_width: int = 64,
        material_depth: int = 4,
        activation: str = "tanh",
        initializer: str = "Glorot normal",
        num_boundary_modes: int = 0,
        init_beta: list[float] | None = None,
        trainable_boundary: bool = True,
        use_boundary_layer: bool = False,
        boundary_mode_type: str = "unit_top",
        domain_x_min: float = 0.0,
        domain_x_max: float = 1.0,
        domain_y_bottom: float = 0.0,
        domain_y_top: float = 1.0,
        plate_center: float | None = None,
        plate_width: float | None = None,
        boundary_mode_scale: float = 1.0,
    ):
        super().__init__()
        self._input_transform = None
        self._output_transform = None
        self.regularizer = None
        self.num_boundary_modes = int(num_boundary_modes)
        self.use_boundary_layer = bool(use_boundary_layer and self.num_boundary_modes > 0)
        self.boundary_mode_type = str(boundary_mode_type)
        self.domain_x_min = float(domain_x_min)
        self.domain_x_max = float(domain_x_max)
        self.domain_y_bottom = float(domain_y_bottom)
        self.domain_y_top = float(domain_y_top)
        self.plate_center = None if plate_center is None else float(plate_center)
        self.plate_width = None if plate_width is None else float(plate_width)
        self.boundary_mode_scale = float(boundary_mode_scale)

        state_layers = [2] + [[int(state_width)] * 5 for _ in range(int(state_depth))] + [5]
        self.state_net = dde.nn.PFNN(state_layers, activation, initializer)
        material_layers = [2] + [int(material_width)] * int(material_depth) + [2]
        self.material_net = dde.nn.FNN(material_layers, activation, initializer)

        if self.num_boundary_modes > 0:
            if init_beta is None:
                init_beta = [0.85] + [0.0] * (self.num_boundary_modes - 1)
            if len(init_beta) != self.num_boundary_modes:
                raise ValueError("init_beta length must equal num_boundary_modes.")
            beta = torch.tensor(init_beta, dtype=torch.float32)
            self.beta = torch.nn.Parameter(beta, requires_grad=bool(trainable_boundary))
        else:
            self.register_buffer("beta", torch.empty(0, dtype=torch.float32))

    def apply_feature_transform(self, transform):
        self._input_transform = transform

    def apply_output_transform(self, transform):
        self._output_transform = transform

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def top_modes_torch(self, inputs):
        if self.num_boundary_modes <= 0:
            return inputs[:, 0:0]
        xp_raw = inputs[:, 0:1]
        if self.boundary_mode_type == "plate":
            if self.plate_center is None or self.plate_width is None:
                raise ValueError("plate boundary modes require plate_center and plate_width.")
            local = (xp_raw - self.plate_center) / (0.5 * self.plate_width)
            modes = [torch.ones_like(local) * self.boundary_mode_scale]
            if self.num_boundary_modes >= 2:
                modes.append(self.boundary_mode_scale * local)
            if self.num_boundary_modes >= 3:
                modes.append(self.boundary_mode_scale * (2.0 * local**2 - 1.0))
            if self.num_boundary_modes >= 4:
                modes.append(self.boundary_mode_scale * torch.sin(torch.pi * (local + 1.0) / 2.0))
            if self.num_boundary_modes >= 5:
                modes.append(self.boundary_mode_scale * torch.sin(torch.pi * local))
            if self.num_boundary_modes > 5:
                raise ValueError(f"Unsupported num_boundary_modes={self.num_boundary_modes}")
            return torch.cat(modes[: self.num_boundary_modes], dim=1)

        x_span = max(self.domain_x_max - self.domain_x_min, 1.0e-12)
        xp = (xp_raw - self.domain_x_min) / x_span
        modes = [
            0.15
            + 0.04 * torch.sin(torch.pi * xp)
            + 0.02 * torch.sin(2.0 * torch.pi * xp)
        ]
        if self.num_boundary_modes >= 2:
            modes.append(0.10 * (2.0 * xp - 1.0))
        if self.num_boundary_modes >= 3:
            modes.append(0.10 * torch.sin(torch.pi * xp))
        if self.num_boundary_modes >= 4:
            modes.append(0.10 * torch.sin(2.0 * torch.pi * xp))
        if self.num_boundary_modes >= 5:
            modes.append(0.10 * torch.cos(torch.pi * xp))
        if self.num_boundary_modes > 5:
            raise ValueError(f"Unsupported num_boundary_modes={self.num_boundary_modes}")
        return torch.cat(modes[: self.num_boundary_modes], dim=1)

    def boundary_displacement(self, inputs):
        if not self.use_boundary_layer:
            zeros = torch.zeros((inputs.shape[0], 1), dtype=inputs.dtype, device=inputs.device)
            return zeros, zeros
        beta = self.beta.to(dtype=inputs.dtype, device=inputs.device).view(-1, 1)
        top_value = self.top_modes_torch(inputs) @ beta
        ub = torch.zeros_like(top_value)
        y_span = max(self.domain_y_top - self.domain_y_bottom, 1.0e-12)
        lift = (inputs[:, 1:2] - self.domain_y_bottom) / y_span
        vb = lift * top_value
        return ub, vb

    def top_boundary_value(self, inputs):
        if self.num_boundary_modes <= 0:
            return torch.zeros((inputs.shape[0], 1), dtype=inputs.dtype, device=inputs.device)
        beta = self.beta.to(dtype=inputs.dtype, device=inputs.device).view(-1, 1)
        return self.top_modes_torch(inputs) @ beta

    def set_group_trainable(self, state: bool = True, material: bool = True, boundary: bool = True):
        for p in self.state_net.parameters():
            p.requires_grad_(bool(state))
        for p in self.material_net.parameters():
            p.requires_grad_(bool(material))
        if isinstance(self.beta, torch.nn.Parameter):
            self.beta.requires_grad_(bool(boundary))

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(inputs)
        state = self.state_net(x)
        material = self.material_net(x)
        ux_core = state[:, 0:1]
        uy_core = state[:, 1:2]
        ux_b, uy_b = self.boundary_displacement(x)
        outputs = torch.cat(
            [
                ux_core + ux_b,
                uy_core + uy_b,
                state[:, 2:5],
                material[:, 0:1],
                material[:, 1:2],
                ux_core,
                uy_core,
                ux_b,
                uy_b,
            ],
            dim=1,
        )
        if self._output_transform is not None:
            outputs = self._output_transform(inputs, outputs)
        return outputs
