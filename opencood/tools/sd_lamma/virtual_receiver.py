# -*- coding: utf-8 -*-
"""Virtual receiver priors for broadcast SD-LAMMA."""

from typing import Tuple

import torch
import torch.nn as nn


def _direction_table(num_tokens: int, radius: float) -> torch.Tensor:
    """Return [dx, dy, dist, prior] tokens around the sender."""
    base = [
        (0.0, 1.0, radius, 1.00),    # front
        (-0.7071, 0.7071, radius, 0.92),
        (0.7071, 0.7071, radius, 0.92),
        (-1.0, 0.0, radius, 0.82),
        (1.0, 0.0, radius, 0.82),
        (0.0, -1.0, radius, 0.68),
        (-0.7071, -0.7071, radius, 0.72),
        (0.7071, -0.7071, radius, 0.72),
    ]
    if num_tokens <= len(base):
        return torch.tensor(base[:num_tokens], dtype=torch.float32)

    tokens = list(base)
    extra = num_tokens - len(base)
    for idx in range(extra):
        angle = 2.0 * torch.pi * torch.tensor(float(idx) / max(extra, 1))
        dx = float(torch.sin(angle).item())
        dy = float(torch.cos(angle).item())
        tokens.append((dx, dy, radius, 0.75))
    return torch.tensor(tokens, dtype=torch.float32)


class VirtualReceiverAttention(nn.Module):
    """
    Lightweight receiver-agnostic demand estimator.

    Each BEV cell builds a small query from its normalized location and sender
    feature energy. Fixed or learnable virtual receiver tokens provide keys and
    values, producing a broadcast demand map without reading the ego demand map.
    """

    def __init__(
        self,
        num_virtual_receivers: int = 8,
        mode: str = "fixed",
        radius: float = 1.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_virtual_receivers = max(1, int(num_virtual_receivers))
        self.mode = str(mode or "fixed").lower()
        self.temperature = max(float(temperature), 1.0e-3)

        tokens = _direction_table(self.num_virtual_receivers, float(radius))
        self.register_buffer("base_tokens", tokens, persistent=False)
        if self.mode == "learnable":
            self.token_delta = nn.Parameter(torch.zeros_like(tokens))
        else:
            self.register_parameter("token_delta", None)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 4:
            raise ValueError("VirtualReceiverAttention expects [N, C, H, W].")

        energy = self._feature_energy(features)
        grid_x, grid_y = self._normalized_grid(
            features.shape[-2], features.shape[-1], features.device, features.dtype
        )
        query = torch.cat(
            [
                grid_x.expand(features.shape[0], -1, -1, -1),
                grid_y.expand(features.shape[0], -1, -1, -1),
                energy,
                torch.ones_like(energy),
            ],
            dim=1,
        )

        tokens = self._tokens(device=features.device, dtype=features.dtype)
        keys = tokens[:, :4]
        values = tokens[:, 3].clamp(0.0, 1.0)
        logits = torch.einsum("nchw,kc->nkhw", query, keys) / self.temperature
        attn = torch.softmax(logits, dim=1)
        spatial_demand = (attn * values.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
        demand = 0.55 * spatial_demand + 0.45 * energy
        return self._normalize_map(demand).clamp(0.0, 1.0)

    def soft_or_demand(self, features: torch.Tensor) -> torch.Tensor:
        """Parameter-free fallback used before VRA is trained or when disabled."""
        energy = self._feature_energy(features)
        grid_x, grid_y = self._normalized_grid(
            features.shape[-2], features.shape[-1], features.device, features.dtype
        )
        tokens = self.base_tokens.to(device=features.device, dtype=features.dtype)
        dx = tokens[:, 0].view(1, -1, 1, 1)
        dy = tokens[:, 1].view(1, -1, 1, 1)
        prior = tokens[:, 3].view(1, -1, 1, 1).clamp(0.0, 1.0)
        alignment = ((grid_x * dx + grid_y * dy) + 1.0) * 0.5
        per_token = (alignment.clamp(0.0, 1.0) * prior).clamp(0.0, 1.0)
        soft_or = 1.0 - torch.prod(1.0 - per_token, dim=1, keepdim=True)
        demand = 0.50 * soft_or.expand_as(energy) + 0.50 * energy
        return self._normalize_map(demand).clamp(0.0, 1.0)

    def _tokens(self, device, dtype) -> torch.Tensor:
        tokens = self.base_tokens.to(device=device, dtype=dtype)
        if self.token_delta is not None:
            delta = torch.tanh(self.token_delta.to(device=device, dtype=dtype)) * 0.25
            tokens = tokens + delta
        direction = tokens[:, :2]
        direction = direction / torch.clamp(direction.norm(dim=1, keepdim=True), min=1.0e-6)
        dist_prior = tokens[:, 2:].clone()
        dist_prior[:, 0].clamp_(0.0, 2.0)
        dist_prior[:, 1].clamp_(0.0, 1.0)
        return torch.cat([direction, dist_prior], dim=1)

    @staticmethod
    def _feature_energy(features: torch.Tensor) -> torch.Tensor:
        energy = features.float().abs().mean(dim=1, keepdim=True)
        return VirtualReceiverAttention._normalize_map(energy).to(
            device=features.device, dtype=features.dtype
        )

    @staticmethod
    def _normalized_grid(
        height: int,
        width: int,
        device,
        dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        return grid_x.view(1, 1, height, width), grid_y.view(1, 1, height, width)

    @staticmethod
    def _normalize_map(x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(start_dim=2)
        min_v = flat.min(dim=-1).values.view(x.shape[0], x.shape[1], 1, 1)
        max_v = flat.max(dim=-1).values.view(x.shape[0], x.shape[1], 1, 1)
        return torch.where(
            max_v > min_v,
            (x - min_v) / torch.clamp(max_v - min_v, min=1.0e-6),
            x.clamp(0.0, 1.0),
        )
