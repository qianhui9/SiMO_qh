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
        learnable_alpha: float = 0.1,
        train_temperature: bool = True,
        train_prior_scale: bool = True,
        prior_scale_alpha: float = 0.25,
    ):
        super().__init__()
        self.num_virtual_receivers = max(1, int(num_virtual_receivers))
        self.mode = str(mode or "fixed").lower()
        self.temperature = max(float(temperature), 1.0e-3)
        self.learnable_alpha = max(float(learnable_alpha), 0.0)
        self.prior_scale_alpha = max(float(prior_scale_alpha), 0.0)

        tokens = _direction_table(self.num_virtual_receivers, float(radius))
        self.register_buffer("base_tokens", tokens, persistent=False)
        if self.mode == "learnable":
            self.token_delta = nn.Parameter(torch.zeros_like(tokens))
            if bool(train_temperature):
                self.log_temperature_delta = nn.Parameter(torch.zeros(1))
            else:
                self.register_parameter("log_temperature_delta", None)
            if bool(train_prior_scale):
                self.prior_scale_delta = nn.Parameter(torch.zeros(1))
            else:
                self.register_parameter("prior_scale_delta", None)
        else:
            self.register_parameter("token_delta", None)
            self.register_parameter("log_temperature_delta", None)
            self.register_parameter("prior_scale_delta", None)

    def forward(
        self,
        features: torch.Tensor,
        token_dropout: float = 0.0,
        token_noise_std: float = 0.0,
    ) -> torch.Tensor:
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

        tokens = self._tokens(
            device=features.device,
            dtype=features.dtype,
            token_dropout=token_dropout,
            token_noise_std=token_noise_std,
        )
        keys = tokens[:, :4]
        values = self._scaled_prior(tokens[:, 3]).clamp(0.0, 1.0)
        logits = torch.einsum("nchw,kc->nkhw", query, keys) / self._temperature(
            features.device, features.dtype
        )
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

    def _tokens(
        self,
        device,
        dtype,
        token_dropout: float = 0.0,
        token_noise_std: float = 0.0,
    ) -> torch.Tensor:
        tokens = self.base_tokens.to(device=device, dtype=dtype)
        if self.token_delta is not None:
            delta = torch.tanh(self.token_delta.to(device=device, dtype=dtype)) * self.learnable_alpha
            tokens = tokens + delta
        if self.training and float(token_noise_std or 0.0) > 0.0:
            tokens = tokens + torch.randn_like(tokens) * float(token_noise_std)
        if self.training and float(token_dropout or 0.0) > 0.0 and tokens.shape[0] > 1:
            keep_prob = 1.0 - max(0.0, min(1.0, float(token_dropout)))
            keep = (torch.rand((tokens.shape[0], 1), device=device, dtype=dtype) < keep_prob).to(dtype)
            if bool((keep.sum() <= 0).item()):
                keep[0].fill_(1.0)
            tokens = tokens.clone()
            tokens[:, 3:4] = tokens[:, 3:4] * keep
        direction = tokens[:, :2]
        direction = direction / torch.clamp(direction.norm(dim=1, keepdim=True), min=1.0e-6)
        dist_prior = tokens[:, 2:].clone()
        dist_prior[:, 0].clamp_(0.0, 2.0)
        dist_prior[:, 1].clamp_(0.0, 1.0)
        return torch.cat([direction, dist_prior], dim=1)

    def _temperature(self, device, dtype) -> torch.Tensor:
        value = torch.tensor(self.temperature, device=device, dtype=dtype)
        if self.log_temperature_delta is not None:
            delta = self.log_temperature_delta.to(device=device, dtype=dtype)
            value = value * torch.exp(0.25 * torch.tanh(delta))
        return torch.clamp(value, min=1.0e-3)

    def _scaled_prior(self, prior: torch.Tensor) -> torch.Tensor:
        if self.prior_scale_delta is None:
            return prior
        scale = 1.0 + self.prior_scale_alpha * torch.tanh(
            self.prior_scale_delta.to(device=prior.device, dtype=prior.dtype)
        )
        return prior * scale

    def learnable_stats(self) -> dict:
        stats = {
            "virtual_receiver_mode": self.mode,
            "num_virtual_receivers": int(self.num_virtual_receivers),
            "learnable_alpha": float(self.learnable_alpha),
        }
        if self.token_delta is not None:
            delta = self.token_delta.detach().float()
            stats["learnable_delta_norm"] = float(delta.norm().item())
            stats["learnable_delta_max_abs"] = float(delta.abs().max().item())
        else:
            stats["learnable_delta_norm"] = 0.0
            stats["learnable_delta_max_abs"] = 0.0
        if self.log_temperature_delta is not None:
            stats["learnable_temperature"] = float(
                self._temperature(
                    self.log_temperature_delta.device,
                    self.log_temperature_delta.dtype,
                ).detach().item()
            )
        else:
            stats["learnable_temperature"] = float(self.temperature)
        if self.prior_scale_delta is not None:
            scale = 1.0 + self.prior_scale_alpha * torch.tanh(
                self.prior_scale_delta.detach().float()
            )
            stats["learnable_prior_scale"] = float(scale.item())
        else:
            stats["learnable_prior_scale"] = 1.0
        return stats

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
