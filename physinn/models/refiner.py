"""Refinement module built on top of the EfficientNet encoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from .efficientnet import EfficientNetEncoder


class EfficientNetRefiner(nn.Module):
    def __init__(
        self,
        m_params: int,
        cond_dim: int,
        backbone_feat_dim: int,
        *,
        delta_scale: float = 0.1,
        max_refine_steps: int = 3,
        encoder_variant: str = "s",
        encoder_width_mult: float = 1.0,
        encoder_depth_mult: float = 1.0,
        encoder_stem_channels: int | None = None,
        encoder_drop_path: float = 0.1,
        encoder_se_ratio: float = 0.25,
        feature_pool: str = "avg",
        shared_hidden_scale: float = 0.5,
        time_embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.delta_scale = float(delta_scale)
        self.m_params = int(m_params)
        self.cond_dim = int(cond_dim)
        self.use_film = True

        self.encoder = EfficientNetEncoder(
            in_channels=2,
            variant=encoder_variant,
            width_mult=encoder_width_mult,
            depth_mult=encoder_depth_mult,
            se_ratio=encoder_se_ratio,
            drop_path_rate=encoder_drop_path,
            stem_channels=encoder_stem_channels,
        )

        feat_dim = self.encoder.feat_dim
        pool_mode = feature_pool.lower()
        if pool_mode == "avg":
            self.feature_head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        elif pool_mode == "max":
            self.feature_head = nn.Sequential(nn.AdaptiveMaxPool1d(1), nn.Flatten())
        elif pool_mode == "avgmax":
            self.feature_head = nn.ModuleList([nn.AdaptiveAvgPool1d(1), nn.AdaptiveMaxPool1d(1)])
        else:
            raise ValueError(f"Unknown feature_pool: {feature_pool}")

        hidden = max(64, int(round(feat_dim * float(shared_hidden_scale))))
        in_shared = feat_dim if pool_mode != "avgmax" else 2 * feat_dim

        self.shared_head = nn.Sequential(
            nn.Linear(in_shared, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

        self.time_embed_dim = int(time_embed_dim) if time_embed_dim is not None else max(16, hidden // 4)
        self.time_embed = nn.Embedding(max_refine_steps, self.time_embed_dim)

        film_in = self.cond_dim + self.time_embed_dim
        self.film_time = nn.Sequential(nn.Linear(film_in, hidden), nn.Tanh(), nn.Linear(hidden, 2 * hidden))

        self.scale_gate = nn.Linear(hidden, m_params)
        self.delta_head = nn.Sequential(
            nn.Linear(hidden + backbone_feat_dim + m_params, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, m_params),
        )

        self._feature_pool_mode = pool_mode
        self._hidden = hidden

    def _pool_features(self, latent: torch.Tensor) -> torch.Tensor:
        if self._feature_pool_mode == "avg":
            return self.feature_head(latent)
        if self._feature_pool_mode == "max":
            return self.feature_head(latent)
        avg_pool, max_pool = self.feature_head
        return torch.cat([avg_pool(latent).flatten(1), max_pool(latent).flatten(1)], dim=1)

    def forward(
        self,
        noisy: torch.Tensor,
        resid: torch.Tensor,
        params_pred_norm: torch.Tensor,
        cond_norm: torch.Tensor | None,
        feat_shared: torch.Tensor,
        t_step: int = 0,
    ) -> torch.Tensor:
        latent, _ = self.encoder(torch.stack([noisy, resid], dim=1))
        feat = self._pool_features(latent)
        hidden = self.shared_head(feat)

        batch_size = hidden.size(0)
        t_ids = torch.full((batch_size,), int(t_step), device=hidden.device, dtype=torch.long)
        tvec = self.time_embed(t_ids)
        cond_in = tvec if cond_norm is None else torch.cat([cond_norm, tvec], dim=1)
        gamma_beta = self.film_time(cond_in)
        gamma, beta = gamma_beta[:, : hidden.shape[1]], gamma_beta[:, hidden.shape[1] :]
        hidden = hidden * (1 + 0.1 * gamma) + 0.1 * beta

        gate = torch.sigmoid(self.scale_gate(hidden))
        scale = self.delta_scale * gate

        z = torch.cat([hidden, feat_shared, params_pred_norm], dim=1)
        raw = self.delta_head(z)
        delta = torch.tanh(raw) * scale
        return delta

