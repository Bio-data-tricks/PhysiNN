"""Lightning module implementing the Physically Informed Auto-Encoder."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from ..config import PARAMS
from ..lowess import lowess_value
from ..losses import ReLoBRaLoLoss
from ..normalization import norm_param_torch, unnorm_param_torch
from ..physics.spectra import batch_physics_forward_multimol_vgrid
from .efficientnet import EfficientNetEncoder
from .refiner import EfficientNetRefiner


class PhysicallyInformedAE(pl.LightningModule):
    def __init__(
        self,
        n_points: int,
        param_names: List[str],
        poly_freq_CH4,
        transitions_dict,
        ranges_train: dict | None = None,
        ranges_val: dict | None = None,
        noise_train: dict | None = None,
        noise_val: dict | None = None,
        lr: float = 1e-4,
        alpha_param: float = 1.0,
        alpha_phys: float = 1.0,
        head_mode: str = "multi",
        predict_params: Optional[List[str]] = None,
        film_params: Optional[List[str]] = None,
        refine_steps: int = 1,
        refine_delta_scale: float = 0.1,
        refine_target: str = "noisy",
        refine_warmup_epochs: int = 30,
        freeze_base_epochs: int = 20,
        stage3_lr_shrink: float = 0.5,
        stage3_refine_steps: Optional[int] = None,
        stage3_delta_scale: Optional[float] = None,
        stage3_alpha_phys: Optional[float] = None,
        stage3_alpha_param: Optional[float] = None,
        recon_max1: bool = True,
        corr_mode: str = "none",
        corr_savgol_win: int = 15,
        corr_savgol_poly: int = 3,
        huber_beta: float = 0.002,
        weight_mf: float = 1.0,
        backbone_variant: str = "s",
        backbone_width_mult: float = 1.0,
        backbone_depth_mult: float = 1.0,
        backbone_stem_channels: Optional[int] = None,
        backbone_drop_path: float = 0.0,
        backbone_se_ratio: float = 0.25,
        refiner_variant: str = "s",
        refiner_width_mult: float = 1.0,
        refiner_depth_mult: float = 1.0,
        refiner_stem_channels: Optional[int] = None,
        refiner_drop_path: float = 0.0,
        refiner_se_ratio: float = 0.25,
        refiner_feature_pool: str = "avg",
        refiner_shared_hidden_scale: float = 0.5,
        refiner_time_embed_dim: Optional[int] = None,
        tipspy=None,
    ) -> None:
        super().__init__()
        self.n_points = n_points
        self.param_names = list(param_names)
        self.poly_freq_CH4 = poly_freq_CH4
        self.transitions_dict = transitions_dict
        self.ranges_train = ranges_train or {}
        self.ranges_val = ranges_val or {}
        self.noise_train = noise_train or {}
        self.noise_val = noise_val or {}
        self.tipspy = tipspy
        self.lr = lr
        self.alpha_param = alpha_param
        self.alpha_phys = alpha_phys
        self.head_mode = str(head_mode).lower()
        assert self.head_mode in {"single", "multi"}

        self.refine_steps = refine_steps
        self.refine_delta_scale = refine_delta_scale
        self.refine_target = refine_target
        self.refine_warmup_epochs = refine_warmup_epochs
        self.freeze_base_epochs = freeze_base_epochs
        self.stage3_lr_shrink = stage3_lr_shrink
        self.stage3_refine_steps = stage3_refine_steps
        self.stage3_delta_scale = stage3_delta_scale
        self.stage3_alpha_phys = stage3_alpha_phys
        self.stage3_alpha_param = stage3_alpha_param
        self.recon_max1 = recon_max1
        self.corr_mode = corr_mode.lower()
        self.corr_savgol_win = corr_savgol_win
        self.corr_savgol_poly = corr_savgol_poly
        self.huber_beta = huber_beta
        self.weight_mf = weight_mf

        if predict_params is None:
            predict_params = self.param_names
        self.predict_params = list(predict_params)
        self.provided_params = [p for p in self.param_names if p not in self.predict_params]
        unknown = set(self.predict_params) - set(self.param_names)
        if unknown:
            raise ValueError(f"Unknown parameters: {unknown}")
        if film_params is None:
            self.film_params = list(self.provided_params)
        else:
            self.film_params = list(film_params)
        invalid_film = set(self.film_params) - set(self.param_names)
        if invalid_film:
            raise ValueError(f"film_params contain unknown entries: {invalid_film}")
        not_provided = set(self.film_params) - set(self.provided_params)
        if not_provided:
            raise ValueError(f"film_params must be subset of provided params: {not_provided}")

        self.name_to_idx = {n: i for i, n in enumerate(self.param_names)}
        self.predict_idx = [self.name_to_idx[p] for p in self.predict_params]
        self.provided_idx = [self.name_to_idx[p] for p in self.provided_params]

        self.backbone = EfficientNetEncoder(
            in_channels=1,
            variant=backbone_variant,
            width_mult=backbone_width_mult,
            depth_mult=backbone_depth_mult,
            se_ratio=backbone_se_ratio,
            drop_path_rate=backbone_drop_path,
            stem_channels=backbone_stem_channels,
        )
        self.feature_head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        feat_dim = self.backbone.feat_dim
        hidden = feat_dim // 2
        self.shared_head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.cond_dim = len(self.film_params)
        if self.cond_dim > 0:
            self.film = nn.Sequential(nn.Linear(self.cond_dim, hidden), nn.Tanh(), nn.Linear(hidden, 2 * hidden))
            self.register_buffer("film_mask", torch.ones(self.cond_dim))
        else:
            self.film = None
            self.film_mask = None
        if self.head_mode == "single":
            self.out_head = nn.Linear(hidden, len(self.predict_params))
        else:
            self.out_heads = nn.ModuleDict({p: nn.Linear(hidden, 1) for p in self.predict_params})

        self.refiner = EfficientNetRefiner(
            m_params=len(self.predict_params),
            cond_dim=self.cond_dim,
            backbone_feat_dim=self.backbone.feat_dim,
            delta_scale=refine_delta_scale,
            max_refine_steps=max(3, self.refine_steps if isinstance(self.refine_steps, int) else 3),
            encoder_variant=refiner_variant,
            encoder_width_mult=refiner_width_mult,
            encoder_depth_mult=refiner_depth_mult,
            encoder_stem_channels=refiner_stem_channels,
            encoder_drop_path=refiner_drop_path,
            encoder_se_ratio=refiner_se_ratio,
            feature_pool=refiner_feature_pool,
            shared_hidden_scale=refiner_shared_hidden_scale,
            time_embed_dim=refiner_time_embed_dim,
        )

        self.loss_names_params = [f"param_{p}" for p in self.predict_params]
        self.relo_params = ReLoBRaLoLoss(self.loss_names_params, alpha=0.9, tau=1.0, history_len=10)
        self.loss_names_top = ["phys_mse", "phys_corr", "param_group"]
        self.relo_top = ReLoBRaLoLoss(self.loss_names_top, alpha=0.9, tau=1.0, history_len=10)

        self._override_stage: Optional[str] = None
        self._override_refine_steps: Optional[int] = None
        self._override_delta_scale: Optional[float] = None
        self._froze_base = False

    # FiLM controls
    def set_film_usage(self, use: bool = True) -> None:
        self.use_film = bool(use)
        if hasattr(self, "refiner") and hasattr(self.refiner, "use_film"):
            self.refiner.use_film = self.use_film

    def set_film_subset(self, names=None) -> None:
        if self.cond_dim == 0 or self.film_mask is None:
            return
        if names is None or names == "all":
            mask = torch.ones(self.cond_dim, device=self.film_mask.device, dtype=self.film_mask.dtype)
        else:
            allowed = set(names)
            mask = torch.zeros(self.cond_dim, device=self.film_mask.device, dtype=self.film_mask.dtype)
            for i, name in enumerate(self.film_params):
                if name in allowed:
                    mask[i] = 1.0
        self.film_mask.copy_(mask)

    # Stage override
    def set_stage_mode(self, mode: Optional[str], refine_steps: Optional[int] = None, delta_scale: Optional[float] = None):
        if mode is not None:
            mode = mode.upper()
            assert mode in {"A", "B1", "B2"}
        self._override_stage = mode
        self._override_refine_steps = refine_steps
        self._override_delta_scale = delta_scale
        if delta_scale is not None:
            self.refiner.delta_scale = float(delta_scale)
        if refine_steps is not None:
            self.refine_steps = int(refine_steps)

    def _predict_params_from_features(self, feat: torch.Tensor, cond_norm: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden = self.shared_head(feat)
        if self.film is not None and cond_norm is not None and getattr(self, "use_film", True):
            if self.film_mask is not None and self.film_mask.numel() == cond_norm.shape[1]:
                cond_input = cond_norm * self.film_mask.unsqueeze(0)
            else:
                cond_input = cond_norm
            gamma_beta = self.film(cond_input)
            H = hidden.shape[1]
            gamma, beta = gamma_beta[:, :H], gamma_beta[:, H:]
            hidden = hidden * (1 + 0.1 * gamma) + 0.1 * beta
        if self.head_mode == "single":
            logits = self.out_head(hidden)
        else:
            logits = torch.cat([self.out_heads[p](hidden) for p in self.predict_params], dim=1)
        return torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)

    def encode(self, spectra: torch.Tensor, pooled: bool = True, detach: bool = False):
        latent, _ = self.backbone(spectra.unsqueeze(1))
        feat = self.feature_head(latent) if pooled else latent
        return feat.detach() if detach else feat

    def _denorm_params_subset(self, y_norm_subset: torch.Tensor, names: List[str]) -> torch.Tensor:
        cols = [unnorm_param_torch(name, y_norm_subset[:, i]) for i, name in enumerate(names)]
        return torch.stack(cols, dim=1)

    def _compose_full_phys(self, pred_phys: torch.Tensor, provided_phys_tensor: torch.Tensor) -> torch.Tensor:
        batch = pred_phys.shape[0]
        full = pred_phys.new_empty((batch, self.n_params))
        for j, idx in enumerate(self.predict_idx):
            full[:, idx] = pred_phys[:, j]
        for j, idx in enumerate(self.provided_idx):
            full[:, idx] = provided_phys_tensor[:, j]
        return full

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    def _physics_reconstruction(self, y_phys_full: torch.Tensor, device, scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        params = {name: y_phys_full[:, i] for i, name in enumerate(self.param_names)}
        v_grid_idx = torch.arange(self.n_points, dtype=torch.float64, device=device)
        baseline_idx = self.name_to_idx["baseline0"]
        baseline_coeffs = y_phys_full[:, baseline_idx : baseline_idx + 3]

        mf_dict = {}
        for mol in self.transitions_dict.keys():
            key = f"mf_{mol}"
            mf_dict[mol] = params[key] if key in params else torch.zeros_like(params["P"])

        spectra, _ = batch_physics_forward_multimol_vgrid(
            params["sig0"],
            params["dsig"],
            self.poly_freq_CH4,
            v_grid_idx,
            baseline_coeffs,
            self.transitions_dict,
            params["P"],
            params["T"],
            mf_dict,
            tipspy=self.tipspy,
            device=device,
        )
        spectra = spectra.to(torch.float32)
        scale_recon = lowess_value(spectra, kind="start", window=30).unsqueeze(1).clamp_min(1e-8)
        spectra = spectra / scale_recon
        return spectra

    def _make_condition_from_norm(self, params_true_norm: torch.Tensor) -> Optional[torch.Tensor]:
        if self.cond_dim == 0:
            return None
        cols = [params_true_norm[:, self.name_to_idx[name]].unsqueeze(1) for name in self.film_params]
        return torch.cat(cols, dim=1)

    def _make_condition_from_phys(self, provided_phys: dict, device, dtype=torch.float32) -> Optional[torch.Tensor]:
        if self.cond_dim == 0:
            return None
        missing = [name for name in self.film_params if name not in provided_phys]
        if missing:
            raise ValueError(f"FiLM missing keys: {missing}")
        cols = []
        for name in self.film_params:
            value = provided_phys[name].to(device)
            if value.ndim > 1:
                value = value.view(-1)
            cols.append(norm_param_torch(name, value).unsqueeze(1))
        return torch.cat(cols, dim=1).to(dtype)

    def _savgol_coeffs(self, window_length: int, polyorder: int, deriv: int = 1, delta: float = 1.0, device=None, dtype=torch.float64) -> torch.Tensor:
        assert deriv >= 0
        W = int(window_length)
        P = int(polyorder)
        if W % 2 == 0:
            W += 1
        if W < 3:
            W = 3
        if P >= W:
            P = W - 1
        m = (W - 1) // 2
        dev = device or self.device
        x = torch.arange(-m, m + 1, device=dev, dtype=dtype)
        A = torch.stack([x ** j for j in range(P + 1)], dim=1)
        pinv = torch.linalg.pinv(A)
        coeff = math.factorial(deriv) * pinv[deriv, :] / (delta ** deriv)
        return coeff.to(dtype=torch.float32)

    def _savgol_deriv(self, y: torch.Tensor, window_length: int, polyorder: int, deriv: int = 1) -> torch.Tensor:
        batch, length = y.shape
        W = int(window_length)
        if W % 2 == 0:
            W += 1
        if W > length:
            W = length if (length % 2 == 1) else (length - 1)
        W = max(W, 3)
        P = min(int(polyorder), W - 1)
        coeff = self._savgol_coeffs(W, P, deriv=deriv, device=y.device, dtype=torch.float64).view(1, 1, -1)
        pad = (W - 1) // 2
        y1 = F.pad(y.unsqueeze(1), (pad, pad), mode="reflect")
        out = F.conv1d(y1, coeff).squeeze(1)
        return out

    def _dx(self, y: torch.Tensor) -> torch.Tensor:
        diff = 0.5 * (y[:, 2:] - y[:, :-2])
        left = diff[:, :1]
        right = diff[:, -1:]
        return torch.cat([left, diff, right], dim=1)

    def _pearson_corr_loss(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        eps: float = 1e-8,
        derivative: str | bool = False,
        savgol_win: int = 11,
        savgol_poly: int = 3,
    ) -> torch.Tensor:
        mode = derivative
        if isinstance(derivative, bool):
            mode = "central" if derivative else "none"
        mode = (mode or "none").lower()
        if mode == "central":
            y_hat = self._dx(y_hat)
            y = self._dx(y)
        elif mode == "savgol":
            y_hat = self._savgol_deriv(y_hat, savgol_win, savgol_poly)
            y = self._savgol_deriv(y, savgol_win, savgol_poly)
        y_hat = y_hat - y_hat.mean(dim=1, keepdim=True)
        y = y - y.mean(dim=1, keepdim=True)
        num = torch.sum(y_hat * y, dim=1)
        den = torch.sqrt(torch.sum(y_hat ** 2, dim=1) * torch.sum(y ** 2, dim=1) + eps)
        corr = num / den
        return 1 - corr.mean()

    def _common_step(self, batch, step_name: str):
        noisy = batch["noisy_spectra"].to(self.device)
        clean = batch["clean_spectra"].to(self.device)
        params_true_norm = batch["params"].to(self.device)
        scale = batch.get("scale", None)
        cond_norm = self._make_condition_from_norm(params_true_norm)

        latent, _ = self.backbone(noisy.unsqueeze(1))
        feat_shared = self.feature_head(latent)
        params_pred_norm = self._predict_params_from_features(feat_shared, cond_norm=cond_norm)

        if len(self.provided_params) > 0:
            provided_list = [params_true_norm[:, self.name_to_idx[name]] for name in self.provided_params]
            provided_phys_tensor = torch.stack(provided_list, dim=1)
        else:
            provided_phys_tensor = params_pred_norm.new_zeros((noisy.size(0), 0))

        pred_phys = self._denorm_params_subset(params_pred_norm, self.predict_params)
        y_full = self._compose_full_phys(pred_phys, provided_phys_tensor)
        spectra_recon = self._physics_reconstruction(y_full, self.device, scale=None)

        if self.recon_max1:
            spectra_recon = spectra_recon / spectra_recon.max(dim=1, keepdim=True)[0].clamp_min(1e-8)

        loss_phys_huber = F.smooth_l1_loss(spectra_recon, clean, beta=self.huber_beta)
        loss_phys_corr = self._pearson_corr_loss(
            spectra_recon,
            clean,
            derivative=self.corr_mode,
            savgol_win=self.corr_savgol_win,
            savgol_poly=self.corr_savgol_poly,
        )
        loss_phys_comb = 0.5 * loss_phys_huber + 0.5 * loss_phys_corr

        per_param_losses = []
        for j, name in enumerate(self.predict_params):
            true_j = params_true_norm[:, self.name_to_idx[name]]
            mult = self.weight_mf if name == "mf_CH4" else 1.0
            lp = mult * F.mse_loss(params_pred_norm[:, j], true_j)
            per_param_losses.append(lp)

        if per_param_losses:
            per_param_tensor = torch.stack(per_param_losses)
            w_params = self.relo_params.compute_weights(per_param_tensor)
            w_params_norm = w_params / (w_params.sum() + 1e-12)
            loss_param_group = torch.sum(w_params_norm * per_param_tensor)
        else:
            loss_param_group = torch.tensor(0.0, device=clean.device)

        top_vec = torch.stack([loss_phys_comb, loss_phys_corr, loss_param_group])
        w_top = self.relo_top.compute_weights(top_vec)
        priors_top = torch.tensor([self.alpha_phys, self.alpha_phys, self.alpha_param], device=top_vec.device, dtype=top_vec.dtype)
        w_top = w_top * priors_top
        w_top = 3.0 * w_top / (w_top.sum() + 1e-12)
        loss = torch.sum(w_top * top_vec)

        self.log(f"{step_name}_loss", loss, on_epoch=True, sync_dist=True)
        self.log(f"{step_name}_loss_phys_huber", loss_phys_huber, on_epoch=True, sync_dist=True)
        self.log(f"{step_name}_loss_phys_corr", loss_phys_corr, on_epoch=True, sync_dist=True)
        self.log(f"{step_name}_loss_param_group", loss_param_group, on_epoch=True, sync_dist=True)
        if per_param_losses:
            self.log(f"{step_name}_loss_param", torch.stack(per_param_losses).mean(), on_epoch=True, sync_dist=True)
        self.log(f"{step_name}_w_top_phys", w_top[0], on_epoch=True, sync_dist=True)
        self.log(f"{step_name}_w_top_phys_corr", w_top[1], on_epoch=True, sync_dist=True)
        self.log(f"{step_name}_w_top_param_group", w_top[2], on_epoch=True, sync_dist=True)
        for j, name in enumerate(self.predict_params):
            self.log(f"{step_name}_loss_param_{name}", per_param_losses[j], on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, "val")

    def on_train_epoch_start(self):
        if self._override_stage is not None:
            stage = self._override_stage
            if stage == "A":
                self._set_requires_grad(self.refiner, False)
                self._set_requires_grad([self.backbone, self.shared_head, getattr(self, "out_head", None), getattr(self, "out_heads", None), self.film], True)
            elif stage == "B1":
                self._set_requires_grad([self.backbone, self.shared_head, getattr(self, "out_head", None), getattr(self, "out_heads", None), self.film], False)
                self._set_requires_grad(self.refiner, True)
            elif stage == "B2":
                self._set_requires_grad([self.backbone, self.shared_head, getattr(self, "out_head", None), getattr(self, "out_heads", None), self.film, self.refiner], True)
            return

        epoch = self.current_epoch
        stage3_start = self.refine_warmup_epochs + self.freeze_base_epochs

        if epoch < self.refine_warmup_epochs:
            self._set_requires_grad(self.refiner, False)
            self._set_requires_grad([self.backbone, self.shared_head, getattr(self, "out_head", None), getattr(self, "out_heads", None), self.film], True)
            self._froze_base = False
        elif epoch < stage3_start:
            if not self._froze_base:
                self._set_requires_grad([self.backbone, self.shared_head, getattr(self, "out_head", None), getattr(self, "out_heads", None), self.film], False)
                self._set_requires_grad(self.refiner, True)
                self._froze_base = True
        else:
            self._set_requires_grad([self.backbone, self.shared_head, getattr(self, "out_head", None), getattr(self, "out_heads", None), self.film, self.refiner], True)
            if epoch == stage3_start and hasattr(self.trainer, "optimizers") and self.trainer.optimizers:
                opt = self.trainer.optimizers[0]
                for group in opt.param_groups:
                    group["lr"] *= self.stage3_lr_shrink
                if self.stage3_refine_steps is not None:
                    self.refine_steps = int(self.stage3_refine_steps)
                if self.stage3_delta_scale is not None:
                    self.refiner.delta_scale = float(self.stage3_delta_scale)
                if self.stage3_alpha_phys is not None:
                    self.alpha_phys = float(self.stage3_alpha_phys)
                if self.stage3_alpha_param is not None:
                    self.alpha_param = float(self.stage3_alpha_param)

    def _set_requires_grad(self, modules, flag: bool):
        if modules is None:
            return
        if not isinstance(modules, (list, tuple)):
            modules = [modules]
        for module in modules:
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad_(flag)

    def configure_optimizers(self):
        base_params = list(self.backbone.parameters()) + list(self.shared_head.parameters())
        if hasattr(self, "out_head"):
            base_params += list(self.out_head.parameters())
        if hasattr(self, "out_heads"):
            base_params += list(self.out_heads.parameters())
        if self.film is not None:
            base_params += list(self.film.parameters())
        refiner_params = list(self.refiner.parameters())

        param_groups = [
            {"params": base_params, "lr": float(getattr(self, "base_lr", self.lr))},
            {"params": refiner_params, "lr": float(getattr(self, "refiner_lr", self.lr))},
        ]

        opt_name = getattr(self.hparams, "optimizer", "adamw").lower() if hasattr(self, "hparams") else "adamw"
        weight_decay = getattr(self, "weight_decay", 1e-4)

        if opt_name == "adamw":
            optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        elif opt_name == "lion":
            from lion_pytorch import Lion

            betas = getattr(self.hparams, "betas", (0.9, 0.99)) if hasattr(self, "hparams") else (0.9, 0.99)
            optimizer = Lion(param_groups, betas=betas, weight_decay=weight_decay)
        elif opt_name == "radam":
            import torch_optimizer as optim_plus

            optimizer = optim_plus.RAdam(param_groups, weight_decay=weight_decay)
        elif opt_name == "adabelief":
            import torch_optimizer as optim_plus

            optimizer = optim_plus.AdaBelief(param_groups, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer is not None else 100,
            eta_min=1e-11,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @torch.no_grad()
    def infer(
        self,
        spectra: torch.Tensor,
        provided_phys: dict,
        *,
        refine: bool = True,
        resid_target: str = "input",
        scale: Optional[torch.Tensor] = None,
    ):
        self.eval()
        device = spectra.device
        batch_size = spectra.shape[0]

        missing = [n for n in self.provided_params if n not in provided_phys]
        if missing:
            raise ValueError(f"Missing provided parameters: {missing}")

        cond_norm = self._make_condition_from_phys(provided_phys, device, dtype=torch.float32)

        latent, _ = self.backbone(spectra.unsqueeze(1))
        feat_shared = self.feature_head(latent)
        params_pred_norm = self._predict_params_from_features(feat_shared, cond_norm=cond_norm)

        if len(self.provided_params) > 0:
            provided_list = [provided_phys[n].to(device) for n in self.provided_params]
            provided_phys_tensor = torch.stack(provided_list, dim=1)
        else:
            provided_phys_tensor = params_pred_norm.new_zeros((batch_size, 0))

        spectra_target = spectra if resid_target in ("input", "noisy") else None
        scale_est = lowess_value(spectra, kind="start", window=30).unsqueeze(1).clamp_min(1e-8)

        if refine and self.refine_steps > 0:
            for step in range(self.refine_steps):
                pred_phys = self._denorm_params_subset(params_pred_norm, self.predict_params)
                y_full = self._compose_full_phys(pred_phys, provided_phys_tensor)
                recon = self._physics_reconstruction(y_full, device, scale=None)
                if spectra_target is None:
                    break
                resid = recon - spectra_target
                delta = self.refiner(
                    noisy=spectra,
                    resid=resid,
                    params_pred_norm=params_pred_norm,
                    cond_norm=cond_norm,
                    feat_shared=feat_shared,
                    t_step=step,
                )
                params_pred_norm = params_pred_norm.add(delta).clamp(1e-4, 1 - 1e-4)

        pred_phys = self._denorm_params_subset(params_pred_norm, self.predict_params)
        y_full = self._compose_full_phys(pred_phys, provided_phys_tensor)
        recon = self._physics_reconstruction(y_full, device, scale=None)

        return {
            "params_pred_norm": params_pred_norm,
            "y_phys_full": y_full,
            "spectra_recon": recon,
            "norm_scale": scale_est,
        }

