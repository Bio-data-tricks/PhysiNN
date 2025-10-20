"""Custom loss utilities."""

from __future__ import annotations

from typing import Iterable, List

import torch


class ReLoBRaLoLoss:
    def __init__(self, loss_names: Iterable[str], alpha: float = 0.9, tau: float = 1.0, history_len: int = 10, seed: int | None = 12345):
        self.loss_names = list(loss_names)
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.history_len = int(history_len)
        self.loss_history = {name: [] for name in self.loss_names}
        self.weights = torch.ones(len(self.loss_names), dtype=torch.float32)
        self._generator = torch.Generator(device="cpu")
        if seed is not None:
            self._generator.manual_seed(int(seed))

    def set_seed(self, seed: int) -> None:
        self._generator.manual_seed(int(seed))

    def to(self, device=None, dtype=None):
        if device is not None or dtype is not None:
            self.weights = self.weights.to(device=device or self.weights.device, dtype=dtype or self.weights.dtype)
        return self

    def _append_history(self, current_losses: torch.Tensor) -> None:
        for idx, name in enumerate(self.loss_names):
            self.loss_history[name].append(float(current_losses[idx].detach().cpu()))
            if len(self.loss_history[name]) > self.history_len:
                self.loss_history[name].pop(0)

    @torch.no_grad()
    def compute_weights(self, current_losses: torch.Tensor) -> torch.Tensor:
        device = current_losses.device
        dtype = current_losses.dtype
        self._append_history(current_losses)
        if len(self.loss_history[self.loss_names[0]]) < 2:
            return self.weights.to(device=device, dtype=dtype)
        ratios: List[float] = []
        for name in self.loss_names:
            history = self.loss_history[name]
            j = int(torch.randint(low=0, high=len(history) - 1, size=(), generator=self._generator).item())
            numerator = float(history[-1])
            denominator = float(history[j]) + 1e-8
            ratios.append(numerator / denominator)
        ratios_tensor = torch.tensor(ratios, device=device, dtype=dtype)
        mean_rel = ratios_tensor.mean()
        balancing = mean_rel / (ratios_tensor + 1e-8)
        num_losses = len(self.loss_names)
        new_weights = num_losses * torch.softmax(balancing / self.tau, dim=0)
        old_weights = self.weights.to(device=device, dtype=dtype)
        blended = self.alpha * old_weights + (1.0 - self.alpha) * new_weights
        self.weights = blended.detach().cpu()
        return blended

