"""Utilities for loading QTpy/TIPS 2021 partition functions."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import torch


class Tips2021QTpy:
    """Reader for pre-computed QTpy/TIPS 2021 partition functions."""

    def __init__(self, qtpy_dir: str | Path, device: str = "cpu") -> None:
        self.base = Path(qtpy_dir).resolve()
        if not self.base.exists():
            raise FileNotFoundError(f"QTpy directory not found: {self.base}")
        self.device = device
        self.cache_dict: Dict[tuple[int, int], Dict[int, float]] = {}
        self.cache_table: Dict[tuple[int, int], np.ndarray] = {}
        self.cache_tmax: Dict[tuple[int, int], int] = {}

    def _path_for(self, mol_id: int, iso: int) -> Path:
        return self.base / f"{int(mol_id)}_{int(iso)}.QTpy"

    def _load_one(self, mol_id: int, iso: int) -> None:
        key = (int(mol_id), int(iso))
        if key in self.cache_dict:
            return
        path = self._path_for(*key)
        if not path.exists():
            raise FileNotFoundError(f"Missing QTpy file for (mol={mol_id}, iso={iso}): {path}")
        with open(path, "rb") as handle:
            data = pickle.loads(handle.read())
        casted = {int(k): float(v) for k, v in data.items()}
        tmax = int(max(casted.keys()))
        table = np.zeros(tmax, dtype=np.float64)
        for temp in range(1, tmax + 1):
            if temp in casted:
                table[temp - 1] = casted[temp]
            else:
                prev = max([k for k in casted if k < temp], default=min(casted.keys()))
                nxt = min([k for k in casted if k > temp], default=max(casted.keys()))
                if nxt == prev:
                    table[temp - 1] = casted[prev]
                else:
                    alpha = (temp - prev) / (nxt - prev)
                    table[temp - 1] = casted[prev] + alpha * (casted[nxt] - casted[prev])
        self.cache_dict[key] = casted
        self.cache_table[key] = table
        self.cache_tmax[key] = tmax

    def q_scalar(self, mol_id: int, iso: int, temperature: float) -> float:
        self._load_one(mol_id, iso)
        key = (int(mol_id), int(iso))
        table = self.cache_table[key]
        tmax = self.cache_tmax[key]
        if temperature <= 1:
            return float(table[0])
        if temperature >= tmax:
            return float(table[-1])
        t0 = int(np.floor(temperature))
        t1 = t0 + 1
        frac = (temperature - t0) / (t1 - t0)
        q0, q1 = table[t0 - 1], table[t1 - 1]
        return float(q0 + frac * (q1 - q0))

    def q_torch(self, mol_id: int, iso: int, temperature: torch.Tensor) -> torch.Tensor:
        self._load_one(mol_id, iso)
        key = (int(mol_id), int(iso))
        table = self.cache_table[key]
        tmax = self.cache_tmax[key]
        temps = temperature.detach().to(dtype=torch.float64, device=self.device)
        temps = torch.clamp(temps, 1.0, float(tmax))
        t0 = torch.floor(temps)
        t1 = torch.clamp(t0 + 1.0, max=float(tmax))
        frac = (temps - t0) / torch.clamp(t1 - t0, min=1e-12)
        i0 = (t0.to(torch.int64) - 1).clamp(0, tmax - 1)
        i1 = (t1.to(torch.int64) - 1).clamp(0, tmax - 1)
        table_t = torch.from_numpy(table).to(dtype=torch.float64, device=self.device)
        q0 = table_t[i0]
        q1 = table_t[i1]
        return q0 + frac * (q1 - q0)

