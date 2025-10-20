"""Distributed training helpers."""

from __future__ import annotations

import os
import socket
from typing import Tuple

import torch


def is_rank0() -> bool:
    return not torch.distributed.is_available() or not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def on_rank_zero() -> bool:
    return is_rank0()


def get_master_addr_and_port(default_port: int = 12910) -> Tuple[str, int]:
    master_addr = os.environ.get("MASTER_ADDR")
    if not master_addr:
        nodelist = os.environ.get("SLURM_NODELIST") or os.environ.get("SLURM_JOB_NODELIST")
        if nodelist:
            try:
                import shlex
                import subprocess

                cmd = f"scontrol show hostnames {shlex.quote(nodelist)} | head -n 1"
                master_addr = subprocess.check_output(cmd, shell=True).decode().strip()
            except Exception:
                master_addr = socket.gethostname()
        else:
            master_addr = socket.gethostname()
    master_port = int(os.environ.get("MASTER_PORT", default_port))
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    return master_addr, master_port


def choose_precision() -> str:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "bf16-mixed"
        major, _ = torch.cuda.get_device_capability(0)
        if major >= 7:
            return "16-mixed"
    return "32-true"

