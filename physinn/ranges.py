"""Utilities for manipulating parameter ranges."""

from __future__ import annotations

from typing import Callable, Dict


def expand_interval(a: float, b: float, factor: float) -> tuple[float, float]:
    centre = 0.5 * (a + b)
    half = 0.5 * (b - a) * float(factor)
    return float(centre - half), float(centre + half)


def map_ranges(base: Dict[str, tuple], fn: Callable[[float, float, float], tuple[float, float]], per_param: Dict[str, float] | None = None) -> Dict[str, tuple]:
    out: Dict[str, tuple] = {}
    for key, (lo, hi) in base.items():
        factor = per_param.get(key, per_param.get("_default", 1.0)) if per_param else 1.0
        out[key] = fn(lo, hi, factor)
    return out


def assert_subset(child: Dict[str, tuple], parent: Dict[str, tuple], name_child: str = "child", name_parent: str = "parent") -> None:
    offending = [key for key in child if not (parent[key][0] <= child[key][0] and child[key][1] <= parent[key][1])]
    if offending:
        raise ValueError(f"{name_child} ⊄ {name_parent} for: {offending}")

