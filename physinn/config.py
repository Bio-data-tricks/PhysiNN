"""Configuration and shared constants for PhysiNN."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, MutableMapping, Tuple

PARAMS = [
    "sig0",
    "dsig",
    "mf_CH4",
    "mf_H2O",
    "baseline0",
    "baseline1",
    "baseline2",
    "P",
    "T",
]
PARAM_TO_IDX = {name: idx for idx, name in enumerate(PARAMS)}
LOG_SCALE_PARAMS = {"mf_CH4", "mf_H2O"}
LOG_FLOOR = 1e-7


@dataclass
class NormalizationRegistry:
    """Registry storing min/max ranges used for parameter normalisation."""

    ranges: MutableMapping[str, Tuple[float, float]] = field(default_factory=dict)

    def update(self, new_ranges: Mapping[str, Tuple[float, float]]) -> None:
        self._validate_keys(new_ranges.keys())
        self.ranges.update({k: (float(v[0]), float(v[1])) for k, v in new_ranges.items()})

    def get(self, name: str) -> Tuple[float, float]:
        if name not in self.ranges:
            raise KeyError(f"Normalisation range for '{name}' is not registered.")
        return self.ranges[name]

    def as_dict(self) -> Dict[str, Tuple[float, float]]:
        return dict(self.ranges)

    def clear(self) -> None:
        self.ranges.clear()

    def _validate_keys(self, keys: Iterable[str]) -> None:
        unknown = set(keys) - set(PARAMS)
        if unknown:
            raise KeyError(f"Unknown parameter(s) for normalisation: {sorted(unknown)}")


NORMALIZATION = NormalizationRegistry()

