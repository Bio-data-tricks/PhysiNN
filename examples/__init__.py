"""PhysiNN demonstration utilities and configuration examples.

This package exposes helper modules used by the example notebooks so that
``import examples.<module>`` works when the project root is on ``sys.path``.
"""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

__all__ = [
    "data_generation",
    "visualization",
]


def __getattr__(name: str) -> ModuleType:
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


if TYPE_CHECKING:  # pragma: no cover - for static type checkers only
    from . import data_generation, visualization  # noqa: F401
