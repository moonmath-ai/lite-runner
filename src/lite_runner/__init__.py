"""lite-runner: Runner for generative models with local and W&B tracking.

© 2026 MoonMath.ai. Some rights reserved.
"""
import logging

from ._version import version as _version
from .backends import JsonBackend, LogBackend, WandbBackend
from .params import UNSET, Metric, Output, Param, ParamType
from .runner import Runner, RunResult

__version__ = _version
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "UNSET",
    "JsonBackend",
    "LogBackend",
    "Metric",
    "Output",
    "Param",
    "ParamType",
    "RunResult",
    "Runner",
    "WandbBackend",
]

