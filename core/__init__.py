"""
Core astrodynamics engine modules: constants, forces, integrators, propagators, and launchers.
"""
try:
    from core._version import version as __version__
except ImportError:
    __version__ = "unknown"

from core.constants import *
from core.forces import *
from core.integrators import rk4_step
from core.propagator import SpacecraftPropagator
from core.launchers import LaunchVehicle, get_launcher, list_available_launchers

__all__ = [
    "__version__",
    "SpacecraftPropagator",
    "rk4_step",
    "LaunchVehicle",
    "get_launcher",
    "list_available_launchers",
]