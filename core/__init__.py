"""
Core astrodynamics engine modules: constants, forces, integrators, propagators, and launchers.
"""

try:
    from core._version import version as __version__
except ImportError:
    __version__ = "unknown"

# Numerical integrators
from core.integrators import rk4, rk4_step, rk45_adaptive

# Propagation engines
from core.propagator import SpacecraftPropagator

# Launch vehicles & performance curves
from core.launchers import LaunchVehicle, get_launcher, list_available_launchers

# Physical & astrodynamic constants
from core.constants import (
    AU,
    G,
    G0,
    G_EARTH,
    G_MARS,
    G_MOON,
    G_SUN,
    G_VENUS,
    R_EARTH,
    R_MARS,
    R_MOON,
    R_VENUS,
)

__all__ = [
    "__version__",
    "SpacecraftPropagator",
    "rk4",
    "rk4_step",
    "rk45_adaptive",
    "LaunchVehicle",
    "get_launcher",
    "list_available_launchers",
    "G",
    "G0",
    "AU",
    "G_EARTH",
    "R_EARTH",
    "G_SUN",
    "G_MOON",
    "R_MOON",
    "G_MARS",
    "R_MARS",
    "G_VENUS",
    "R_VENUS",
]