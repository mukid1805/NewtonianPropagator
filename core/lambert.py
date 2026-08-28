"""
core/lambert.py
---------------
High-precision universal variable Lambert Problem solver (Bate-Mueller-White formulation).
Handles elliptical, parabolic, and hyperbolic two-point boundary value transfers
with sub-meter terminal targeting precision and robust divergence clamping.
"""
import numpy as np
from typing import Tuple


def _stumpff_c(z: float) -> float:
    """Evaluates Stumpff function C(z)."""
    if z > 1e-6:
        return (1.0 - np.cos(np.sqrt(z))) / z
    elif z < -1e-6:
        return (np.cosh(np.sqrt(-z)) - 1.0) / (-z)
    else:
        return 0.5 - z / 24.0 + (z**2) / 720.0


def _stumpff_s(z: float) -> float:
    """Evaluates Stumpff function S(z)."""
    if z > 1e-6:
        sz = np.sqrt(z)
        return (sz - np.sin(sz)) / (sz**3)
    elif z < -1e-6:
        sz = np.sqrt(-z)
        return (np.sinh(sz) - sz) / (sz**3)
    else:
        return (1.0 / 6.0) - z / 120.0 + (z**2) / 5040.0


def solve_lambert(
    r1: np.ndarray,
    r2: np.ndarray,
    tof: float,
    mu: float,
    prograde: bool = True,
    max_iter: int = 100,
    tol: float = 1e-11
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves Lambert's Problem given position vectors r1 and r2, time of flight, and mu.
    Returns: (v1, v2)
    """
    if tof <= 0.0:
        raise ValueError("Time of flight (tof) must be positive.")
    if mu <= 0.0:
        raise ValueError("Gravitational parameter (mu) must be positive.")

    r1 = np.asarray(r1, dtype=np.float64)
    r2 = np.asarray(r2, dtype=np.float64)
    r1_norm = float(np.linalg.norm(r1))
    r2_norm = float(np.linalg.norm(r2))

    # True anomaly transfer angle dtheta
    cos_dtheta = np.dot(r1, r2) / (r1_norm * r2_norm)
    cos_dtheta = float(np.clip(cos_dtheta, -1.0, 1.0))
    cross_12 = np.cross(r1, r2)

    if prograde:
        dtheta = np.arccos(cos_dtheta) if cross_12[2] >= 0.0 else 2.0 * np.pi - np.arccos(cos_dtheta)
    else:
        dtheta = np.arccos(cos_dtheta) if cross_12[2] <= 0.0 else 2.0 * np.pi - np.arccos(cos_dtheta)

    A = np.sin(dtheta) * np.sqrt(r1_norm * r2_norm / (1.0 - cos_dtheta))
    if abs(A) < 1e-12:
        raise ValueError("180-degree transfer plane singularity in Lambert solver.")

    # Newton-Raphson root solver for universal variable z
    z = 0.0
    converged = False

    for _ in range(max_iter):
        cz = max(1e-15, _stumpff_c(z))
        sz = _stumpff_s(z)
        yz = r1_norm + r2_norm + A * (z * sz - 1.0) / np.sqrt(cz)

        if yz < 0.0:
            raise ValueError("Lambert unfeasible: y(z) < 0 (unphysical trajectory).")

        tof_calc = ((yz / cz)**1.5 * sz + A * np.sqrt(yz)) / np.sqrt(mu)
        f_val = tof_calc - tof

        if abs(f_val) < tol:
            converged = True
            break

        # Analytical/central difference gradient
        dz = 1e-5
        cz_plus = max(1e-15, _stumpff_c(z + dz))
        sz_plus = _stumpff_s(z + dz)
        yz_plus = r1_norm + r2_norm + A * ((z + dz) * sz_plus - 1.0) / np.sqrt(cz_plus)

        if yz_plus < 0.0:
            dt_dz = 1e-8  # Prevent invalid state
        else:
            t_plus = ((yz_plus / cz_plus)**1.5 * sz_plus + A * np.sqrt(yz_plus)) / np.sqrt(mu)
            dt_dz = (t_plus - tof_calc) / dz

        if abs(dt_dz) < 1e-15:
            break

        step = f_val / dt_dz

        # CLAMPING: Prevent massive jumps into overflow territory
        step = max(-50.0, min(50.0, step))
        z = z - step

        # LIMITS: Restrict z to physical 1-rev domain [-10000 (hyperbola), 4*pi^2 (ellipse)]
        z = max(-10000.0, min(39.4, z))

    if not converged:
        raise ValueError("Lambert solver failed to converge.")

    # Compute final state vectors
    yz = r1_norm + r2_norm + A * (z * _stumpff_s(z) - 1.0) / np.sqrt(max(1e-15, _stumpff_c(z)))
    f = 1.0 - yz / r1_norm
    g = A * np.sqrt(yz / mu)
    gdot = 1.0 - yz / r2_norm

    v1 = (r2 - f * r1) / g
    v2 = (gdot * r2 - r1) / g

    return v1, v2