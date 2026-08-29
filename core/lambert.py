"""
Universal Variables Lambert Solver (Bate-Mueller-White formulation).
Includes Canonical Scaling and strict bounds for extreme hyperbolic/elliptical transfers.
"""
import numpy as np


def _stumpff_c(z: float) -> float:
    if z > 1e-6:
        return (1.0 - np.cos(np.sqrt(z))) / z
    elif z < -1e-6:
        return (np.cosh(np.sqrt(-z)) - 1.0) / (-z)
    return 0.5 - z / 24.0 + (z**2) / 720.0


def _stumpff_s(z: float) -> float:
    if z > 1e-6:
        sz = np.sqrt(z)
        return (sz - np.sin(sz)) / (sz**3)
    elif z < -1e-6:
        sz = np.sqrt(-z)
        return (np.sinh(sz) - sz) / (sz**3)
    return (1.0 / 6.0) - z / 120.0 + (z**2) / 5040.0


def solve_lambert(r1, r2, tof, mu, prograde=True, max_iter=100, tol=1e-10):
    """
    Solves the Lambert Boundary Value Problem.
    Returns v1, v2 (velocity vectors at r1 and r2).
    """
    # Canonical Scaling for numerical stability
    r_unit = float(np.linalg.norm(r1))
    v_unit = np.sqrt(mu / r_unit)
    t_unit = r_unit / v_unit

    r1_nd = np.asarray(r1, dtype=np.float64) / r_unit
    r2_nd = np.asarray(r2, dtype=np.float64) / r_unit
    tof_nd = tof / t_unit
    mu_nd = 1.0

    r1_norm = 1.0
    r2_norm = float(np.linalg.norm(r2_nd))

    cos_dtheta = np.clip(np.dot(r1_nd, r2_nd) / (r1_norm * r2_norm), -1.0, 1.0)
    cross_12 = np.cross(r1_nd, r2_nd)

    if prograde:
        dtheta = np.arccos(cos_dtheta) if cross_12[2] >= 0.0 else 2.0 * np.pi - np.arccos(cos_dtheta)
    else:
        dtheta = np.arccos(cos_dtheta) if cross_12[2] <= 0.0 else 2.0 * np.pi - np.arccos(cos_dtheta)

    sin_dtheta = np.sin(dtheta)
    A = sin_dtheta * np.sqrt(r1_norm * r2_norm / (1.0 - cos_dtheta))

    if abs(A) < 1e-12:
        raise ValueError("Collinear singularity (180 degree transfer).")

    z = 0.0
    converged = False

    for _ in range(max_iter):
        cz = max(1e-15, _stumpff_c(z))
        sz = _stumpff_s(z)
        yz = r1_norm + r2_norm + A * (z * sz - 1.0) / np.sqrt(cz)

        if yz < 0.0:
            # Step back if unphysical
            z += 1.0
            continue

        tof_calc = ((yz / cz)**1.5 * sz + A * np.sqrt(yz)) / np.sqrt(mu_nd)
        f_val = tof_calc - tof_nd

        if abs(f_val) < tol:
            converged = True
            break

        dz = 1e-5
        cz_plus = max(1e-15, _stumpff_c(z + dz))
        sz_plus = _stumpff_s(z + dz)
        yz_plus = r1_norm + r2_norm + A * ((z + dz) * sz_plus - 1.0) / np.sqrt(cz_plus)

        if yz_plus < 0.0:
            dt_dz = 1e-8
        else:
            t_plus = ((yz_plus / cz_plus)**1.5 * sz_plus + A * np.sqrt(yz_plus)) / np.sqrt(mu_nd)
            dt_dz = (t_plus - tof_calc) / dz

        if abs(dt_dz) < 1e-15:
            break

        # Newton Step with strict clamping
        step = f_val / dt_dz
        step = max(-50.0, min(50.0, step))
        z = z - step
        z = max(-10000.0, min(39.4, z))

    if not converged:
        raise ValueError("Lambert solver failed to converge.")

    yz = r1_norm + r2_norm + A * (z * _stumpff_s(z) - 1.0) / np.sqrt(max(1e-15, _stumpff_c(z)))
    f = 1.0 - yz / r1_norm
    g = A * np.sqrt(yz / mu_nd)
    gdot = 1.0 - yz / r2_norm

    v1_nd = (r2_nd - f * r1_nd) / g
    v2_nd = (gdot * r2_nd - r1_nd) / g

    return v1_nd * v_unit, v2_nd * v_unit