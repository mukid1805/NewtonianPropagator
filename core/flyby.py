r"""Hyperbolic gravity assist (flyby) dynamics and patched-conic matching engine.

Evaluates turning angles, periapsis altitudes, and required powered flyby Delta-Vs.
"""

from typing import Any, Dict
import numpy as np
from core.ephemeris import PLANET_DATA


def evaluate_unpowered_flyby(
    v_inf_in: np.ndarray,
    v_inf_out: np.ndarray,
    planet_name: str,
    min_altitude_km: float = 300.0
) -> Dict[str, Any]:
    r"""Evaluate an unpowered hyperbolic gravity assist maneuver.

    Args:
        v_inf_in: Inbound hyperbolic excess velocity vector $\mathbf{v}_\infty^-$
            relative to planet [$\text{km/s}$].
        v_inf_out: Outbound hyperbolic excess velocity vector $\mathbf{v}_\infty^+$
            relative to planet [$\text{km/s}$].
        planet_name: Flyby planet key (e.g. `'venus'`, `'earth'`, `'mars'`, `'jupiter'`).
        min_altitude_km: Atmospheric clearance margin above planet radius [$\text{km}$]. Defaults to `300.0`.

    Returns:
        Dict[str, Any]: Flyby evaluation parameters containing:
            * `is_feasible` (*bool*): Unpowered geometric and altitude feasibility.
            * `delta_angle_deg` (*float*): Geometric turn angle $\delta$ [$\text{deg}$].
            * `v_inf_in_mag` (*float*): Inbound excess speed $\|\mathbf{v}_\infty^-\|$ [$\text{km/s}$].
            * `v_inf_out_mag` (*float*): Outbound excess speed $\|\mathbf{v}_\infty^+\|$ [$\text{km/s}$].
            * `dv_powered` (*float*): Deficit $\Delta V$ required if speeds mismatch [$\text{km/s}$].
            * `r_p` (*float*): Computed periapsis radius [$\text{km}$].
            * `h_p` (*float*): Computed periapsis altitude above surface ($r_p - R_p$) [$\text{km}$].
    """
    p_info = PLANET_DATA[planet_name.lower()]
    mu_p = p_info['mu']
    r_body = p_info['radius']

    v_in_mag = float(np.linalg.norm(v_inf_in))
    v_out_mag = float(np.linalg.norm(v_inf_out))

    cos_delta = float(np.dot(v_inf_in, v_inf_out) / (v_in_mag * v_out_mag))
    cos_delta = float(np.clip(cos_delta, -1.0, 1.0))
    delta = np.arccos(cos_delta)

    dv_powered = abs(v_out_mag - v_in_mag)
    v_inf_avg = 0.5 * (v_in_mag + v_out_mag)

    sin_half_delta = np.sin(delta / 2.0)

    if sin_half_delta <= 1e-6:
        r_p = np.inf
        h_p = np.inf
        is_feasible = True
    else:
        e = 1.0 / sin_half_delta
        r_p = (mu_p / (v_inf_avg ** 2)) * (e - 1.0)
        h_p = r_p - r_body
        is_feasible = (h_p >= min_altitude_km) and (dv_powered < 0.25)

    return {
        'is_feasible': bool(is_feasible),
        'delta_angle_deg': float(np.degrees(delta)),
        'v_inf_in_mag': float(v_in_mag),
        'v_inf_out_mag': float(v_out_mag),
        'dv_powered': float(dv_powered),
        'r_p': float(r_p),
        'h_p': float(h_p)
    }