r"""Circular Restricted Three-Body Problem (CR3BP) Engine.

Supports non-dimensional synodic equations of motion, Jacobi energy constant
calculations, Lagrange libration point ($L_1$-$L_5$) solvers, and coordinate
transformations into dimensional inertial reference frames.
"""

from typing import Dict, Final
import numpy as np
from scipy.optimize import root_scalar

# Earth-Moon characteristic parameters
MU_EARTH_MOON: Final[float] = 0.01215058560962404
L_STAR: Final[float] = 384_400_000.0
T_STAR: Final[float] = 375_190.258
V_STAR: Final[float] = L_STAR / T_STAR


def cr3bp_derivatives(t: float, state: np.ndarray, mu: float = MU_EARTH_MOON) -> np.ndarray:
    r"""Evaluate equations of motion in the non-dimensional synodic rotating frame.

    Args:
        t: Non-dimensional time parameter $\tau$.
        state: Non-dimensional state vector $[x, y, z, \dot{x}, \dot{y}, \dot{z}]$ of shape `(6,)`.
        mu: Primary mass ratio $\mu = \frac{m_2}{m_1 + m_2}$. Defaults to `MU_EARTH_MOON`.

    Returns:
        np.ndarray: State derivatives $[\dot{x}, \dot{y}, \dot{z}, \ddot{x}, \ddot{y}, \ddot{z}]$ of shape `(6,)`.
    """
    x, y, z, vx, vy, vz = state

    r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1.0 + mu) ** 2 + y ** 2 + z ** 2)

    omega_x = x - (1.0 - mu) * (x + mu) / (r1 ** 3) - mu * (x - 1.0 + mu) / (r2 ** 3)
    omega_y = y - (1.0 - mu) * y / (r1 ** 3) - mu * y / (r2 ** 3)
    omega_z = -(1.0 - mu) * z / (r1 ** 3) - mu * z / (r2 ** 3)

    ax = 2.0 * vy + omega_x
    ay = -2.0 * vx + omega_y
    az = omega_z

    return np.array([vx, vy, vz, ax, ay, az])


def compute_jacobi_constant(state: np.ndarray, mu: float = MU_EARTH_MOON) -> float:
    r"""Compute the conserved Jacobi Energy Constant ($C_J$).

    $$
    C_J = 2\Omega(x, y, z) - v^2
    $$

    Args:
        state: Non-dimensional synodic state vector $[x, y, z, \dot{x}, \dot{y}, \dot{z}]$ of shape `(6,)`.
        mu: Primary mass ratio $\mu$. Defaults to `MU_EARTH_MOON`.

    Returns:
        float: Conserved Jacobi energy constant $C_J$.
    """
    x, y, z, vx, vy, vz = state
    r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1.0 + mu) ** 2 + y ** 2 + z ** 2)

    v_sq = vx ** 2 + vy ** 2 + vz ** 2
    omega = 0.5 * (x ** 2 + y ** 2) + (1.0 - mu) / r1 + mu / r2

    return float(2.0 * omega - v_sq)


def compute_lagrange_points(mu: float = MU_EARTH_MOON) -> Dict[str, np.ndarray]:
    r"""Compute non-dimensional coordinates of all five Lagrange equilibrium points.

    Args:
        mu: Primary mass ratio $\mu$. Defaults to `MU_EARTH_MOON`.

    Returns:
        Dict[str, np.ndarray]: Dictionary mapping labels (`'L1'` through `'L5'`)
            to synodic position vectors $[x, y, z]$ of shape `(3,)`.
    """
    def domega_dx(x):
        r1 = abs(x + mu)
        r2 = abs(x - 1.0 + mu)
        return x - (1.0 - mu) * np.sign(x + mu) / (r1 ** 2) - mu * np.sign(x - 1.0 + mu) / (r2 ** 2)

    l1_x = root_scalar(domega_dx, bracket=[0.0, 1.0 - mu - 1e-4]).root
    l2_x = root_scalar(domega_dx, bracket=[1.0 - mu + 1e-4, 1.5]).root
    l3_x = root_scalar(domega_dx, bracket=[-1.5, -mu - 1e-4]).root

    return {
        "L1": np.array([l1_x, 0.0, 0.0]),
        "L2": np.array([l2_x, 0.0, 0.0]),
        "L3": np.array([l3_x, 0.0, 0.0]),
        "L4": np.array([0.5 - mu, np.sqrt(3.0) / 2.0, 0.0]),
        "L5": np.array([0.5 - mu, -np.sqrt(3.0) / 2.0, 0.0]),
    }


def synodic_to_inertial(
    times_nd: np.ndarray,
    states_nd: np.ndarray,
    mu: float = MU_EARTH_MOON
) -> np.ndarray:
    r"""Transform synodic state trajectories into dimensional ECI coordinates.

    Args:
        times_nd: Array of non-dimensional time steps of shape `(N,)`.
        states_nd: Array of non-dimensional synodic states of shape `(N, 6)`.
        mu: Primary mass ratio $\mu$. Defaults to `MU_EARTH_MOON`.

    Returns:
        np.ndarray: Dimensional ECI state vectors of shape `(N, 6)` in $[\text{km}, \text{km/s}]$.
    """
    num_steps = len(times_nd)
    states_eci_km = np.zeros((num_steps, 6))
    earth_offset_nd = np.array([-mu, 0.0, 0.0])

    for i in range(num_steps):
        t = times_nd[i]
        theta = t

        cos_t, sin_t = np.cos(theta), np.sin(theta)

        r_rot = np.array([
            [cos_t, -sin_t, 0.0],
            [sin_t,  cos_t, 0.0],
            [0.0,    0.0,   1.0]
        ])

        r_dot = np.array([
            [-sin_t, -cos_t, 0.0],
            [ cos_t, -sin_t, 0.0],
            [ 0.0,    0.0,   0.0]
        ])

        r_syn = states_nd[i, 0:3] - earth_offset_nd
        v_syn = states_nd[i, 3:6]

        r_eci_nd = r_rot @ r_syn
        v_eci_nd = r_rot @ v_syn + r_dot @ r_syn

        states_eci_km[i, 0:3] = r_eci_nd * (L_STAR / 1000.0)
        states_eci_km[i, 3:6] = v_eci_nd * (V_STAR / 1000.0)

    return states_eci_km
