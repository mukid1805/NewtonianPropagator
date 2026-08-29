"""
Circular Restricted Three-Body Problem (CR3BP) Module.
Supports non-dimensional synodic propagation, Lagrange point solvers, and dimensional conversions.
"""
import numpy as np
from scipy.optimize import root_scalar

# Earth-Moon characteristic parameters
MU_EARTH_MOON = 0.01215058560962404  # Mass parameter mu = m2 / (m1 + m2)
L_STAR = 384_400_000.0  # Characteristic length (Earth-Moon distance in meters)
T_STAR = 375_190.258  # Characteristic time (seconds) ~ 4.34 days
V_STAR = L_STAR / T_STAR  # Characteristic velocity (~1024.55 m/s)


def cr3bp_derivatives(t: float, state: np.ndarray, mu: float = MU_EARTH_MOON) -> np.ndarray:
    """
    Computes equations of motion in the non-dimensional synodic (rotating) frame:
    state = [x, y, z, vx, vy, vz]
    """
    x, y, z, vx, vy, vz = state

    r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1.0 + mu) ** 2 + y ** 2 + z ** 2)

    # Gradient of pseudo-potential dOmega/dx, dOmega/dy, dOmega/dz
    omega_x = x - (1.0 - mu) * (x + mu) / (r1 ** 3) - mu * (x - 1.0 + mu) / (r2 ** 3)
    omega_y = y - (1.0 - mu) * y / (r1 ** 3) - mu * y / (r2 ** 3)
    omega_z = - (1.0 - mu) * z / (r1 ** 3) - mu * z / (r2 ** 3)

    # Synodic accelerations including Coriolis terms (2*vy, -2*vx)
    ax = 2.0 * vy + omega_x
    ay = -2.0 * vx + omega_y
    az = omega_z

    return np.array([vx, vy, vz, ax, ay, az])


def compute_jacobi_constant(state: np.ndarray, mu: float = MU_EARTH_MOON) -> float:
    """Computes Jacobi Constant C = 2*Omega - v^2."""
    x, y, z, vx, vy, vz = state
    r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1.0 + mu) ** 2 + y ** 2 + z ** 2)

    v_sq = vx ** 2 + vy ** 2 + vz ** 2
    omega = 0.5 * (x ** 2 + y ** 2) + (1.0 - mu) / r1 + mu / r2

    return float(2.0 * omega - v_sq)


def compute_lagrange_points(mu: float = MU_EARTH_MOON) -> dict:
    """
    Computes coordinates of all 5 equilibrium points L1 through L5 in non-dimensional synodic frame.
    """

    # Collinear condition: dOmega/dx = 0 along y = z = 0
    def domega_dx(x):
        r1 = abs(x + mu)
        r2 = abs(x - 1.0 + mu)
        return x - (1.0 - mu) * np.sign(x + mu) / (r1 ** 2) - mu * np.sign(x - 1.0 + mu) / (r2 ** 2)

    # Root finding for collinear points
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


def synodic_to_inertial(times_nd: np.ndarray, states_nd: np.ndarray, mu: float = MU_EARTH_MOON) -> np.ndarray:
    """
    Transforms non-dimensional synodic states into dimensional Earth-Centered Inertial (ECI) coordinates (km, km/s).
    """
    num_steps = len(times_nd)
    states_eci_km = np.zeros((num_steps, 6))

    earth_offset_nd = np.array([-mu, 0.0, 0.0])

    for i in range(num_steps):
        t = times_nd[i]
        theta = t  # In non-dimensional units, n = 1 rad/nd_time

        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # Position rotation matrix
        r_rot = np.array([
            [cos_t, -sin_t, 0.0],
            [sin_t,  cos_t, 0.0],
            [0.0,    0.0,   1.0]
        ])

        # Derivative of rotation matrix (d/dt with n = 1)
        r_dot = np.array([
            [-sin_t, -cos_t, 0.0],
            [ cos_t, -sin_t, 0.0],
            [ 0.0,    0.0,   0.0]
        ])

        r_syn = states_nd[i, 0:3] - earth_offset_nd
        v_syn = states_nd[i, 3:6]

        # Kinematic velocity transformation: v_inertial = R * v_syn + R_dot * r_syn
        r_eci_nd = r_rot @ r_syn
        v_eci_nd = r_rot @ v_syn + r_dot @ r_syn

        # Convert to km and km/s
        states_eci_km[i, 0:3] = r_eci_nd * (L_STAR / 1000.0)
        states_eci_km[i, 3:6] = v_eci_nd * (V_STAR / 1000.0)

    return states_eci_km