"""
Acceleration models for the Newtonian superposition summation junction.
All functions accept position vectors r [m] and velocity vectors v [m/s] in ECI coordinates.
"""
import numpy as np
from core.constants import (
    G_EARTH, G_MOON, R_EARTH, J2_EARTH, J3_EARTH, J4_EARTH, OMEGA_EARTH,
    RHO_0, SCALE_HEIGHT, R_MOON_ORBIT, OMEGA_MOON, AU, P_SUN_1AU
)

def accel_earth_gravity(r: np.ndarray) -> np.ndarray:
    """Primary central body Newtonian gravitational acceleration: a = -GM/r^3 * r."""
    r_mag = np.linalg.norm(r)
    return -G_EARTH * r / (r_mag ** 3)


def accel_j2_perturbation(r: np.ndarray) -> np.ndarray:
    """Nonspherical Earth geopotential J2 harmonic perturbation."""
    x, y, z = r
    r_mag = np.linalg.norm(r)
    factor = 1.5 * J2_EARTH * G_EARTH * (R_EARTH ** 2) / (r_mag ** 5)
    z_sq = (z / r_mag) ** 2
    return factor * np.array([
        x * (5.0 * z_sq - 1.0),
        y * (5.0 * z_sq - 1.0),
        z * (5.0 * z_sq - 3.0)
    ])

def accel_j3_perturbation(r: np.ndarray) -> np.ndarray:
    """Nonspherical Earth geopotential J3 harmonic perturbation (pear shape)."""
    x, y, z = r
    r_mag = np.linalg.norm(r)
    s = z / r_mag  # sin(phi)

    factor = 0.5 * J3_EARTH * G_EARTH * (R_EARTH ** 3) / (r_mag ** 5)

    ax = 5.0 * factor * x * (7.0 * (s ** 3) - 3.0 * s)
    ay = 5.0 * factor * y * (7.0 * (s ** 3) - 3.0 * s)
    az = factor * (35.0 * (s ** 4) - 30.0 * (s ** 2) + 3.0) * r_mag

    return np.array([ax, ay, az])


def accel_j4_perturbation(r: np.ndarray) -> np.ndarray:
    """Nonspherical Earth geopotential J4 harmonic perturbation."""
    x, y, z = r
    r_mag = np.linalg.norm(r)
    s = z / r_mag

    factor = (5.0 / 8.0) * J4_EARTH * G_EARTH * (R_EARTH ** 4) / (r_mag ** 6)

    term_xy = 3.0 - 42.0 * (s ** 2) + 63.0 * (s ** 4)
    term_z = 15.0 - 70.0 * (s ** 2) + 63.0 * (s ** 4)

    ax = factor * x * term_xy
    ay = factor * y * term_xy
    az = factor * z * term_z

    return np.array([ax, ay, az])


def rv_to_keplerian(r: np.ndarray, v: np.ndarray, mu: float = G_EARTH) -> dict:
    """
    Converts Cartesian state (r, v) to Classical Orbital Elements (COE):
    returns: a (semi-major axis), e (eccentricity), inc (inclination),
             raan (RAAN), argp (argument of perigee), nu (true anomaly).
    """
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    h_vec = np.cross(r, v)
    h_mag = np.linalg.norm(h_vec)

    # Specific mechanical energy
    energy = 0.5 * (v_mag ** 2) - (mu / r_mag)
    a = -mu / (2.0 * energy) if abs(energy) > 1e-12 else np.nan

    # Eccentricity vector
    e_vec = (1.0 / mu) * ((v_mag ** 2 - mu / r_mag) * r - np.dot(r, v) * v)
    e = np.linalg.norm(e_vec)

    # Inclination
    inc = np.arccos(np.clip(h_vec[2] / h_mag, -1.0, 1.0))

    # Node vector
    k_hat = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(k_hat, h_vec)
    n_mag = np.linalg.norm(n_vec)

    # RAAN (Omega)
    if n_mag > 1e-9:
        raan = np.arccos(np.clip(n_vec[0] / n_mag, -1.0, 1.0))
        if n_vec[1] < 0:
            raan = 2.0 * np.pi - raan
    else:
        raan = 0.0

    # Argument of Perigee (omega)
    if n_mag > 1e-9 and e > 1e-6:
        argp = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n_mag * e), -1.0, 1.0))
        if e_vec[2] < 0:
            argp = 2.0 * np.pi - argp
    else:
        argp = 0.0

    return {
        "a": a,
        "e": e,
        "inc_deg": np.degrees(inc),
        "raan_deg": np.degrees(raan),
        "argp_deg": np.degrees(argp)
    }

def accel_lunar_gravity(r: np.ndarray, t: float) -> np.ndarray:
    """Third-body lunar gravitational attraction (direct minus indirect acceleration)."""
    r_moon = np.array([
        R_MOON_ORBIT * np.cos(OMEGA_MOON * t),
        R_MOON_ORBIT * np.sin(OMEGA_MOON * t),
        0.0
    ])
    r_rel = r_moon - r
    return G_MOON * (r_rel / (np.linalg.norm(r_rel) ** 3) - r_moon / (R_MOON_ORBIT ** 3))


def accel_atmospheric_drag(r: np.ndarray, v: np.ndarray, cd: float, area: float, mass: float) -> np.ndarray:
    """
    Atmospheric drag accounting for Earth rotation and exponential density decay:
    a_drag = -0.5 * rho * (Cd * A / m) * |v_rel| * v_rel
    """
    altitude = np.linalg.norm(r) - R_EARTH
    if altitude < 0:
        return np.array([0.0, 0.0, 0.0])  # Spacecraft has impacted Earth surface

    rho = RHO_0 * np.exp(-altitude / SCALE_HEIGHT)
    omega_vec = np.array([0.0, 0.0, OMEGA_EARTH])
    v_rel = v - np.cross(omega_vec, r)
    v_rel_mag = np.linalg.norm(v_rel)

    return -0.5 * rho * (cd * area / mass) * v_rel_mag * v_rel


def accel_solar_radiation_pressure(r: np.ndarray, cr: float, area: float, mass: float) -> np.ndarray:
    """
    Cannonball Solar Radiation Pressure (SRP) with cylindrical Earth shadow eclipse check.
    """
    r_sun = np.array([AU, 0.0, 0.0])
    r_sc_sun = r - r_sun
    sun_dir = r_sc_sun / np.linalg.norm(r_sc_sun)

    # Cylindrical Earth shadow check
    if r[0] < 0 and np.linalg.norm(r[1:3]) < R_EARTH:
        return np.array([0.0, 0.0, 0.0])  # Umbra

    return P_SUN_1AU * cr * (area / mass) * sun_dir


def accel_fixed_thrust(t: float, start_t: float, duration: float, thrust_vec: np.ndarray, mass: float) -> np.ndarray:
    """Impulsive or constant timed directional thrust vector."""
    if start_t <= t <= (start_t + duration):
        return thrust_vec / mass
    return np.array([0.0, 0.0, 0.0])


def accel_electric_prograde(v: np.ndarray, thrust_mag: float, mass: float) -> np.ndarray:
    """Continuous low-thrust acceleration steered along velocity unit vector."""
    v_mag = np.linalg.norm(v)
    if v_mag == 0.0:
        return np.array([0.0, 0.0, 0.0])
    return (thrust_mag / mass) * (v / v_mag)


def eci_to_lvlh(r_chief: np.ndarray, v_chief: np.ndarray, r_deputy: np.ndarray) -> np.ndarray:
    """
    Transforms deputy position relative to chief from ECI to LVLH (Hill's) frame.
    Returns:
        delta_r_lvlh = [x_radial, y_along_track, z_cross_track] in meters.
    """
    # Unit vectors of LVLH frame
    r_c_mag = np.linalg.norm(r_chief)
    e_r = r_chief / r_c_mag

    h_vec = np.cross(r_chief, v_chief)
    e_h = h_vec / np.linalg.norm(h_vec)

    e_theta = np.cross(e_h, e_r)

    # Rotation matrix (rows are the unit basis vectors)
    r_eci_to_lvlh = np.vstack([e_r, e_theta, e_h])

    # Relative displacement in ECI
    delta_r_eci = r_deputy - r_chief

    # Transform to LVLH
    return r_eci_to_lvlh @ delta_r_eci