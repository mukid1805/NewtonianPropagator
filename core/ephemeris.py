"""
core/ephemeris.py - High-accuracy Keplerian analytical planetary ephemeris
Based on Standish / JPL secular variations of planetary elements (1800-2050).
"""
import numpy as np
from typing import Tuple

# Gravitational parameters (km^3/s^2) and Astronomical Unit (km)
AU = 149597870.7
MU_SUN = 1.32712440018e11
OBLIQUITY_J2000 = np.radians(23.4392811)  # Earth mean obliquity

# Standish / JPL secular elements at J2000: [a (AU), e, i (deg), Om (deg), varpi (deg), L (deg)]
# Rates are per Julian century (T = (MJD - 51544.5) / 36525.0)
PLANET_DATA = {
    'mercury': {
        'mu': 22032.0, 'soi_a': 0.387098 * AU, 'mass_ratio': 1.660e-7,
        'base': [0.38709893, 0.20563069, 7.00487, 48.33167, 77.45645, 252.25084],
        'rates': [0.00000066, 0.00002527, -23.51 / 3600, -446.30 / 3600, 573.57 / 3600, 538101628.29 / 3600]
    },
    'venus': {
        'mu': 324859.0, 'soi_a': 0.723332 * AU, 'mass_ratio': 2.447e-6,
        'base': [0.72333199, 0.00677323, 3.39471, 76.68069, 131.53298, 181.97973],
        'rates': [0.00000092, -0.00004938, -2.86 / 3600, -996.89 / 3600, -108.80 / 3600, 210664136.06 / 3600]
    },
    'earth': {
        'mu': 398600.4418, 'soi_a': 1.000000 * AU, 'mass_ratio': 3.003e-6,
        'base': [1.00000011, 0.01671022, 0.00005, -11.26064, 102.94719, 100.46435],
        'rates': [-0.00000005, -0.00003804, -46.94 / 3600, -18228.25 / 3600, 1198.28 / 3600, 129597740.63 / 3600]
    },
    'mars': {
        'mu': 42828.37, 'soi_a': 1.523679 * AU, 'mass_ratio': 3.227e-7,
        'base': [1.52366231, 0.09341233, 1.85061, 49.55740, 336.04084, 355.45332],
        'rates': [-0.00007221, 0.00011902, -25.47 / 3600, -1020.19 / 3600, 1560.78 / 3600, 68905103.78 / 3600]
    },
    'jupiter': {
        'mu': 126686534.0, 'soi_a': 5.204267 * AU, 'mass_ratio': 9.543e-4,
        'base': [5.20336301, 0.04839266, 1.30530, 100.55615, 14.75385, 34.40438],
        'rates': [0.00060737, -0.00012880, -4.15 / 3600, 1217.17 / 3600, 839.93 / 3600, 10925078.35 / 3600]
    }
}


def get_planet_state(
        planet_name: str,
        mjd2000: float,
        frame: str = 'ecliptic'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Heliocentric 3D position and velocity (km, km/s).

    Parameters
    ----------
    planet_name : str
        Name of planet ('mercury', 'venus', 'earth', 'mars', 'jupiter').
    mjd2000 : float
        Days elapsed since J2000 epoch (Jan 1, 2000 12:00 TT).
    frame : str, default='ecliptic'
        Target coordinate frame: 'ecliptic' or 'equatorial' (ICRF/J2000).

    Returns
    -------
    r_vec : np.ndarray
        Position vector (3,) [km].
    v_vec : np.ndarray
        Velocity vector (3,) [km/s].
    """
    p = PLANET_DATA[planet_name.lower()]
    t_centuries = mjd2000 / 36525.0

    # Instantaneous Keplerian elements
    elem = [b + r * t_centuries for b, r in zip(p['base'], p['rates'])]
    a = elem[0] * AU
    e = elem[1]
    inc = np.radians(elem[2])
    raan = np.radians(elem[3])
    varpi = np.radians(elem[4])
    mean_long = np.radians(elem[5])

    omega = (varpi - raan) % (2.0 * np.pi)
    m_anom = (mean_long - varpi) % (2.0 * np.pi)

    # Solve Kepler's equation for Eccentric Anomaly (E)
    e_anom = m_anom if e < 0.8 else np.pi
    for _ in range(15):
        f = e_anom - e * np.sin(e_anom) - m_anom
        f_prime = 1.0 - e * np.cos(e_anom)
        d_e = -f / f_prime
        e_anom += d_e
        if abs(d_e) < 1e-12:
            break

    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(e_anom / 2.0),
        np.sqrt(1.0 - e) * np.cos(e_anom / 2.0)
    )
    r_mag = a * (1.0 - e * np.cos(e_anom))

    # Perifocal coordinates
    r_pf = np.array([r_mag * np.cos(nu), r_mag * np.sin(nu), 0.0])
    p_orb = a * (1.0 - e ** 2)
    h = np.sqrt(MU_SUN * p_orb)
    v_pf = np.array([
        -(MU_SUN / h) * np.sin(nu),
        (MU_SUN / h) * (e + np.cos(nu)),
        0.0
    ])

    # Direct 3-1-3 Euler Direction Cosine Matrix (Perifocal -> Ecliptic)
    p_vec = np.array([
        np.cos(raan) * np.cos(omega) - np.sin(raan) * np.sin(omega) * np.cos(inc),
        np.sin(raan) * np.cos(omega) + np.cos(raan) * np.sin(omega) * np.cos(inc),
        np.sin(omega) * np.sin(inc)
    ])

    q_vec = np.array([
        -np.cos(raan) * np.sin(omega) - np.sin(raan) * np.cos(omega) * np.cos(inc),
        -np.sin(raan) * np.sin(omega) + np.cos(raan) * np.cos(omega) * np.cos(inc),
        np.cos(omega) * np.sin(inc)
    ])

    w_vec = np.cross(p_vec, q_vec)

    r_pf2ecl = np.column_stack([p_vec, q_vec, w_vec])

    r_ecl = r_pf2ecl @ r_pf
    v_ecl = r_pf2ecl @ v_pf

    if frame.lower() == 'equatorial':
        r_eq = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(OBLIQUITY_J2000), -np.sin(OBLIQUITY_J2000)],
            [0.0, np.sin(OBLIQUITY_J2000), np.cos(OBLIQUITY_J2000)]
        ])
        return r_eq @ r_ecl, r_eq @ v_ecl

    return r_ecl, v_ecl


def get_soi_radius(planet_name: str) -> float:
    """Computes Laplace Sphere of Influence radius (km)."""
    p = PLANET_DATA[planet_name.lower()]
    return float(p['soi_a'] * (p['mass_ratio']) ** 0.4)