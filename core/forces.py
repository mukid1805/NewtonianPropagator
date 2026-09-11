r"""
Acceleration models for the Newtonian superposition summation junction.

All acceleration models operate in the Earth-Centered Inertial (ECI) coordinate
frame with position vectors $\mathbf{r} \in \mathbb{R}^3\text{ [m]}$ and velocity
vectors $\mathbf{v} \in \mathbb{R}^3\text{ [m/s]}$.
"""

from typing import Dict
import numpy as np
from core.constants import (
    AU,
    G_EARTH,
    G_MOON,
    J2_EARTH,
    J3_EARTH,
    J4_EARTH,
    OMEGA_EARTH,
    OMEGA_MOON,
    P_SUN_1AU,
    R_EARTH,
    R_MOON_ORBIT,
    RHO_0,
    SCALE_HEIGHT,
)


def accel_earth_gravity(r: np.ndarray) -> np.ndarray:
    r"""Compute primary central-body Newtonian point-mass gravitational acceleration.

    $$
    \mathbf{a}_{\text{grav}} = -\frac{\mu_{\oplus}}{\|\mathbf{r}\|^3} \mathbf{r}
    $$

    Args:
        r: Spacecraft position vector $\mathbf{r}$ in the ECI frame [$\text{m}$].

    Returns:
        np.ndarray: Gravitational acceleration vector $\mathbf{a}_{\text{grav}}$ [$\text{m/s}^2$].
    """
    r_mag = np.linalg.norm(r)
    return -G_EARTH * r / (r_mag ** 3)


def accel_j2_perturbation(r: np.ndarray) -> np.ndarray:
    r"""Compute Earth oblateness ($J_2$ zonal harmonic) gravitational perturbation.

    $$
    \mathbf{a}_{J_2} = \frac{3 J_2 \mu_{\oplus} R_{\oplus}^2}{2 \|\mathbf{r}\|^5}
    \begin{bmatrix}
    x \left(5 \frac{z^2}{\|\mathbf{r}\|^2} - 1\right) \\
    y \left(5 \frac{z^2}{\|\mathbf{r}\|^2} - 1\right) \\
    z \left(5 \frac{z^2}{\|\mathbf{r}\|^2} - 3\right)
    \end{bmatrix}
    $$

    Args:
        r: Spacecraft position vector $\mathbf{r} = [x, y, z]^T$ in the ECI frame [$\text{m}$].

    Returns:
        np.ndarray: Perturbation acceleration vector $\mathbf{a}_{J_2}$ [$\text{m/s}^2$].
    """
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
    r"""Compute Earth pear-shape asymmetry ($J_3$ zonal harmonic) perturbation.

    With $s = \sin\phi = \frac{z}{\|\mathbf{r}\|}$:

    $$
    \mathbf{a}_{J_3} = \frac{1}{2} \frac{J_3 \mu_{\oplus} R_{\oplus}^3}{\|\mathbf{r}\|^5}
    \begin{bmatrix}
    5 x \left(7s^3 - 3s\right) \\
    5 y \left(7s^3 - 3s\right) \\
    \|\mathbf{r}\| \left(35s^4 - 30s^2 + 3\right)
    \end{bmatrix}
    $$

    Args:
        r: Spacecraft position vector $\mathbf{r} = [x, y, z]^T$ in the ECI frame [$\text{m}$].

    Returns:
        np.ndarray: Perturbation acceleration vector $\mathbf{a}_{J_3}$ [$\text{m/s}^2$].
    """
    x, y, z = r
    r_mag = np.linalg.norm(r)
    s = z / r_mag

    factor = 0.5 * J3_EARTH * G_EARTH * (R_EARTH ** 3) / (r_mag ** 5)

    ax = 5.0 * factor * x * (7.0 * (s ** 3) - 3.0 * s)
    ay = 5.0 * factor * y * (7.0 * (s ** 3) - 3.0 * s)
    az = factor * (35.0 * (s ** 4) - 30.0 * (s ** 2) + 3.0) * r_mag

    return np.array([ax, ay, az])


def accel_j4_perturbation(r: np.ndarray) -> np.ndarray:
    r"""Compute Earth second-order oblateness ($J_4$ zonal harmonic) perturbation.

    With $s = \sin\phi = \frac{z}{\|\mathbf{r}\|}$:

    $$
    \mathbf{a}_{J_4} = \frac{5}{8} \frac{J_4 \mu_{\oplus} R_{\oplus}^4}{\|\mathbf{r}\|^6}
    \begin{bmatrix}
    x \left(3 - 42s^2 + 63s^4\right) \\
    y \left(3 - 42s^2 + 63s^4\right) \\
    z \left(15 - 70s^2 + 63s^4\right)
    \end{bmatrix}
    $$

    Args:
        r: Spacecraft position vector $\mathbf{r} = [x, y, z]^T$ in the ECI frame [$\text{m}$].

    Returns:
        np.ndarray: Perturbation acceleration vector $\mathbf{a}_{J_4}$ [$\text{m/s}^2$].
    """
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


def rv_to_keplerian(r: np.ndarray, v: np.ndarray, mu: float = G_EARTH) -> Dict[str, float]:
    r"""Convert Cartesian state vectors $(\mathbf{r}, \mathbf{v})$ to Classical Orbital Elements (COE).

    Computes specific orbital energy $\varepsilon$, specific angular momentum vector $\mathbf{h}$,
    and eccentricity vector $\mathbf{e}$:

    $$
    \varepsilon = \frac{\|\mathbf{v}\|^2}{2} - \frac{\mu}{\|\mathbf{r}\|}, \quad
    a = -\frac{\mu}{2\varepsilon}, \quad
    \mathbf{h} = \mathbf{r} \times \mathbf{v}, \quad
    \mathbf{e} = \frac{1}{\mu} \left[\left(\|\mathbf{v}\|^2 - \frac{\mu}{\|\mathbf{r}\|}\right)\mathbf{r} - (\mathbf{r} \cdot \mathbf{v})\mathbf{v}\right]
    $$

    Args:
        r: Position vector $\mathbf{r}$ in the ECI frame [$\text{m}$].
        v: Velocity vector $\mathbf{v}$ in the ECI frame [$\text{m/s}$].
        mu: Central body gravitational parameter [$\text{m}^3/\text{s}^2$]. Defaults to Earth ($G_{\text{EARTH}}$).

    Returns:
        Dict[str, float]: Classical Keplerian orbital elements containing:
            * `a`: Semi-major axis $a$ [$\text{m}$].
            * `e`: Eccentricity $e$ [dimensionless].
            * `inc_deg`: Inclination $i$ [$\text{deg}$].
            * `raan_deg`: Right Ascension of the Ascending Node $\Omega$ [$\text{deg}$].
            * `argp_deg`: Argument of Perigee $\omega$ [$\text{deg}$].
    """
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    h_vec = np.cross(r, v)
    h_mag = np.linalg.norm(h_vec)

    energy = 0.5 * (v_mag ** 2) - (mu / r_mag)
    a = -mu / (2.0 * energy) if abs(energy) > 1e-12 else np.nan

    e_vec = (1.0 / mu) * ((v_mag ** 2 - mu / r_mag) * r - np.dot(r, v) * v)
    e = np.linalg.norm(e_vec)

    inc = np.arccos(np.clip(h_vec[2] / h_mag, -1.0, 1.0))

    k_hat = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(k_hat, h_vec)
    n_mag = np.linalg.norm(n_vec)

    if n_mag > 1e-9:
        raan = np.arccos(np.clip(n_vec[0] / n_mag, -1.0, 1.0))
        if n_vec[1] < 0:
            raan = 2.0 * np.pi - raan
    else:
        raan = 0.0

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
    r"""Compute third-body lunar gravitational perturbation.

    Accounts for both direct and indirect inertial acceleration:

    $$
    \mathbf{a}_{3\text{rd}} = \mu_{\text{moon}} \left( \frac{\mathbf{r}_{\text{moon}} - \mathbf{r}}{\|\mathbf{r}_{\text{moon}} - \mathbf{r}\|^3} - \frac{\mathbf{r}_{\text{moon}}}{\|\mathbf{r}_{\text{moon}}\|^3} \right)
    $$

    Args:
        r: Spacecraft position vector $\mathbf{r}$ in the ECI frame [$\text{m}$].
        t: Elapsed simulation epoch time $t$ [$\text{s}$].

    Returns:
        np.ndarray: Third-body lunar acceleration vector $\mathbf{a}_{3\text{rd}}$ [$\text{m/s}^2$].
    """
    r_moon = np.array([
        R_MOON_ORBIT * np.cos(OMEGA_MOON * t),
        R_MOON_ORBIT * np.sin(OMEGA_MOON * t),
        0.0
    ])
    r_rel = r_moon - r
    return G_MOON * (r_rel / (np.linalg.norm(r_rel) ** 3) - r_moon / (R_MOON_ORBIT ** 3))


def accel_atmospheric_drag(
    r: np.ndarray,
    v: np.ndarray,
    cd: float,
    area: float,
    mass: float
) -> np.ndarray:
    r"""Compute atmospheric drag acceleration with exponential density falloff.

    Accounts for Earth diurnal rotation in calculating relative wind velocity:

    $$
    \rho(h) = \rho_0 \exp\left(-\frac{h}{H}\right), \quad
    \mathbf{v}_{\text{rel}} = \mathbf{v} - \mathbf{\omega}_{\oplus} \times \mathbf{r}
    $$

    $$
    \mathbf{a}_{\text{drag}} = -\frac{1}{2} \rho \left( \frac{C_d A}{m} \right) \|\mathbf{v}_{\text{rel}}\| \mathbf{v}_{\text{rel}}
    $$

    Args:
        r: Spacecraft position vector $\mathbf{r}$ in the ECI frame [$\text{m}$].
        v: Spacecraft inertial velocity vector $\mathbf{v}$ in the ECI frame [$\text{m/s}$].
        cd: Aerodynamic drag coefficient $C_d$ [dimensionless].
        area: Frontal aerodynamic reference cross-sectional area $A$ [$\text{m}^2$].
        mass: Instantaneous spacecraft mass $m$ [$\text{kg}$].

    Returns:
        np.ndarray: Aerodynamic drag acceleration vector $\mathbf{a}_{\text{drag}}$ [$\text{m/s}^2$].
    """
    altitude = np.linalg.norm(r) - R_EARTH
    if altitude < 0:
        return np.array([0.0, 0.0, 0.0])

    rho = RHO_0 * np.exp(-altitude / SCALE_HEIGHT)
    omega_vec = np.array([0.0, 0.0, OMEGA_EARTH])
    v_rel = v - np.cross(omega_vec, r)
    v_rel_mag = np.linalg.norm(v_rel)

    return -0.5 * rho * (cd * area / mass) * v_rel_mag * v_rel


def accel_solar_radiation_pressure(
    r: np.ndarray,
    cr: float,
    area: float,
    mass: float
) -> np.ndarray:
    r"""Compute cannonball Solar Radiation Pressure (SRP) with cylindrical Earth eclipse shadow.

    $$
    \mathbf{a}_{\text{srp}} = \nu P_{\text{sun}} C_r \left(\frac{A}{m}\right) \frac{\mathbf{r}_{\text{sc}} - \mathbf{r}_{\odot}}{\|\mathbf{r}_{\text{sc}} - \mathbf{r}_{\odot}\|}
    $$

    where $\nu = 0$ inside the cylindrical umbra ($x < 0$ and $\sqrt{y^2 + z^2} < R_{\oplus}$), and $\nu = 1$ in direct sunlight.

    Args:
        r: Spacecraft position vector $\mathbf{r}$ in the ECI frame [$\text{m}$].
        cr: Radiation pressure reflectivity coefficient $C_r$ [dimensionless].
        area: Solar exposed cross-sectional area $A$ [$\text{m}^2$].
        mass: Instantaneous spacecraft mass $m$ [$\text{kg}$].

    Returns:
        np.ndarray: Solar radiation pressure acceleration vector $\mathbf{a}_{\text{srp}}$ [$\text{m/s}^2$].
    """
    r_sun = np.array([AU, 0.0, 0.0])
    r_sc_sun = r - r_sun
    sun_dir = r_sc_sun / np.linalg.norm(r_sc_sun)

    if r[0] < 0 and np.linalg.norm(r[1:3]) < R_EARTH:
        return np.array([0.0, 0.0, 0.0])

    return P_SUN_1AU * cr * (area / mass) * sun_dir


def accel_fixed_thrust(
    t: float,
    start_t: float,
    duration: float,
    thrust_vec: np.ndarray,
    mass: float
) -> np.ndarray:
    r"""Compute constant directional thrust acceleration over an active window.

    $$
    \mathbf{a}_{\text{thrust}} =
    \begin{cases}
    \frac{\mathbf{F}_{\text{thrust}}}{m}, & t_{\text{start}} \le t \le t_{\text{start}} + \Delta t \\
    \mathbf{0}, & \text{otherwise}
    \end{cases}
    $$

    Args:
        t: Current simulation epoch time $t$ [$\text{s}$].
        start_t: Engine ignition epoch $t_{\text{start}}$ [$\text{s}$].
        duration: Total engine burn duration $\Delta t$ [$\text{s}$].
        thrust_vec: Applied thrust vector $\mathbf{F}_{\text{thrust}}$ [$\text{N}$].
        mass: Instantaneous spacecraft mass $m$ [$\text{kg}$].

    Returns:
        np.ndarray: Thrust acceleration vector $\mathbf{a}_{\text{thrust}}$ [$\text{m/s}^2$].
    """
    if start_t <= t <= (start_t + duration):
        return thrust_vec / mass
    return np.array([0.0, 0.0, 0.0])


def accel_electric_prograde(
    v: np.ndarray,
    thrust_mag: float,
    mass: float
) -> np.ndarray:
    r"""Compute continuous low-thrust propulsion aligned with the velocity vector.

    $$
    \mathbf{a}_{\text{low-thrust}} = \frac{T}{m} \hat{\mathbf{v}} = \frac{T}{m} \left( \frac{\mathbf{v}}{\|\mathbf{v}\|} \right)
    $$

    Args:
        v: Spacecraft inertial velocity vector $\mathbf{v}$ in the ECI frame [$\text{m/s}$].
        thrust_mag: Continuous thrust magnitude $T$ [$\text{N}$].
        mass: Instantaneous spacecraft mass $m$ [$\text{kg}$].

    Returns:
        np.ndarray: Prograde low-thrust acceleration vector $\mathbf{a}_{\text{low-thrust}}$ [$\text{m/s}^2$].
    """
    v_mag = np.linalg.norm(v)
    if v_mag == 0.0:
        return np.array([0.0, 0.0, 0.0])
    return (thrust_mag / mass) * (v / v_mag)


def eci_to_lvlh(
    r_chief: np.ndarray,
    v_chief: np.ndarray,
    r_deputy: np.ndarray
) -> np.ndarray:
    r"""Transform deputy position relative to chief from ECI to Local-Vertical/Local-Horizontal (LVLH).

    Constructs the standard orbital reference triad:

    $$
    \hat{\mathbf{e}}_r = \frac{\mathbf{r}_c}{\|\mathbf{r}_c\|}, \quad
    \hat{\mathbf{e}}_h = \frac{\mathbf{r}_c \times \mathbf{v}_c}{\|\mathbf{r}_c \times \mathbf{v}_c\|}, \quad
    \hat{\mathbf{e}}_\theta = \hat{\mathbf{e}}_h \times \hat{\mathbf{e}}_r
    $$

    and maps relative displacement $\delta \mathbf{r}_{\text{ECI}} = \mathbf{r}_d - \mathbf{r}_c$:

    $$
    \delta \mathbf{r}_{\text{LVLH}} =
    \begin{bmatrix}
    \hat{\mathbf{e}}_r^T \\
    \hat{\mathbf{e}}_\theta^T \\
    \hat{\mathbf{e}}_h^T
    \end{bmatrix}
    \delta \mathbf{r}_{\text{ECI}} =
    \begin{bmatrix}
    x_{\text{radial}} \\
    y_{\text{along-track}} \\
    z_{\text{cross-track}}
    \end{bmatrix}
    $$

    Args:
        r_chief: Chief position vector $\mathbf{r}_c$ in the ECI frame [$\text{m}$].
        v_chief: Chief velocity vector $\mathbf{v}_c$ in the ECI frame [$\text{m/s}$].
        r_deputy: Deputy position vector $\mathbf{r}_d$ in the ECI frame [$\text{m}$].

    Returns:
        np.ndarray: Relative displacement $\delta \mathbf{r}_{\text{LVLH}} = [x_{\text{radial}}, y_{\text{along-track}}, z_{\text{cross-track}}]^T$ [$\text{m}$].
    """
    r_c_mag = np.linalg.norm(r_chief)
    e_r = r_chief / r_c_mag

    h_vec = np.cross(r_chief, v_chief)
    e_h = h_vec / np.linalg.norm(h_vec)

    e_theta = np.cross(e_h, e_r)

    r_eci_to_lvlh = np.vstack([e_r, e_theta, e_h])
    delta_r_eci = r_deputy - r_chief

    return r_eci_to_lvlh @ delta_r_eci