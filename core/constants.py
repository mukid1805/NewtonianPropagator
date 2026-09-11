r"""Universal physical, planetary, and orbital constants (SI Units).

## Fundamental & Reference Constants

| Constant | Value | Units | Description |
| :--- | :--- | :--- | :--- |
| `G` | `6.67430e-11` | $\text{m}^3 / (\text{kg} \cdot \text{s}^2)$ | CODATA 2018 Newtonian gravitational constant |
| `G0` | `9.80665` | $\text{m} / \text{s}^2$ | Standard Earth gravitational acceleration |
| `AU` | `149597870700.0` | $\text{m}$ | Astronomical Unit (IAU 2012 standard) |

## Planetary & Gravitational Parameters ($\mu = GM$)

| Constant | Value | Units | Description |
| :--- | :--- | :--- | :--- |
| `G_EARTH` | `3.986004418e14` | $\text{m}^3 / \text{s}^2$ | Earth standard gravitational parameter ($\mu_\oplus$) |
| `R_EARTH` | `6378137.0` | $\text{m}$ | WGS-84 Earth equatorial radius ($R_\oplus$) |
| `OMEGA_EARTH` | `7.2921159e-5` | $\text{rad} / \text{s}$ | Earth nominal rotation rate ($\omega_\oplus$) |
| `G_MOON` | `4.9048695e12` | $\text{m}^3 / \text{s}^2$ | Lunar standard gravitational parameter ($\mu_{\text{Moon}}$) |
| `R_MOON` | `1737400.0` | $\text{m}$ | Mean volumetric radius of the Moon |
| `R_MOON_ORBIT` | `3.844e8` | $\text{m}$ | Earth-Moon mean semi-major axis |
| `OMEGA_MOON` | `2.6617e-6` | $\text{rad} / \text{s}$ | Lunar mean orbital motion ($\omega_{\text{Moon}}$) |
| `G_SUN` | `1.32712440018e20` | $\text{m}^3 / \text{s}^2$ | Solar gravitational parameter ($\mu_\odot$) |
| `G_MARS` | `4.282837e13` | $\text{m}^3 / \text{s}^2$ | Mars gravitational parameter ($\mu_{\text{Mars}}$) |
| `R_MARS` | `3396190.0` | $\text{m}$ | Mars equatorial radius |
| `G_VENUS` | `3.24859e14` | $\text{m}^3 / \text{s}^2$ | Venus gravitational parameter ($\mu_{\text{Venus}}$) |
| `R_VENUS` | `6051800.0` | $\text{m}$ | Venus mean planetary radius |
| `G_JUPITER` | `1.26686534e17` | $\text{m}^3 / \text{s}^2$ | Jupiter gravitational parameter ($\mu_{\text{Jupiter}}$) |
| `R_JUPITER` | `71492000.0` | $\text{m}$ | Jupiter equatorial radius |

## Geopotential Zonal Harmonics (EGM96 / WGS-84)

| Constant | Value | Units | Description |
| :--- | :--- | :--- | :--- |
| `J2_EARTH` | `1.08262668e-3` | — | Unnormalized second zonal harmonic (oblateness) |
| `J3_EARTH` | `-2.53265649e-6` | — | Unnormalized third zonal harmonic (pear shape) |
| `J4_EARTH` | `-1.61962160e-6` | — | Unnormalized fourth zonal harmonic |

## Atmosphere, Radiation & Heliocentric Orbits

| Constant | Value | Units | Description |
| :--- | :--- | :--- | :--- |
| `RHO_0` | `1.225` | $\text{kg} / \text{m}^3$ | Sea-level atmospheric reference density ($\rho_0$) |
| `SCALE_HEIGHT` | `8500.0` | $\text{m}$ | Atmospheric density scale height ($H$) |
| `P_SUN_1AU` | `4.56e-6` | $\text{N} / \text{m}^2$ | Solar radiation pressure flux at 1 AU |
| `R_ORBIT_EARTH` | `1.000 * AU` | $\text{m}$ | Mean orbital radius of Earth |
| `R_ORBIT_MARS` | `1.524 * AU` | $\text{m}$ | Mean orbital radius of Mars |
| `R_ORBIT_VENUS` | `0.723 * AU` | $\text{m}$ | Mean orbital radius of Venus |
| `R_ORBIT_JUPITER` | `5.204 * AU` | $\text{m}$ | Mean orbital radius of Jupiter |
"""

from typing import Final

# Fundamental Universal Physical Constants
G: Final[float] = 6.67430e-11
G0: Final[float] = 9.80665
AU: Final[float] = 149_597_870_700.0

# Earth Gravitational & Physical Parameters
G_EARTH: Final[float] = 3.986004418e14
R_EARTH: Final[float] = 6_378_137.0
OMEGA_EARTH: Final[float] = 7.2921159e-5

# Geopotential Zonal Harmonics (EGM96 / WGS-84)
J2_EARTH: Final[float] = 1.08262668e-3
J3_EARTH: Final[float] = -2.53265649e-6
J4_EARTH: Final[float] = -1.61962160e-6

# Atmospheric Exponential Model Parameters
RHO_0: Final[float] = 1.225
SCALE_HEIGHT: Final[float] = 8500.0

# Lunar Gravitational & Orbit Parameters
G_MOON: Final[float] = 4.9048695e12
R_MOON: Final[float] = 1_737_400.0
R_MOON_ORBIT: Final[float] = 384_400_000.0
OMEGA_MOON: Final[float] = 2.6617e-6

# Solar & Radiation Pressure Parameters
G_SUN: Final[float] = 1.32712440018e20
P_SUN_1AU: Final[float] = 4.56e-6

# Planetary Gravitational Parameters & Physical Radii
G_MARS: Final[float] = 4.282837e13
R_MARS: Final[float] = 3_396_190.0

G_VENUS: Final[float] = 3.24859e14
R_VENUS: Final[float] = 6_051_800.0

G_JUPITER: Final[float] = 1.26686534e17
R_JUPITER: Final[float] = 71_492_000.0

# Approximate Mean Heliocentric Distances
R_ORBIT_EARTH: Final[float] = 1.000 * AU
R_ORBIT_MARS: Final[float] = 1.524 * AU
R_ORBIT_VENUS: Final[float] = 0.723 * AU
R_ORBIT_JUPITER: Final[float] = 5.204 * AU