"""
Universal physical, planetary, and orbital constants (SI Units).
"""

# Earth Gravitational & Physical Parameters
G_EARTH = 3.986004418e14       # Gravitational parameter (m^3 / s^2)
R_EARTH = 6_378_137.0          # WGS-84 equatorial radius (m)
OMEGA_EARTH = 7.2921159e-5     # Rotation rate (rad/s)

# Geopotential Zonal Harmonics (EGM96 / WGS-84)
J2_EARTH = 1.08262668e-3       # Oblateness
J3_EARTH = -2.53265649e-6      # Pear-shape asymmetry
J4_EARTH = -1.61962160e-6      # Second-order oblateness

# Atmosphere Parameters (Exponential Model)
RHO_0 = 1.225                  # Sea-level atmospheric density (kg/m^3)
SCALE_HEIGHT = 8500.0          # Atmospheric scale height (m)

# Third-Body Lunar Parameters
G_MOON = 4.9048695e12          # Lunar gravitational parameter (m^3 / s^2)
R_MOON_ORBIT = 384_400_000.0   # Earth-Moon semi-major axis (m)
OMEGA_MOON = 2.6617e-6         # Lunar mean motion (rad/s)

# Solar & Radiation Pressure Parameters
AU = 149_597_870_700.0         # Astronomical Unit (m)
P_SUN_1AU = 4.56e-6            # Solar radiation pressure flux at 1 AU (N/m^2)

# Propulsion Reference Constants
G0 = 9.80665                   # Standard gravity acceleration (m/s^2)