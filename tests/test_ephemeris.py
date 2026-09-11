"""
Unit tests for planetary ephemerides and heliocentric state coordinates.
"""

import unittest
import numpy as np

from core.ephemeris import get_planet_state, get_soi_radius


class TestEphemeris(unittest.TestCase):
    """Verifies heliocentric state projections and Sphere of Influence calculations."""

    # AU in kilometers for ephemeris checks
    AU_KM = 149_597_870.7

    def test_j2000_epoch_earth_distance_near_1au(self):
        """At J2000 epoch (MJD 51544.5), Earth distance from Sun must be ~1.0 AU (0.98 - 1.02 AU)."""
        mjd_j2000 = 51544.5
        r_earth, v_earth = get_planet_state("earth", mjd_j2000)

        # r_earth is in km
        dist_au = float(np.linalg.norm(r_earth) / self.AU_KM)
        self.assertGreater(dist_au, 0.98)
        self.assertLess(dist_au, 1.02)

        # Earth mean orbital speed in km/s should be ~29.78 km/s
        speed_kms = float(np.linalg.norm(v_earth))
        self.assertGreater(speed_kms, 28.5)
        self.assertLess(speed_kms, 31.0)

    def test_mars_orbital_distance_range(self):
        """Mars distance from the Sun must remain bounded between perihelion and aphelion."""
        mjd_samples = np.linspace(51544.5, 51544.5 + 730.0, 15)

        for mjd in mjd_samples:
            r_mars, _ = get_planet_state("mars", mjd)
            dist_au = float(np.linalg.norm(r_mars) / self.AU_KM)
            self.assertGreater(dist_au, 1.35)
            self.assertLess(dist_au, 1.70)

    def test_sphere_of_influence_radius_bounds(self):
        r"""Laplace SOI radius r_soi = a * (m / M_sun)^(2/5) in km must be physically realistic."""
        # Earth SOI is ~925,000 km
        r_soi_earth = get_soi_radius("earth")
        self.assertGreater(r_soi_earth, 8.0e5)
        self.assertLess(r_soi_earth, 1.1e6)

        # Mars SOI is ~577,000 km
        r_soi_mars = get_soi_radius("mars")
        self.assertGreater(r_soi_mars, 4.5e5)
        self.assertLess(r_soi_mars, 7.0e5)

    def test_unknown_planet_raises_keyerror(self):
        """Requesting an invalid planet name should raise KeyError."""
        with self.assertRaises(KeyError):
            get_planet_state("pluto_x", 51544.5)


if __name__ == "__main__":
    unittest.main()