"""
Unit tests for core.forces acceleration models.
Validates Earth gravity, J2-J4 harmonics, lunar perturbations,
atmospheric drag, SRP shadow occultation, and frame conversions.
"""

import unittest
import numpy as np

from core.constants import (
    G_EARTH,
    R_EARTH,
    P_SUN_1AU,
    AU,
    OMEGA_EARTH,
)
from core.forces import (
    accel_earth_gravity,
    accel_j2_perturbation,
    accel_j3_perturbation,
    accel_j4_perturbation,
    accel_lunar_gravity,
    accel_atmospheric_drag,
    accel_solar_radiation_pressure,
    accel_electric_prograde,
    rv_to_keplerian,
    eci_to_lvlh,
)


class TestForces(unittest.TestCase):
    def test_earth_gravity_magnitude(self):
        """Newtonian point-mass gravity at surface should equal standard g (~9.798 m/s^2 at equatorial radius)."""
        r_surface = np.array([R_EARTH, 0.0, 0.0])
        acc = accel_earth_gravity(r_surface)

        # Acceleration must oppose position vector
        self.assertAlmostEqual(acc[1], 0.0)
        self.assertAlmostEqual(acc[2], 0.0)
        self.assertLess(acc[0], 0.0)

        expected_g = G_EARTH / (R_EARTH ** 2)
        self.assertAlmostEqual(abs(acc[0]), expected_g, places=4)

    def test_j2_equatorial_vs_polar(self):
        """
        J2 perturbation:
        - At equator (z = 0): Extra mass bulge pulls inward (-x direction).
        - Along pole (x = y = 0, z > 0): (5*1 - 3) = +2z, perturbation points outward (+z direction).
        """
        r_eq = np.array([R_EARTH + 500e3, 0.0, 0.0])
        r_pol = np.array([0.0, 0.0, R_EARTH + 500e3])

        acc_eq = accel_j2_perturbation(r_eq)
        acc_pol = accel_j2_perturbation(r_pol)

        # At equator z = 0: (5*0 - 1) = -1, acts inward toward center (-x)
        self.assertLess(acc_eq[0], 0.0)
        self.assertAlmostEqual(acc_eq[1], 0.0)
        self.assertAlmostEqual(acc_eq[2], 0.0)

        # Along polar axis: z * (5*1 - 3) = +2z, acts in +z direction
        self.assertGreater(acc_pol[2], 0.0)
        self.assertAlmostEqual(acc_pol[0], 0.0)
        self.assertAlmostEqual(acc_pol[1], 0.0)

    def test_j3_j4_zero_at_symmetry_points(self):
        """Test symmetric boundary evaluations for higher-order harmonics."""
        r_eq = np.array([R_EARTH + 400e3, 0.0, 0.0])
        acc_j3 = accel_j3_perturbation(r_eq)
        acc_j4 = accel_j4_perturbation(r_eq)

        # At equator z=0, s=0: ax and ay should be zero for J3
        self.assertAlmostEqual(acc_j3[0], 0.0)
        self.assertAlmostEqual(acc_j3[1], 0.0)

        # J4 should produce only radial equatorial acceleration
        self.assertAlmostEqual(acc_j4[1], 0.0)
        self.assertAlmostEqual(acc_j4[2], 0.0)

    def test_drag_opposes_relative_velocity(self):
        """Atmospheric drag must strictly oppose relative air velocity."""
        r = np.array([R_EARTH + 200e3, 0.0, 0.0])
        v = np.array([0.0, 7800.0, 0.0])
        mass = 500.0
        cd = 2.2
        area = 2.0

        a_drag = accel_atmospheric_drag(r, v, cd=cd, area=area, mass=mass)

        # Relative velocity in y-direction accounting for Earth rotation
        v_rel_y = v[1] - (OMEGA_EARTH * r[0])
        self.assertLess(a_drag[1], 0.0)  # Must oppose positive prograde motion
        self.assertAlmostEqual(a_drag[0], 0.0)
        self.assertAlmostEqual(a_drag[2], 0.0)

    def test_drag_subsurface_cutoff(self):
        """Negative altitude must trigger 0 acceleration to prevent divergence on surface contact."""
        r_sub = np.array([R_EARTH - 1000.0, 0.0, 0.0])
        v = np.array([0.0, 5000.0, 0.0])
        a_drag = accel_atmospheric_drag(r_sub, v, cd=2.2, area=1.0, mass=100.0)
        np.testing.assert_array_equal(a_drag, np.zeros(3))

    def test_srp_sunlit_vs_umbra(self):
        """SRP should push away from the Sun in daylight, and vanish inside Earth's shadow."""
        cr = 1.5
        area = 10.0
        mass = 1000.0

        # Spacecraft on dayside (+x toward Sun at AU)
        r_dayside = np.array([R_EARTH + 1000e3, 0.0, 0.0])
        a_srp_day = accel_solar_radiation_pressure(r_dayside, cr=cr, area=area, mass=mass)
        # Vector points from Sun (+x) to SC: (r - r_sun) / |r - r_sun| => [-1, 0, 0]
        self.assertLess(a_srp_day[0], 0.0)
        expected_srp_mag = P_SUN_1AU * cr * (area / mass)
        self.assertAlmostEqual(np.linalg.norm(a_srp_day), expected_srp_mag, places=7)

        # Spacecraft in nightside umbra (x < 0 and radius < R_EARTH)
        r_nightside = np.array([-R_EARTH - 500e3, 100.0, 0.0])
        a_srp_umbra = accel_solar_radiation_pressure(r_nightside, cr=cr, area=area, mass=mass)
        np.testing.assert_array_equal(a_srp_umbra, np.zeros(3))

    def test_electric_prograde_alignment(self):
        """Continuous low-thrust acceleration should align with velocity vector."""
        v = np.array([0.0, 7000.0, 0.0])
        thrust_mag = 0.5
        mass = 250.0

        acc = accel_electric_prograde(v, thrust_mag=thrust_mag, mass=mass)
        expected_acc_mag = thrust_mag / mass

        self.assertAlmostEqual(acc[1], expected_acc_mag)
        self.assertAlmostEqual(acc[0], 0.0)
        self.assertAlmostEqual(acc[2], 0.0)

    def test_rv_to_keplerian_circular_orbit(self):
        """Verify orbital element conversion for a standard circular LEO."""
        alt = 400e3
        r_mag = R_EARTH + alt
        v_mag = float(np.sqrt(G_EARTH / r_mag))

        r = np.array([r_mag, 0.0, 0.0])
        v = np.array([0.0, v_mag, 0.0])

        elements = rv_to_keplerian(r, v, mu=G_EARTH)
        self.assertAlmostEqual(elements["a"], r_mag, delta=1.0)
        self.assertAlmostEqual(elements["e"], 0.0, places=4)
        self.assertAlmostEqual(elements["inc_deg"], 0.0, places=4)

    def test_eci_to_lvlh_transformation(self):
        """Verify deputy offset projection into radial and along-track components."""
        r_chief = np.array([7000e3, 0.0, 0.0])
        v_chief = np.array([0.0, 7500.0, 0.0])

        # Deputy 100 meters ahead along track (y-direction in ECI)
        r_deputy = r_chief + np.array([0.0, 100.0, 0.0])
        delta_lvlh = eci_to_lvlh(r_chief, v_chief, r_deputy)

        # In LVLH: x=radial, y=along-track, z=cross-track
        self.assertAlmostEqual(delta_lvlh[0], 0.0, places=4)
        self.assertAlmostEqual(delta_lvlh[1], 100.0, places=4)
        self.assertAlmostEqual(delta_lvlh[2], 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
