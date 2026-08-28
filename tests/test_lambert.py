"""
tests/test_lambert.py
---------------------
Numerical validation of the Lambert targeter against analytical Keplerian transfers.
"""

import unittest
import numpy as np
from core.constants import G_SUN, AU, R_ORBIT_EARTH, R_ORBIT_MARS
from core.lambert import solve_lambert


class TestLambertSolver(unittest.TestCase):
    def test_earth_mars_hohmann_transfer(self):
        """Validates Lambert velocities against a 180-degree Hohmann transfer to Mars."""
        r1 = np.array([R_ORBIT_EARTH, 0.0, 0.0])
        r2 = np.array([-R_ORBIT_MARS, 0.0, 0.0])

        # Semi-major axis and theoretical Hohmann TOF
        a_trans = 0.5 * (R_ORBIT_EARTH + R_ORBIT_MARS)
        tof_hohmann = np.pi * np.sqrt(a_trans**3 / G_SUN)

        # Expected initial and final speeds from vis-viva equation
        v1_expected_mag = np.sqrt(G_SUN * (2.0 / R_ORBIT_EARTH - 1.0 / a_trans))
        v2_expected_mag = np.sqrt(G_SUN * (2.0 / R_ORBIT_MARS - 1.0 / a_trans))

        # Solve via Lambert
        v1_sol, v2_sol = solve_lambert(r1, r2, tof=tof_hohmann, mu=G_SUN)

        # Assert velocity magnitudes match vis-viva within 0.01%
        self.assertAlmostEqual(np.linalg.norm(v1_sol), v1_expected_mag, delta=v1_expected_mag * 1e-4)
        self.assertAlmostEqual(np.linalg.norm(v2_sol), v2_expected_mag, delta=v2_expected_mag * 1e-4)

        # Assert correct direction (transverse tangential burn)
        self.assertGreater(v1_sol[1], 0.0)
        self.assertLess(v2_sol[1], 0.0)


if __name__ == "__main__":
    unittest.main()