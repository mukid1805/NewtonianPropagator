"""
tests/test_lambert.py
---------------------
Unit test suite verifying Lambert BVP solver accuracy.
"""
import unittest
import numpy as np

from core.ephemeris import MU_SUN, AU
from core.lambert import solve_lambert
from core.integrators import rk4_step


class TestLambertSolver(unittest.TestCase):

    def test_earth_mars_transfer_arc(self):
        """Validates Lambert velocity vectors against RK4 numerical propagation."""
        # Use consistent km-based units (1 AU ~ 1.496e8 km)
        r1 = np.array([1.0 * AU, 0.0, 0.0])
        theta = np.radians(120.0)
        r2 = np.array([1.524 * AU * np.cos(theta), 1.524 * AU * np.sin(theta), 0.0])

        # Approximate 200-day transfer
        tof = 200.0 * 86400.0

        # Solve boundary value problem
        v1_sol, v2_sol = solve_lambert(r1, r2, tof, MU_SUN, prograde=True)

        # Propagate forward using RK4 to verify arrival at r2
        dt = 1800.0  # 30-minute steps
        num_steps = int(np.ceil(tof / dt))
        state = np.concatenate([r1, v1_sol])
        t = 0.0

        def solar_gravity_derivs(time, s):
            r = s[0:3]
            v = s[3:6]
            a = -(MU_SUN / (np.linalg.norm(r)**3)) * r
            return np.concatenate([v, a])

        for _ in range(num_steps):
            h = min(dt, tof - t)
            state = rk4_step(solar_gravity_derivs, t, state, h)
            t += h

        r_final = state[0:3]
        v_final = state[3:6]

        # Terminal position miss distance must be < 1 km over 200 days
        miss_dist = np.linalg.norm(r_final - r2)
        self.assertLess(miss_dist, 1.0)

        # Terminal velocity must match Lambert solution exactly
        v_err = np.linalg.norm(v_final - v2_sol)
        self.assertLess(v_err, 1e-4)


if __name__ == '__main__':
    unittest.main()