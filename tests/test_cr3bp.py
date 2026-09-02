"""
Unit tests for core.cr3bp dynamics, equilibrium geometry, and energy conservation.
Validates L1-L5 Lagrange equilibrium coordinates, equations of motion zero-residuals,
Jacobi constant conservation under propagation, and synodic-to-inertial conversions.
"""

import unittest
import numpy as np

from core.cr3bp import (
    cr3bp_derivatives,
    compute_lagrange_points,
    compute_jacobi_constant,
    synodic_to_inertial,
    MU_EARTH_MOON,
    L_STAR,
)
from core.integrators import rk45_adaptive


class TestCR3BP(unittest.TestCase):
    def setUp(self):
        self.mu = MU_EARTH_MOON
        self.pts = compute_lagrange_points(self.mu)

    def test_triangular_points_analytic_geometry(self):
        """
        L4 and L5 must analytically form exact equilateral triangles with the primaries:
        x = 0.5 - mu
        y = +sqrt(3)/2 (L4) and -sqrt(3)/2 (L5)
        z = 0.0
        """
        expected_x = 0.5 - self.mu
        expected_y_l4 = np.sqrt(3.0) / 2.0
        expected_y_l5 = -np.sqrt(3.0) / 2.0

        l4 = self.pts['L4']
        l5 = self.pts['L5']

        self.assertAlmostEqual(l4[0], expected_x, places=7)
        self.assertAlmostEqual(l4[1], expected_y_l4, places=7)
        self.assertAlmostEqual(l4[2], 0.0, places=7)

        self.assertAlmostEqual(l5[0], expected_x, places=7)
        self.assertAlmostEqual(l5[1], expected_y_l5, places=7)
        self.assertAlmostEqual(l5[2], 0.0, places=7)

    def test_collinear_points_spatial_ordering(self):
        """
        Collinear points along the synodic x-axis must follow the canonical sequence:
        L3 < -mu (Primary 1 / Earth) < L1 < (1 - mu, Primary 2 / Moon) < L2.
        All must lie precisely on the synodic y = 0, z = 0 plane.
        """
        x_earth = -self.mu
        x_moon = 1.0 - self.mu

        x_l1 = self.pts['L1'][0]
        x_l2 = self.pts['L2'][0]
        x_l3 = self.pts['L3'][0]

        # Verify y and z planar alignment
        for name in ['L1', 'L2', 'L3']:
            self.assertAlmostEqual(self.pts[name][1], 0.0, places=7)
            self.assertAlmostEqual(self.pts[name][2], 0.0, places=7)

        # Sequence ordering along synodic line of centers
        self.assertLess(x_l3, x_earth)
        self.assertGreater(x_l1, x_earth)
        self.assertLess(x_l1, x_moon)
        self.assertGreater(x_l2, x_moon)

    def test_equations_of_motion_zero_at_lagrange_points(self):
        """
        At every equilibrium point with zero velocity (vx=vy=vz=0),
        the effective gravitational and centrifugal accelerations must cancel (ax=ay=az=0).
        """
        for name, coord in self.pts.items():
            state_stationary = np.array([coord[0], coord[1], coord[2], 0.0, 0.0, 0.0], dtype=np.float64)
            derivs = cr3bp_derivatives(0.0, state_stationary, self.mu)

            # Velocities (derivs[0:3]) are zero
            np.testing.assert_allclose(derivs[0:3], np.zeros(3), atol=1e-12)

            # Accelerations (derivs[3:6]) must vanish at equilibrium
            np.testing.assert_allclose(
                derivs[3:6],
                np.zeros(3),
                atol=1e-5,
                err_msg=f"Acceleration non-zero at equilibrium point {name}: {derivs[3:6]}"
            )

    def test_jacobi_constant_conservation(self):
        """
        Jacobi integral of motion C_J must remain conserved along a numerical trajectory.
        Propagates a trajectory perturbed around L1 for 1.0 non-dimensional time unit (~4.35 days).
        """
        # Small perturbation offset near L1
        l1 = self.pts['L1']
        state0 = np.array([l1[0] + 0.01, 0.0, 0.0, 0.0, 0.02, 0.0], dtype=np.float64)
        c0 = compute_jacobi_constant(state0, self.mu)

        def eom(t, state):
            return cr3bp_derivatives(t, state, self.mu)

        times, states = rk45_adaptive(
            derivs_func=eom,
            t_span=(0.0, 1.0),
            y0=state0,
            rtol=1e-9,
            atol=1e-11,
            h_init=1e-3,
        )

        c_history = np.array([compute_jacobi_constant(s, self.mu) for s in states])
        max_jacobi_drift = np.max(np.abs(c_history - c0))

        self.assertLess(
            max_jacobi_drift,
            1e-7,
            f"Jacobi constant drift {max_jacobi_drift:.2e} exceeded tolerance 1e-7."
        )

    def test_synodic_to_inertial_scaling(self):
        """
        Dimensional transformation must correctly convert non-dimensional positions
        to kilometers using L_STAR.
        """
        times_nd = np.array([0.0])
        # Position 1.0 ND units along X with 0 velocity
        states_syn = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        states_eci_km = synodic_to_inertial(times_nd, states_syn, self.mu)

        expected_earth_distance_km = (1.0 + self.mu) * (L_STAR / 1000.0)
        actual_distance_km = float(np.linalg.norm(states_eci_km[0, 0:3]))

        self.assertAlmostEqual(actual_distance_km, expected_earth_distance_km, places=1)


if __name__ == '__main__':
    unittest.main()
    