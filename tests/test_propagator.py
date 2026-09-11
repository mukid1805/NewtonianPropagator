"""
Unit tests for SpacecraftPropagator state vectors, thrust laws, and solvers.
"""

import unittest
import numpy as np

from core.constants import G0, G_EARTH, R_EARTH
from core.propagator import SpacecraftPropagator


class TestSpacecraftPropagator(unittest.TestCase):
    """Verifies SpacecraftPropagator API, 6-DOF/7-DOF dynamics, and numerical integration."""

    def setUp(self):
        # Nominal 400 km circular LEO
        self.r_mag = R_EARTH + 400e3
        self.v_mag = np.sqrt(G_EARTH / self.r_mag)
        self.r0 = np.array([self.r_mag, 0.0, 0.0], dtype=np.float64)
        self.v0 = np.array([0.0, self.v_mag, 0.0], dtype=np.float64)
        self.period = 2.0 * np.pi * np.sqrt(self.r_mag**3 / G_EARTH)

    def test_state_dimension_6dof_vs_7dof(self):
        """Propagator must return shape (N, 6) when mass tracking is disabled, and (N, 7) when enabled."""
        prop = SpacecraftPropagator(mass=750.0, use_j2=False)

        # 6-DOF standard run
        t_6dof, s_6dof = prop.propagate(self.r0, self.v0, t_span=100.0, dt=10.0, track_mass=False)
        self.assertEqual(s_6dof.shape[1], 6)
        self.assertEqual(len(t_6dof), len(s_6dof))

        # 7-DOF tracking enabled
        t_7dof, s_7dof = prop.propagate(self.r0, self.v0, t_span=100.0, dt=10.0, track_mass=True)
        self.assertEqual(s_7dof.shape[1], 7)
        self.assertEqual(len(t_7dof), len(s_7dof))
        self.assertAlmostEqual(s_7dof[0, 6], 750.0, places=6)

    def test_analytical_mass_depletion_rate(self):
        r"""Continuous thrust must deplete mass precisely per m_dot = -T / (Isp * g0)."""
        thrust_mag = 50.0  # N
        isp = 2500.0  # s
        initial_mass = 1200.0  # kg
        t_span = 3600.0  # 1 hour
        dt = 10.0

        prop = SpacecraftPropagator(mass=initial_mass, isp=isp, use_j2=False)
        prop.configure_electric_burn(thrust_magnitude=thrust_mag)

        times, states = prop.propagate(self.r0, self.v0, t_span=t_span, dt=dt, track_mass=True)

        expected_m_dot = -thrust_mag / (isp * G0)
        expected_final_mass = initial_mass + expected_m_dot * t_span
        actual_final_mass = states[-1, 6]

        # Analytical mass depletion should match within 1e-5 kg
        self.assertAlmostEqual(actual_final_mass, expected_final_mass, delta=1e-5)

    def test_fixed_burn_window_activation(self):
        """Fixed burn must remain inert before start_t and after start_t + duration."""
        prop = SpacecraftPropagator(mass=500.0, isp=300.0, use_j2=False)
        thrust_vec = np.array([100.0, 0.0, 0.0])
        start_t = 200.0
        duration = 100.0
        prop.configure_fixed_burn(start_t=start_t, duration=duration, thrust_vec=thrust_vec)

        # Pre-burn epoch
        a_pre = prop.thrust_steering_law(150.0, self.r0, self.v0, 500.0)
        np.testing.assert_allclose(a_pre, np.zeros(3), atol=1e-9)

        # Active burn epoch
        a_active = prop.thrust_steering_law(250.0, self.r0, self.v0, 500.0)
        np.testing.assert_allclose(a_active, thrust_vec / 500.0, atol=1e-9)

        # Post-burn epoch
        a_post = prop.thrust_steering_law(350.0, self.r0, self.v0, 500.0)
        np.testing.assert_allclose(a_post, np.zeros(3), atol=1e-9)

    def test_rk4_vs_rk45_unperturbed_agreement(self):
        """Adaptive RK45 and fixed RK4 must agree on energy and short-arc trajectory."""
        prop = SpacecraftPropagator(use_j2=False)
        t_test = 300.0  # 5-minute arc

        _, states_rk4 = prop.propagate(
            self.r0, self.v0, t_span=t_test, dt=1.0, method="rk4"
        )
        _, states_rk45 = prop.propagate(
            self.r0, self.v0, t_span=t_test, dt=10.0, method="rk45", rtol=1e-10, atol=1e-12
        )

        # Position difference over 5 minutes must remain under 1 meter
        diff_norm = np.linalg.norm(states_rk4[-1, :3] - states_rk45[-1, :3])
        self.assertLess(diff_norm, 1.0)

        # Specific mechanical energy must match to within 1e-7 relative error
        eps_rk4 = 0.5 * np.linalg.norm(states_rk4[-1, 3:6])**2 - G_EARTH / np.linalg.norm(states_rk4[-1, :3])
        eps_rk45 = 0.5 * np.linalg.norm(states_rk45[-1, 3:6])**2 - G_EARTH / np.linalg.norm(states_rk45[-1, :3])
        rel_energy_diff = abs((eps_rk4 - eps_rk45) / eps_rk4)
        self.assertLess(rel_energy_diff, 1e-7)


if __name__ == "__main__":
    unittest.main()