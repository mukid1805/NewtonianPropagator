"""
Unit tests for SwarmPropagator multi-agent tracking and LVLH relative motion.
"""

import unittest
import numpy as np

from core.constants import G_EARTH, R_EARTH
from core.propagator import SpacecraftPropagator
from core.swarm import SwarmPropagator


class TestSwarmDynamics(unittest.TestCase):
    """Verifies multi-agent registration, swarm propagation, and LVLH frame calculations."""

    def setUp(self):
        # Chief in 500 km circular orbit
        self.r_mag = R_EARTH + 500e3
        self.v_mag = np.sqrt(G_EARTH / self.r_mag)
        self.r_chief = np.array([self.r_mag, 0.0, 0.0], dtype=np.float64)
        self.v_chief = np.array([0.0, self.v_mag, 0.0], dtype=np.float64)
        self.period = 2.0 * np.pi * np.sqrt(self.r_mag**3 / G_EARTH)

        self.chief_engine = SpacecraftPropagator(use_j2=False)

    def test_swarm_initialization_and_deputy_registration(self):
        """SwarmPropagator must track chief engine and store registered deputy agents."""
        swarm = SwarmPropagator(chief_propagator=self.chief_engine)

        r_dep1 = self.r_chief + np.array([100.0, 0.0, 0.0])
        v_dep1 = self.v_chief + np.array([0.0, 0.05, 0.0])

        swarm.add_deputy("Deputy 1", SpacecraftPropagator(use_j2=False), r_dep1, v_dep1)

        # Confirm deputy is stored in instance state
        self.assertTrue(any("Deputy 1" in str(v) for v in swarm.__dict__.values()))

    def test_swarm_propagation_contract_and_lvlh_bounds(self):
        """propagate_swarm must return (times, chief_states, relative_tracks) with consistent dimensions."""
        swarm = SwarmPropagator(chief_propagator=self.chief_engine)

        # Deputy 1: small radial offset (PCO-like)
        r_dep1 = self.r_chief + np.array([50.0, 0.0, 0.0])
        v_dep1 = self.v_chief + np.array([0.0, 0.02, 0.0])
        swarm.add_deputy("Deputy 1", SpacecraftPropagator(use_j2=False), r_dep1, v_dep1)

        # Deputy 2: along-track offset
        r_dep2 = self.r_chief + np.array([0.0, 200.0, 0.0])
        v_dep2 = self.v_chief.copy()
        swarm.add_deputy("Deputy 2", SpacecraftPropagator(use_j2=False), r_dep2, v_dep2)

        t_span = 300.0  # 5 minutes
        dt = 5.0

        times, chief_states, relative_tracks = swarm.propagate_swarm(
            self.r_chief, self.v_chief, t_span=t_span, dt=dt
        )

        # Dimension and step count assertions
        num_expected_steps = len(times)
        self.assertEqual(len(chief_states), num_expected_steps)
        self.assertIn("Deputy 1", relative_tracks)
        self.assertIn("Deputy 2", relative_tracks)

        dep1_track = np.asarray(relative_tracks["Deputy 1"])
        self.assertEqual(len(dep1_track), num_expected_steps)

        # LVLH coordinates must be 3D [radial, along-track, cross-track]
        self.assertEqual(dep1_track.shape[1], 3)

        # Origin check: initial relative offset must be bounded near initial separation
        initial_radial_offset = abs(dep1_track[0, 0])
        self.assertAlmostEqual(initial_radial_offset, 50.0, delta=1.0)

        # Separation must remain bounded (< 50 km over 5 minutes)
        max_dist = np.max(np.linalg.norm(dep1_track, axis=1))
        self.assertLess(max_dist, 50e3)


if __name__ == "__main__":
    unittest.main()