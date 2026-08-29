"""
Unit tests for hyperbolic gravity assist and flyby dynamics.
"""
import unittest
import numpy as np
from core.flyby import evaluate_unpowered_flyby


class TestFlybyDynamics(unittest.TestCase):

    def test_venus_flyby_conservation(self):
        """Verifies that equal inbound/outbound v_inf results in zero powered dV."""
        v_in = np.array([10.0, 5.0, 0.0])
        v_out = np.array([5.0, 10.0, 0.0])  # Same magnitude ~ 11.18 km/s

        res = evaluate_unpowered_flyby(v_in, v_out, 'venus', min_altitude_km=300.0)

        self.assertAlmostEqual(res['dv_powered'], 0.0, places=6)
        self.assertGreater(res['h_p'], -6051.8)
        self.assertAlmostEqual(res['delta_angle_deg'], np.degrees(np.arccos(np.dot(v_in, v_out) / (np.linalg.norm(v_in)*np.linalg.norm(v_out)))), places=4)


if __name__ == '__main__':
    unittest.main()