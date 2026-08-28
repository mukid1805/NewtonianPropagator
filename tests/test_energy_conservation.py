"""
Numerical verification test: Specific Mechanical Energy Conservation in an unperturbed two-body field.
"""
import unittest
import numpy as np
import matplotlib.pyplot as plt

from core.propagator import SpacecraftPropagator
from core.constants import G_EARTH, R_EARTH


class TestEnergyConservation(unittest.TestCase):
    def test_rk4_energy_drift(self):
        """
        Verify that specific mechanical energy drift is < 1e-5 (0.001%)
        over 5 complete orbital revolutions using RK4 integration.
        """
        # Initial conditions: Circular LEO (altitude = 500 km)
        r_mag = R_EARTH + 500_000.0
        v_mag = float(np.sqrt(G_EARTH / r_mag))  # Circular velocity

        r0 = np.array([r_mag, 0.0, 0.0])
        v0 = np.array([0.0, v_mag, 0.0])

        # Orbital period
        period = 2.0 * np.pi * np.sqrt((r_mag ** 3) / G_EARTH)
        num_orbits = 5
        t_span = num_orbits * period
        dt = 5.0  # 5-second step

        # Unperturbed engine (no J2, no drag, no thrust)
        engine = SpacecraftPropagator(use_j2=False, use_lunar=False, use_drag=False, use_srp=False)
        times, states = engine.propagate(r0, v0, t_span=t_span, dt=dt)

        # Extract positions and velocities
        r_vectors = states[:, 0:3]
        v_vectors = states[:, 3:6]

        r_norms = np.linalg.norm(r_vectors, axis=1)
        v_norms = np.linalg.norm(v_vectors, axis=1)

        # Compute specific mechanical energy: E = v^2 / 2 - mu / r
        energies = 0.5 * (v_norms ** 2) - (G_EARTH / r_norms)
        initial_energy = float(energies[0])

        # Relative energy error across all steps
        relative_errors = np.abs((energies - initial_energy) / initial_energy)
        max_energy_drift = float(np.max(relative_errors))

        print(f"\n--- Energy Conservation Verification ---")
        print(f"Number of Orbits: {num_orbits}")
        print(f"Initial Specific Energy: {initial_energy:.4f} J/kg")
        print(f"Maximum Relative Energy Drift: {max_energy_drift:.2e}")

        # Assert that the maximum relative error is below threshold (0.001%)
        self.assertLess(
            max_energy_drift,
            1e-5,
            f"Energy drift {max_energy_drift:.2e} exceeded tolerance threshold 1e-5."
        )


def plot_energy_drift():
    """Helper visualization function to plot energy drift over time."""
    r_mag = R_EARTH + 500_000.0
    v_mag = float(np.sqrt(G_EARTH / r_mag))
    r0 = np.array([r_mag, 0.0, 0.0])
    v0 = np.array([0.0, v_mag, 0.0])

    period = 2.0 * np.pi * np.sqrt((r_mag ** 3) / G_EARTH)
    t_span = 5.0 * period
    dt = 5.0

    engine = SpacecraftPropagator(use_j2=False, use_lunar=False, use_drag=False, use_srp=False)
    times, states = engine.propagate(r0, v0, t_span=t_span, dt=dt)

    r_norms = np.linalg.norm(states[:, 0:3], axis=1)
    v_norms = np.linalg.norm(states[:, 3:6], axis=1)
    energies = 0.5 * (v_norms ** 2) - (G_EARTH / r_norms)
    relative_errors = (energies - energies[0]) / np.abs(energies[0])

    plt.figure(figsize=(9.0, 4.5))
    plt.plot(times / period, relative_errors, color='navy', linewidth=1.2)
    plt.axhline(0, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Orbital Revolutions')
    plt.ylabel(r'Relative Energy Drift $\Delta\mathcal{E} / \mathcal{E}_0$')
    plt.title('Numerical Energy Conservation over 5 Orbits (RK4, dt = 5s)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    unittest.main(exit=False)
    plot_energy_drift()