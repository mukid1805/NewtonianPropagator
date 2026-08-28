"""
Scenario 2: LEO trajectory with J2 harmonics, atmospheric drag (with Earth rotation), and Solar Radiation Pressure.
"""
import numpy as np
from core.propagator import SpacecraftPropagator
from core.constants import R_EARTH


def run():
    # Initial state: 400 km altitude at 28.5 degree inclination
    inc = np.radians(28.5)
    v_mag = 7613.0
    r0 = np.array([R_EARTH + 400_000.0, 0.0, 0.0])
    v0 = np.array([0.0, v_mag * np.cos(inc), v_mag * np.sin(inc)])

    engine = SpacecraftPropagator(mass=500.0, drag_area=2.0, cd=2.2, srp_area=4.0, cr=1.2,
                                  use_j2=True, use_drag=True, use_srp=True)

    print("Propagating Scenario 2 (J2, Atmospheric Drag & SRP)...")
    times, states = engine.propagate(r0, v0, t_span=14400.0, dt=5.0)

    print(f"Propagated {len(times)} steps successfully.")
    SpacecraftPropagator.plot_3d(states, title="Scenario 2: LEO Trajectory with Drag, SRP & J2")


if __name__ == '__main__':
    run()