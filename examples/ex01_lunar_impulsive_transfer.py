"""
Scenario 1: Low Earth Orbit with J2 oblateness, Lunar third-body gravity, and a timed directional thrust burn.
"""
import numpy as np
from core.propagator import SpacecraftPropagator
from core.constants import R_EARTH


def run():
    # Initial state: ~400 km altitude, 15 degree inclination
    r0 = np.array([R_EARTH + 400_000.0, 0.0, 0.0])
    v0 = np.array([0.0, 7411.0, 1985.0])

    # Propagator setup
    engine = SpacecraftPropagator(use_j2=True, use_lunar=True)

    # 250 N prograde burn for 200 seconds starting at t = 1000s
    engine.configure_fixed_burn(start_t=1000.0, duration=200.0, thrust_vec=np.array([250.0, 0.0, 0.0]))

    print("Propagating Scenario 1 (Lunar 3rd-Body & Impulsive Burn)...")
    times, states = engine.propagate(r0, v0, t_span=3 * 3600, dt=5.0)

    print(f"Propagated {len(times)} steps successfully.")
    SpacecraftPropagator.plot_3d(states, title="Scenario 1: Impulsive Burn & Lunar Gravity Perturbation")


if __name__ == '__main__':
    run()