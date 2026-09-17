"""
Scenario 1: Low Earth Orbit with J2 oblateness, Lunar third-body gravity,
and a true retrograde de-boost burn.
"""
import numpy as np
from core.propagator import SpacecraftPropagator
from core.constants import R_EARTH


def run():
    r0 = np.array([R_EARTH + 400_000.0, 0.0, 0.0], dtype=np.float64)
    v0 = np.array([0.0, 7411.0, 1985.0], dtype=np.float64)

    engine = SpacecraftPropagator(use_j2=True, use_lunar=True)

    # Dynamic steering law: opposes instantaneous velocity during burn window
    def retrograde_burn_steering(t: float, r: np.ndarray, v: np.ndarray, mass: float) -> np.ndarray:
        if 1000.0 <= t <= 1200.0 and mass > 20.0:
            v_norm = np.linalg.norm(v)
            if v_norm > 0.0:

                return (250.0 / mass) * (v / v_norm)
        return np.zeros(3, dtype=np.float64)

    engine.configure_thrust(thrust_mag=250.0, steering_law=retrograde_burn_steering)

    print("Propagating Scenario 1 (True Dynamic Burn)...")
    times, states = engine.propagate(r0, v0, t_span=3 * 3600, dt=5.0)

    print(f"Propagated {len(times)} steps successfully.")
    SpacecraftPropagator.plot_3d(states, title="Scenario 1: Dynamic Burn")


if __name__ == '__main__':
    run()