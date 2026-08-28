"""
Scenario 3: 30-day continuous low-thrust orbit raising spiral with dynamic fuel mass depletion.
"""
import numpy as np
from core.propagator import SpacecraftPropagator
from core.constants import R_EARTH


def run():
    # Initial state: 500 km circular LEO
    r0 = np.array([R_EARTH + 500_000.0, 0.0, 0.0])
    v0 = np.array([0.0, 7672.0, 500.0])
    m0 = 300.0

    engine = SpacecraftPropagator(mass=m0, isp=1800.0, use_j2=True)
    # Continuous 80 mN Hall thruster aligned with velocity vector
    engine.configure_electric_burn(thrust_magnitude=0.08)

    days = 30
    t_span = days * 86400.0
    dt = 30.0

    print(f"Propagating Scenario 3 ({days}-Day Low-Thrust Electric Spiral)...")
    times, states = engine.propagate(r0, v0, t_span=t_span, dt=dt, track_mass=True)

    final_mass = states[-1, 6]
    final_alt = np.linalg.norm(states[-1, 0:3]) / 1000.0 - R_EARTH / 1000.0
    print(f"Simulation Complete!")
    print(f"Propellant consumed: {m0 - final_mass:.2f} kg / {m0} kg")
    print(f"Final Altitude: {final_alt:.2f} km")

    SpacecraftPropagator.plot_3d(states, title=f"Scenario 3: Continuous Low-Thrust Orbit Raising ({days} Days)")


if __name__ == '__main__':
    run()