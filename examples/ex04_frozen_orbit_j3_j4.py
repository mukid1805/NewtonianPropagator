"""
Scenario 4: Higher-Order Geopotential Harmonics (J2 + J3 + J4) & Frozen Orbit Analysis.
Demonstrates stability at critical inclination (i = 63.435 deg, omega = 270 deg).
"""
import numpy as np
import matplotlib.pyplot as plt

from core.propagator import SpacecraftPropagator
from core.constants import G_EARTH, R_EARTH
from core.forces import rv_to_keplerian


def run():
    # Orbit Parameters: High-altitude LEO / Molniya-like perigee
    a = R_EARTH + 800_000.0  # Semi-major axis (7178 km)
    e_frozen = 0.001  # Low eccentricity
    inc_crit = np.radians(63.4349)  # Critical inclination (cos^2(i) = 1/5)

    # 1. State initialization for Frozen Orbit (omega = 270 deg)
    # At omega = 270 deg and true anomaly = 0, perigee is on negative Z in orbital plane
    r_mag0 = a * (1.0 - e_frozen)
    v_mag0 = np.sqrt(G_EARTH * (2.0 / r_mag0 - 1.0 / a))

    r0_frozen = np.array([r_mag0, 0.0, 0.0])
    v0_frozen = np.array([0.0, v_mag0 * np.cos(inc_crit), v_mag0 * np.sin(inc_crit)])

    # 2. State initialization for Non-Frozen Drift Orbit (i = 45 deg, omega = 45 deg)
    inc_drift = np.radians(45.0)
    r0_drift = np.array([r_mag0, 0.0, 0.0])
    v0_drift = np.array([0.0, v_mag0 * np.cos(inc_drift), v_mag0 * np.sin(inc_drift)])

    # Propagate with J2 + J3 + J4 over 60 days
    days = 60
    t_span = float(days * 86400.0)
    dt = 60.0  # 60s step for long-term evolution

    engine = SpacecraftPropagator(use_j2=True, use_j3=True, use_j4=True)

    print(f"Propagating Frozen Orbit under J2+J3+J4 for {days} days...")
    times, states_frozen = engine.propagate(r0_frozen, v0_frozen, t_span=t_span, dt=dt)

    print(f"Propagating Non-Critical Drift Orbit for {days} days...")
    _, states_drift = engine.propagate(r0_drift, v0_drift, t_span=t_span, dt=dt)

    # Convert state histories to Classical Orbital Elements
    print("Extracting orbital elements...")
    elements_frozen = [rv_to_keplerian(states_frozen[i, 0:3], states_frozen[i, 3:6]) for i in range(0, len(times), 10)]
    elements_drift = [rv_to_keplerian(states_drift[i, 0:3], states_drift[i, 3:6]) for i in range(0, len(times), 10)]
    t_days = times[::10] / 86400.0

    e_f = [elem["e"] for elem in elements_frozen]
    argp_f = [elem["argp_deg"] for elem in elements_frozen]

    e_d = [elem["e"] for elem in elements_drift]
    argp_d = [elem["argp_deg"] for elem in elements_drift]

    # Plot Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.0))

    # Eccentricity Evolution
    ax1.plot(t_days, e_f, color='navy', label=r'Frozen Orbit ($i=63.43^\circ, \omega=270^\circ$)')
    ax1.plot(t_days, e_d, color='firebrick', linestyle='--', label=r'Drifting Orbit ($i=45^\circ$)')
    ax1.set_xlabel('Time (Days)')
    ax1.set_ylabel('Eccentricity $e$')
    ax1.set_title('Eccentricity Stability under J2+J3+J4')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()

    # Argument of Perigee Drift
    ax2.plot(t_days, argp_f, color='navy', label=r'Frozen Orbit ($\omega$ Locked)')
    ax2.plot(t_days, argp_d, color='firebrick', linestyle='--', label=r'Drifting Orbit ($\dot{\omega} \neq 0$)')
    ax2.set_xlabel('Time (Days)')
    ax2.set_ylabel(r'Argument of Perigee $\omega$ (deg)')
    ax2.set_title(r'Apsidal Line Drift ($\dot{\omega}$)')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run()