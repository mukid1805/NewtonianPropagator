"""
Starter Template for Custom Orbital Mechanics Experiments.
Location: customscripts/template_custom_orbit.py
"""
import sys
from pathlib import Path

# Add project root directory to sys.path for standalone script execution
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from core.constants import G_EARTH, R_EARTH
from core.propagator import SpacecraftPropagator
from core.forces import rv_to_keplerian


def run_experiment():
    print("=" * 60)
    print("      CUSTOM ORBITAL PROPAGATION EXPERIMENT")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. ORBITAL INITIAL CONDITIONS (ECI Frame)
    # ---------------------------------------------------------
    # Define an eccentric inclined orbit (e.g., Molniya-like or GTO)
    perigee_alt = 600_000.0       # 600 km altitude at perigee [m]
    apogee_alt = 7_000_000.0      # 7,000 km altitude at apogee [m]
    inclination_deg = 51.6        # Orbital inclination [deg]

    r_p = R_EARTH + perigee_alt
    r_a = R_EARTH + apogee_alt
    a = (r_p + r_a) / 2.0
    ecc = (r_a - r_p) / (r_a + r_p)

    # Initial state at perigee
    v_perigee = float(np.sqrt(G_EARTH * (2.0 / r_p - 1.0 / a)))
    inc_rad = np.radians(inclination_deg)

    r0 = np.array([r_p, 0.0, 0.0])
    v0 = np.array([0.0, v_perigee * np.cos(inc_rad), v_perigee * np.sin(inc_rad)])

    print(f"Initial Semi-Major Axis (a): {a / 1000.0:.2f} km")
    print(f"Initial Eccentricity (e):     {ecc:.4f}")
    print(f"Initial Inclination (i):      {inclination_deg:.2f} deg")

    # ---------------------------------------------------------
    # 2. CONFIGURE PROPAGATOR & PERTURBATION FORCES
    # ---------------------------------------------------------
    engine = SpacecraftPropagator(
        mass=500.0,          # Spacecraft mass [kg]
        drag_area=2.0,       # Cross-sectional aerodynamic area [m^2]
        cd=2.2,              # Drag coefficient
        srp_area=4.0,        # Solar radiation cross-section [m^2]
        cr=1.2,              # SRP reflectivity coefficient
        isp=1800.0,          # Specific impulse [s] (for propulsion modes)
        use_j2=True,         # Earth oblateness harmonic
        use_j3=True,         # Earth pear-shape harmonic
        use_j4=True,         # Secondary oblateness harmonic
        use_lunar=True,      # Third-body lunar gravity
        use_drag=True,       # Atmospheric drag with Earth rotation
        use_srp=True         # Solar radiation pressure & Earth shadow
    )

    # Optional: Configure a directional impulsive burn or low-thrust burn
    # engine.configure_fixed_burn(start_t=3600.0, duration=120.0, thrust_vec=np.array([100.0, 0.0, 0.0]))
    # engine.configure_electric_burn(thrust_magnitude=0.05)

    # ---------------------------------------------------------
    # 3. NUMERICAL PROPAGATION
    # ---------------------------------------------------------
    orbital_period = 2.0 * np.pi * np.sqrt((a ** 3) / G_EARTH)
    num_orbits = 6
    t_span = float(num_orbits * orbital_period)
    dt = 10.0  # Integration time step [s]

    print(f"\nPropagating for {num_orbits} orbits ({t_span / 3600.0:.2f} hours, dt = {dt}s)...")
    times, states = engine.propagate(r0, v0, t_span=t_span, dt=dt, track_mass=False)
    print(f"Simulation completed across {len(times)} timesteps.")

    # ---------------------------------------------------------
    # 4. TELEMETRY & ORBITAL ELEMENTS ANALYSIS
    # ---------------------------------------------------------
    # Extract Keplerian elements every 10 steps
    stride = 10
    time_sampled = times[::stride] / 3600.0  # Time in hours
    elements = [rv_to_keplerian(states[i, 0:3], states[i, 3:6]) for i in range(0, len(times), stride)]

    altitudes_km = (np.linalg.norm(states[::stride, 0:3], axis=1) - R_EARTH) / 1000.0
    eccentricities = [el["e"] for el in elements]
    raan_deg = [el["raan_deg"] for el in elements]
    argp_deg = [el["argp_deg"] for el in elements]

    # ---------------------------------------------------------
    # 5. DIAGNOSTIC PLOTS
    # ---------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(12.0, 8.0))

    # Altitude Profile
    axs[0, 0].plot(time_sampled, altitudes_km, color='navy', linewidth=1.2)
    axs[0, 0].set_xlabel('Time (Hours)')
    axs[0, 0].set_ylabel('Altitude (km)')
    axs[0, 0].set_title('Geocentric Altitude Profile')
    axs[0, 0].grid(True, linestyle=':', alpha=0.6)

    # Eccentricity Evolution
    axs[0, 1].plot(time_sampled, eccentricities, color='crimson', linewidth=1.2)
    axs[0, 1].set_xlabel('Time (Hours)')
    axs[0, 1].set_ylabel('Eccentricity $e$')
    axs[0, 1].set_title('Eccentricity Evolution (J2 + Drag + Lunar)')
    axs[0, 1].grid(True, linestyle=':', alpha=0.6)

    # RAAN Precession (Nodal Drift from J2)
    axs[1, 0].plot(time_sampled, raan_deg, color='darkgreen', linewidth=1.2)
    axs[1, 0].set_xlabel('Time (Hours)')
    axs[1, 0].set_ylabel(r'RAAN $\Omega$ (deg)')
    axs[1, 0].set_title(r'Nodal Precession ($\dot{\Omega}_{J2}$)')
    axs[1, 0].grid(True, linestyle=':', alpha=0.6)

    # Argument of Perigee Drift
    axs[1, 1].plot(time_sampled, argp_deg, color='darkorange', linewidth=1.2)
    axs[1, 1].set_xlabel('Time (Hours)')
    axs[1, 1].set_ylabel(r'Arg of Perigee $\omega$ (deg)')
    axs[1, 1].set_title(r'Apsidal Rotation ($\dot{\omega}_{J2}$)')
    axs[1, 1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # 6. 3D TRAJECTORY VISUALIZATION
    # ---------------------------------------------------------
    SpacecraftPropagator.plot_3d(states, title=f"Custom Orbit: {num_orbits} Revolutions under Superimposed Perturbations")


if __name__ == '__main__':
    run_experiment()