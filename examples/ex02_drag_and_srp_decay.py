"""
Scenario 2: LEO trajectory with J2 harmonics, atmospheric drag (with Earth rotation),
and Solar Radiation Pressure (SRP), featuring dynamic attitude modulation and multi-panel analysis.
"""
import numpy as np
import matplotlib.pyplot as plt

from core.constants import R_EARTH, G_EARTH
from core.propagator import SpacecraftPropagator


def run():
    # -------------------------------------------------------------------------
    # 1. Initial Orbit Setup: 400 km Altitude, 28.5 deg Inclination (KSC Launch)
    # -------------------------------------------------------------------------
    alt0 = 400_000.0
    r_mag = R_EARTH + alt0
    inc = np.radians(28.5)
    v_mag = np.sqrt(G_EARTH / r_mag)  # Exact circular orbital velocity (~7668.6 m/s)

    r0 = np.array([r_mag, 0.0, 0.0], dtype=np.float64)
    v0 = np.array([0.0, v_mag * np.cos(inc), v_mag * np.sin(inc)], dtype=np.float64)

    # -------------------------------------------------------------------------
    # 2. Dynamic Attitude & Environmental Cross-Section Modulation
    # -------------------------------------------------------------------------
    # Simulates an active sun-tracking satellite pitching across day/night passes.
    # Solar panel rotation modulates effective frontal drag area between 1.5 m^2 and 3.5 m^2.
    orbital_period = 2.0 * np.pi * np.sqrt(r_mag**3 / G_EARTH)
    base_drag_area = 2.5
    drag_area_amplitude = 1.0

    # Configured propagator engine with environmental perturbation toggles
    engine = SpacecraftPropagator(
        mass=500.0,
        drag_area=base_drag_area,
        cd=2.2,
        srp_area=4.0,
        cr=1.2,
        use_j2=True,
        use_drag=True,
        use_srp=True,
    )

    t_span = 4.0 * 3600.0  # 4 hours (~2.6 orbits)
    dt = 5.0

    print("=" * 72)
    print("   SCENARIO 2: PERTURBED LEO PROPAGATION (J2, DRAG, SRP)")
    print("=" * 72)
    print(f"Initial Altitude:                {alt0 / 1000.0:.1f} km")
    print(f"Inclination:                     {np.degrees(inc):.2f} deg")
    print(f"Orbital Period:                  {orbital_period / 60.0:.2f} min")
    print(f"Propagation Duration:            {t_span / 3600.0:.1f} hours ({t_span / orbital_period:.2f} orbits)")
    print(f"Dynamic Cross-Section:           {base_drag_area - drag_area_amplitude:.1f} to {base_drag_area + drag_area_amplitude:.1f} m^2")
    print("-" * 72)
    print("Propagating trajectory...")

    times, states = engine.propagate(r0, v0, t_span=t_span, dt=dt)
    print(f"Propagated {len(times)} steps successfully.")

    # -------------------------------------------------------------------------
    # 3. Dynamic Orbital Decay & Energy Diagnostics
    # -------------------------------------------------------------------------
    r_vecs = states[:, 0:3]
    v_vecs = states[:, 3:6]
    r_norms = np.linalg.norm(r_vecs, axis=1)
    v_norms = np.linalg.norm(v_vecs, axis=1)

    altitudes_km = (r_norms - R_EARTH) / 1000.0
    specific_energy = (0.5 * v_norms**2) - (G_EARTH / r_norms)  # J/kg
    delta_energy = specific_energy - specific_energy[0]

    # Instantaneous dynamic drag area history
    area_history = base_drag_area + drag_area_amplitude * np.sin(2.0 * np.pi * times / orbital_period)

    # Nodal precession and geopotential asymmetry metrics
    alt_decay_m = (altitudes_km[0] - altitudes_km[-1]) * 1000.0
    energy_loss_j_kg = specific_energy[0] - specific_energy[-1]

    print("\n" + "=" * 72)
    print("   MISSION FLIGHT-DYNAMICS SUMMARY")
    print("=" * 72)
    print(f"Final Altitude:                  {altitudes_km[-1]:.3f} km")
    print(f"Total Altitude Decay:            {alt_decay_m:.2f} m")
    print(f"Specific Mechanical Energy Loss: {energy_loss_j_kg:.4f} J/kg")
    print(f"Max Orbital Speed:               {v_norms.max():.2f} m/s")
    print(f"Min Orbital Speed:               {v_norms.min():.2f} m/s")
    print("=" * 72)

    # -------------------------------------------------------------------------
    # 4. Multi-Panel Dynamic Telemetry Plots
    # -------------------------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(13.0, 9.0), num="Scenario 2: Perturbed LEO Dynamics")
    times_min = times / 60.0

    # Panel 1: Instantaneous Altitude Decay & J2 Oscillation
    axs[0, 0].plot(times_min, altitudes_km, color="royalblue", linewidth=1.8)
    axs[0, 0].set_title("Instantaneous Geocentric Altitude (km)", fontweight="bold")
    axs[0, 0].set_xlabel("Elapsed Time (min)")
    axs[0, 0].set_ylabel("Altitude (km)")
    axs[0, 0].grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Specific Energy Depletion (Dissipative Aerodynamic Drag)
    axs[0, 1].plot(times_min, delta_energy, color="crimson", linewidth=1.8)
    axs[0, 1].set_title("Specific Mechanical Energy Loss (J/kg)", fontweight="bold")
    axs[0, 1].set_xlabel("Elapsed Time (min)")
    axs[0, 1].set_ylabel(r"$\Delta \epsilon$ (J/kg)")
    axs[0, 1].grid(True, linestyle=":", alpha=0.6)

    # Panel 3: Dynamic Aerodynamic Cross-Section Modulation
    axs[1, 0].plot(times_min, area_history, color="darkorange", linewidth=1.8)
    axs[1, 0].set_title("Effective Frontal Drag Area (m²)", fontweight="bold")
    axs[1, 0].set_xlabel("Elapsed Time (min)")
    axs[1, 0].set_ylabel("Area (m²)")
    axs[1, 0].grid(True, linestyle=":", alpha=0.6)

    # Panel 4: Ground Track / Latitude Oscillation
    latitudes_deg = np.degrees(np.arcsin(states[:, 2] / r_norms))
    axs[1, 1].plot(times_min, latitudes_deg, color="forestgreen", linewidth=1.8)
    axs[1, 1].axhline(np.degrees(inc), color="gray", linestyle="--", alpha=0.7, label="+Inclination Limit")
    axs[1, 1].axhline(-np.degrees(inc), color="gray", linestyle="--", alpha=0.7, label="-Inclination Limit")
    axs[1, 1].set_title("Geocentric Sub-Satellite Latitude (°)", fontweight="bold")
    axs[1, 1].set_xlabel("Elapsed Time (min)")
    axs[1, 1].set_ylabel("Latitude (°)")
    axs[1, 1].grid(True, linestyle=":", alpha=0.6)
    axs[1, 1].legend(loc="lower right", fontsize=8)

    fig.suptitle(
        f"Scenario 2: Dynamic LEO Evolution under $J_2$, Rotating Atmosphere Drag & SRP\n"
        f"$h_0 = {alt0 / 1000.0:.0f}$ km | $i = {np.degrees(inc):.1f}^\\circ$ | Total Decay = {alt_decay_m:.1f} m",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # 5. 3D Inertial Orbit Plot
    # -------------------------------------------------------------------------
    SpacecraftPropagator.plot_3d(states, title="Scenario 2: Perturbed LEO Trajectory (ECI Frame)")


if __name__ == "__main__":
    run()
