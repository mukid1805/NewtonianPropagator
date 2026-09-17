"""
Scenario 10: End-to-End Two-Impulse Orbital Rendezvous and Docking Scenario.
Executes an STM-based differential corrections targeting solver to calculate
optimal departure and braking burns (Delta-V0, Delta-Vf) under Earth's J2
perturbation, and maps relative motion into the Target-centred Hill/LVLH frame.
"""
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

from core.constants import G_EARTH, R_EARTH
from core.propagator import SpacecraftPropagator
from core.targeting import (
    solve_complete_rendezvous,
    eci_to_lvlh,
    plot_rendezvous_3d,
    plot_lvlh_docking,
)


def run():
    print("=" * 72)
    print("   TWO-IMPULSE RENDEZVOUS TARGETING (LEO J2)")
    print("=" * 72)

    # -------------------------------------------------------------------------
    # 1. Mission Configuration & Orbit Geometry
    # -------------------------------------------------------------------------
    engine = SpacecraftPropagator(use_j2=True)

    alt_target = 820_000.0   # 420 km circular orbit (ISS-like)
    r_target_mag = R_EARTH + alt_target
    v_target_mag = np.sqrt(G_EARTH / r_target_mag)
    inc = np.radians(51.64)

    alt_chaser = 390_000.0   # 390 km circular phasing orbit (-30 km)
    r_chaser_mag = R_EARTH + alt_chaser
    v_chaser_mag = np.sqrt(G_EARTH / r_chaser_mag)

    # Chaser state at epoch t0 (ascending node)
    r0_chaser = np.array([r_chaser_mag, 0.0, 0.0], dtype=np.float64)
    v0_chaser = np.array(
        [0.0, v_chaser_mag * np.cos(inc), v_chaser_mag * np.sin(inc)],
        dtype=np.float64,
    )

    # Target state at epoch t0 (offset by 1.0 deg true anomaly downrange)
    theta_offset = np.radians(1.0)
    r0_target = np.array(
        [
            r_target_mag * np.cos(theta_offset),
            r_target_mag * np.sin(theta_offset) * np.cos(inc),
            r_target_mag * np.sin(theta_offset) * np.sin(inc),
        ],
        dtype=np.float64,
    )
    v0_target = np.array(
        [
            -v_target_mag * np.sin(theta_offset),
            v_target_mag * np.cos(theta_offset) * np.cos(inc),
            v_target_mag * np.cos(theta_offset) * np.sin(inc),
        ],
        dtype=np.float64,
    )

    # Planned transfer time-of-flight: 1.25 orbital periods
    t_period_target = 2.0 * np.pi * np.sqrt((r_target_mag**3) / G_EARTH)
    tof = 1.25 * t_period_target

    print(f"Target Orbit Altitude:           {alt_target / 1000.0:.1f} km (i = 51.64 deg)")
    print(f"Chaser Phasing Altitude:         {alt_chaser / 1000.0:.1f} km (-30.0 km delta-h)")
    print(f"Initial True Anomaly Gap:        {np.degrees(theta_offset):.2f} deg (~{(r_target_mag * theta_offset) / 1000.0:.1f} km downrange)")
    print(f"Transfer Time of Flight:         {tof:.1f} s ({tof / 60.0:.2f} min / {tof / t_period_target:.2f} periods)")
    print("-" * 72)

    # -------------------------------------------------------------------------
    # 2. Execute Two-Impulse Differential Corrections Solver
    # -------------------------------------------------------------------------
    print("Executing Newton-Raphson STM Differential Corrections...")
    t_start = time.perf_counter()

    solution = solve_complete_rendezvous(
        engine=engine,
        r0_chaser=r0_chaser,
        v0_chaser=v0_chaser,
        r0_target=r0_target,
        v0_target=v0_target,
        tof=tof,
        tol_m=0.01,  # 1 cm terminal position tolerance
        max_iter=15,
        dt=10.0,
    )

    solve_time = time.perf_counter() - t_start

    print("\n" + "=" * 72)
    print("   OPTIMAL TWO-IMPULSE RENDEZVOUS MANOEUVRE IDENTIFIED")
    print("=" * 72)
    print(f"Differential Corrector Runtime:  {solve_time:.4f} seconds")
    print(f"Iterations to Converge:          {solution['iterations']}")
    print(f"Terminal Miss Distance:          {solution['final_miss_m'] * 1000.0:.2f} mm")
    print("-" * 72)
    print(f"Departure Burn Delta-V0:         {solution['delta_v0_mag']:.3f} m/s")
    print(f"  Delta-V0 Vector (ECI):         [{solution['delta_v0'][0]:.3f}, {solution['delta_v0'][1]:.3f}, {solution['delta_v0'][2]:.3f}] m/s")
    print(f"Arrival Braking Burn Delta-Vf:   {solution['delta_vf_mag']:.3f} m/s")
    print(f"  Delta-Vf Vector (ECI):         [{solution['delta_vf'][0]:.3f}, {solution['delta_vf'][1]:.3f}, {solution['delta_vf'][2]:.3f}] m/s")
    print(f"Total Manoeuvre Budget:          {solution['delta_v_total']:.3f} m/s")
    print(f"Residual Relative Velocity:      {solution['relative_vf_mag']:.2e} m/s")
    print("=" * 72)

    # -------------------------------------------------------------------------
    # 3. Kinematic Transformation to Target Hill / LVLH Frame
    # -------------------------------------------------------------------------
    chaser_states = solution["chaser_transfer_states"]
    target_states = solution["target_states"]

    r_lvlh, _ = eci_to_lvlh(
        r_target=target_states[:, 0:3],
        v_target=target_states[:, 3:6],
        r_chaser=chaser_states[:, 0:3],
        v_chaser=chaser_states[:, 3:6],
    )

    # -------------------------------------------------------------------------
    # 4. Trajectory Visualisation
    # -------------------------------------------------------------------------
    print("\nRendering 3D Inertial Trajectory & 2D Relative Motion...")

    plot_rendezvous_3d(
        chaser_states=chaser_states,
        target_states=target_states,
        delta_v0=solution["delta_v0"],
        delta_vf=solution["delta_vf"],
        burn_scale=25.0,
        title="Scenario 10: Two-Impulse Orbital Rendezvous (ECI Frame)",
    )

    plot_lvlh_docking(
        r_lvlh=r_lvlh,
        title="Scenario 10: Relative Rendezvous Trajectory in Target LVLH Frame",
    )

    # print("\nRendering done.")

main = run

if __name__ == "__main__":
    run()
