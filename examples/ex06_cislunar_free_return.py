"""
Scenario 6: Cislunar Free-Return Trajectory & Earth-Moon Lagrange Points in CR3BP.
Propagates an Apollo-style free-return path and benchmarks Fixed-Step RK4 against
Adaptive-Step RK45 (Dormand-Prince) in Synodic & Inertial reference frames.
"""
import time
import numpy as np
import matplotlib.pyplot as plt

from core.integrators import rk4_step, rk45_adaptive
from core.constants import R_EARTH
from core.cr3bp import (
    cr3bp_derivatives,
    compute_lagrange_points,
    compute_jacobi_constant,
    synodic_to_inertial,
    MU_EARTH_MOON,
    L_STAR,
    T_STAR,
    V_STAR,
)


def run():
    print("=" * 72)
    print("    CISLUNAR CR3BP & FREE-RETURN TRAJECTORY (RK4 vs RK45 BENCHMARK)")
    print("=" * 72)

    # 1. Compute and display Earth-Moon Lagrange Points
    lagrange_pts = compute_lagrange_points(MU_EARTH_MOON)
    print("\nComputed Earth-Moon Lagrange Points (Synodic Coordinates):")
    for name, coord in lagrange_pts.items():
        print(f"  {name}: X = {coord[0]:.6f}, Y = {coord[1]:.6f} "
              f"(dim: {coord[0] * L_STAR / 1000.0:.1f} km)")

    # 2. Free-Return Trajectory Initial Conditions (Non-Dimensional)
    # Trans-Lunar Injection (TLI) from ~200 km Earth LEO
    r_leo_nd = (R_EARTH + 200_000.0) / L_STAR
    x0_syn = -MU_EARTH_MOON + r_leo_nd
    y0_syn = 0.0
    z0_syn = 0.0

    # Injected with high prograde velocity in synodic frame (~10.915 km/s dimensional)
    v_inj_dim = 10_915.0  # m/s
    vy0_syn = (v_inj_dim / V_STAR) - r_leo_nd  # Rotating frame transport correction

    state0 = np.array([x0_syn, y0_syn, z0_syn, 0.0, vy0_syn, 0.0], dtype=np.float64)
    c0 = compute_jacobi_constant(state0, MU_EARTH_MOON)
    print(f"\nInitial State Trans-Lunar Jacobi Constant C0 = {c0:.6f}")

    t_span_nd = 2.4  # ~10.44 days (1 ND time ~ 4.348 days)
    dt_rk4_nd = 0.0001
    num_steps_rk4 = int(t_span_nd / dt_rk4_nd)

    def cr3bp_derivs_wrapped(t, state):
        return cr3bp_derivatives(t, state, MU_EARTH_MOON)

    # -------------------------------------------------------------------------
    # 3A. Propagate using Fixed-Step RK4 (Baseline)
    # -------------------------------------------------------------------------
    print(f"\n[1/2] Propagating with Fixed-Step RK4 (dt = {dt_rk4_nd}, {num_steps_rk4:,} steps)...")
    times_rk4 = np.linspace(0.0, t_span_nd, num_steps_rk4)
    states_rk4 = np.zeros((num_steps_rk4, 6), dtype=np.float64)
    states_rk4[0] = state0

    t0_wall = time.perf_counter()
    for i in range(1, num_steps_rk4):
        states_rk4[i] = rk4_step(cr3bp_derivs_wrapped, times_rk4[i - 1], states_rk4[i - 1], dt_rk4_nd)
    wall_rk4 = time.perf_counter() - t0_wall

    c_final_rk4 = compute_jacobi_constant(states_rk4[-1], MU_EARTH_MOON)
    drift_rk4 = abs(c_final_rk4 - c0)
    print(f"      Completed in {wall_rk4:.4f} s | Jacobi Drift: {drift_rk4:.2e}")

    # -------------------------------------------------------------------------
    # 3B. Propagate using Adaptive RK45 (Dormand-Prince)
    # -------------------------------------------------------------------------
    rtol = 1e-8
    atol = 1e-10
    print(f"\n[2/2] Propagating with Adaptive RK45 (rtol = {rtol:.1e}, atol = {atol:.1e})...")
    t0_wall = time.perf_counter()
    times_rk45, states_rk45 = rk45_adaptive(
        derivs_func=cr3bp_derivs_wrapped,
        t_span=(0.0, t_span_nd),
        y0=state0,
        rtol=rtol,
        atol=atol,
        h_init=1e-3,
        h_min=1e-7,
        h_max=0.01,
    )
    wall_rk45 = time.perf_counter() - t0_wall
    num_steps_rk45 = len(times_rk45)

    c_final_rk45 = compute_jacobi_constant(states_rk45[-1], MU_EARTH_MOON)
    drift_rk45 = abs(c_final_rk45 - c0)
    print(f"      Completed in {wall_rk45:.4f} s ({num_steps_rk45:,} steps) | Jacobi Drift: {drift_rk45:.2e}")

    # -------------------------------------------------------------------------
    # Performance Summary Telemetry
    # -------------------------------------------------------------------------
    speedup = wall_rk4 / wall_rk45 if wall_rk45 > 0 else 0.0
    step_reduction = (1.0 - (num_steps_rk45 / num_steps_rk4)) * 100.0

    print("\n" + "-" * 72)
    print(f"{'Performance Metric':<28} | {'Fixed RK4':<18} | {'Adaptive RK45':<18}")
    print("-" * 72)
    print(f"{'Execution Time (s)':<28} | {wall_rk4:<18.4f} | {wall_rk45:<18.4f}")
    print(f"{'Total Integration Steps':<28} | {num_steps_rk4:<18,} | {num_steps_rk45:<18,}")
    print(f"{'Jacobi Energy Drift |ΔC|':<28} | {drift_rk4:<18.2e} | {drift_rk45:<18.2e}")
    print("-" * 72)
    print(f"Runtime Speedup        : {speedup:.2f}x faster")
    print(f"Step Count Reduction   : {step_reduction:.2f}% fewer evaluations\n")

    # 4. Transform to Dimensional Inertial Frame (using high-fidelity RK45 states)
    states_eci_km = synodic_to_inertial(times_rk45, states_rk45, MU_EARTH_MOON)

    # 5. Dual-Frame Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 6.0))

    # --- Subplot 1: Earth-Moon Synodic (Rotating) Frame ---
    ax1.plot(states_rk4[:, 0], states_rk4[:, 1], color='gray', linestyle='--', linewidth=1.0,
             alpha=0.6, label='RK4 Path (Fixed dt)')
    ax1.plot(states_rk45[:, 0], states_rk45[:, 1], color='crimson', linewidth=1.5,
             label='RK45 Path (Adaptive)')

    ax1.scatter(-MU_EARTH_MOON, 0.0, color='dodgerblue', s=180, edgecolors='black', label='Earth', zorder=5)
    ax1.scatter(1.0 - MU_EARTH_MOON, 0.0, color='gray', s=70, edgecolors='black', label='Moon', zorder=5)

    theta_orbit = np.linspace(0, 2 * np.pi, 300)
    x_moon_orbit = -MU_EARTH_MOON + 1.0 * np.cos(theta_orbit)
    y_moon_orbit = 1.0 * np.sin(theta_orbit)
    ax1.plot(x_moon_orbit, y_moon_orbit, color='silver', linestyle='--', linewidth=1.0, label="Lunar Distance", zorder=1)

    l_colors = {'L1': 'gold', 'L2': 'orange', 'L3': 'teal', 'L4': 'purple', 'L5': 'magenta'}
    for name, coord in lagrange_pts.items():
        ax1.scatter(coord[0], coord[1], color=l_colors[name], s=50, marker='^', edgecolors='black', label=f'{name}', zorder=4)

    ax1.set_xlim([-1.2, 1.2])
    ax1.set_ylim([-1.2, 1.2])
    ax1.set_xlabel('Synodic X (Non-Dimensional)')
    ax1.set_ylabel('Synodic Y (Non-Dimensional)')
    ax1.set_title('Earth-Moon Synodic Frame (CR3BP)')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.axis('equal')
    ax1.legend(loc='lower left', fontsize=8, ncol=2)

    # --- Subplot 2: Earth-Centered Inertial (ECI) Frame ---
    x_eci = states_eci_km[:, 0]
    y_eci = states_eci_km[:, 1]

    ax2.plot(x_eci, y_eci, color='crimson', linewidth=1.5, label='Spacecraft Trajectory (RK45)')

    theta_orbit = np.linspace(0, 2 * np.pi, 200)
    r_moon_km = L_STAR / 1000.0
    ax2.plot(r_moon_km * np.cos(theta_orbit), r_moon_km * np.sin(theta_orbit), color='silver', linestyle='--', label="Moon's Orbit")

    ax2.scatter(0.0, 0.0, color='dodgerblue', s=180, edgecolors='black', label='Earth (Origin)', zorder=5)

    ax2.scatter(r_moon_km, 0.0, color='lightgray', s=60, edgecolors='black', label='Moon @ Launch', zorder=4)
    t_end_rad = times_rk45[-1]
    ax2.scatter(r_moon_km * np.cos(t_end_rad), r_moon_km * np.sin(t_end_rad), color='gray', s=80, edgecolors='black', label='Moon @ Return', zorder=5)

    ax2.set_xlim([-500000, 500000])
    ax2.set_ylim([-500000, 500000])
    ax2.set_xlabel('ECI X (km)')
    ax2.set_ylabel('ECI Y (km)')
    ax2.set_title('Earth-Centered Inertial (ECI) Trajectory')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.axis('equal')
    ax2.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run()
