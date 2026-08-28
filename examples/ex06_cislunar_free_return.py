"""
Scenario 6: Cislunar Free-Return Trajectory & Earth-Moon Lagrange Points in CR3BP.
Propagates an Apollo-style Figure-8 free-return path and plots both Synodic & Inertial frames.
"""
import numpy as np
import matplotlib.pyplot as plt

from core.integrators import rk4_step
from core.constants import R_EARTH
from core.cr3bp import (
    cr3bp_derivatives,
    compute_lagrange_points,
    compute_jacobi_constant,
    synodic_to_inertial,
    MU_EARTH_MOON,
    L_STAR,
    T_STAR,
    V_STAR
)


def run():
    print("=" * 65)
    print("    CISLUNAR CR3BP & FREE-RETURN TRAJECTORY PROPAGATOR")
    print("=" * 65)

    # 1. Compute and display Earth-Moon Lagrange Points
    lagrange_pts = compute_lagrange_points(MU_EARTH_MOON)
    print("\nComputed Earth-Moon Lagrange Points (Synodic Coordinates):")
    for name, coord in lagrange_pts.items():
        print(f"  {name}: X = {coord[0]:.6f}, Y = {coord[1]:.6f} (dim: {coord[0] * L_STAR / 1000.0:.1f} km)")

    # 2. Free-Return Trajectory Initial Conditions (Non-Dimensional)
    # Trans-Lunar Injection (TLI) from ~200 km Earth LEO
    r_leo_nd = (R_EARTH + 200_000.0) / L_STAR
    x0_syn = -MU_EARTH_MOON + r_leo_nd
    y0_syn = 0.0
    z0_syn = 0.0

    # Injected with high prograde velocity in synodic frame (~10.9 km/s dimensional)
    v_inj_dim = 10_915.0  # m/s
    vy0_syn = (v_inj_dim / V_STAR) - r_leo_nd  # Correct for rotating frame transport velocity

    state0 = np.array([x0_syn, y0_syn, z0_syn, 0.0, vy0_syn, 0.0])
    c0 = compute_jacobi_constant(state0, MU_EARTH_MOON)
    print(f"\nInitial State Trans-Lunar Jacobi Constant C = {c0:.6f}")

    # 3. Propagate in Synodic Frame across ~7.5 days (~1.72 non-dimensional time)
    t_span_nd = 1.72
    dt_nd = 0.0001
    num_steps = int(t_span_nd / dt_nd)

    times_nd = np.linspace(0, t_span_nd, num_steps)
    states_syn = np.zeros((num_steps, 6))
    states_syn[0] = state0

    print(f"Propagating {t_span_nd * T_STAR / 86400.0:.2f}-day cislunar free-return path...")
    for i in range(1, num_steps):
        states_syn[i] = rk4_step(cr3bp_derivatives, times_nd[i - 1], states_syn[i - 1], dt_nd)

    c_final = compute_jacobi_constant(states_syn[-1], MU_EARTH_MOON)
    print(f"Jacobi Constant Drift: {abs(c_final - c0):.2e}")

    # 4. Transform to Dimensional Inertial Frame
    states_eci_km = synodic_to_inertial(times_nd, states_syn, MU_EARTH_MOON)

    # 5. Dual-Frame Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 6.0))

    # --- Subplot 1: Earth-Moon Synodic (Rotating) Frame ---
    # Trajectory
    ax1.plot(states_syn[:, 0], states_syn[:, 1], color='crimson', linewidth=1.5, label='Free-Return Path')

    # Earth and Moon
    ax1.scatter(-MU_EARTH_MOON, 0.0, color='dodgerblue', s=180, edgecolors='black', label='Earth', zorder=5)
    ax1.scatter(1.0 - MU_EARTH_MOON, 0.0, color='gray', s=70, edgecolors='black', label='Moon', zorder=5)

    # Plot Lagrange Points
    l_colors = {'L1': 'gold', 'L2': 'orange', 'L3': 'teal', 'L4': 'purple', 'L5': 'magenta'}
    for name, coord in lagrange_pts.items():
        ax1.scatter(coord[0], coord[1], color=l_colors[name], s=50, marker='^', edgecolors='black', label=f'{name}',
                    zorder=4)

    ax1.set_xlabel('Synodic X (Non-Dimensional)')
    ax1.set_ylabel('Synodic Y (Non-Dimensional)')
    ax1.set_title('Earth-Moon Synodic Frame with Lagrange Points')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.axis('equal')
    ax1.legend(loc='lower left', fontsize=8, ncol=2)

    # --- Subplot 2: Earth-Centered Inertial (ECI) Frame ---
    x_eci = states_eci_km[:, 0]
    y_eci = states_eci_km[:, 1]

    ax2.plot(x_eci, y_eci, color='crimson', linewidth=1.5, label='Spacecraft Inertial Path')

    # Plot Moon's circular orbit
    theta_orbit = np.linspace(0, 2 * np.pi, 200)
    r_moon_km = L_STAR / 1000.0
    ax2.plot(r_moon_km * np.cos(theta_orbit), r_moon_km * np.sin(theta_orbit), color='silver', linestyle='--',
             label="Moon's Orbit")

    # Earth center
    ax2.scatter(0.0, 0.0, color='dodgerblue', s=180, edgecolors='black', label='Earth (Origin)', zorder=5)

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