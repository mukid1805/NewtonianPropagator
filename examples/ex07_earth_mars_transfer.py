"""
Scenario 7: Earth-to-Mars Interplanetary Mission Design & Flight Dynamics.
Performs global Porkchop optimization across calendar launch windows,
evaluates deliverable science payload capacity via core.launchers,
extracts optimal transfer opportunities, and benchmarks ballistic arc
propagation comparing Fixed-Step RK4 against Adaptive RK45 (Dormand-Prince).
"""
import time
import warnings
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from core.time import datetime_to_jd, jd_to_datetime, JD_J2000
from core.ephemeris import get_planet_state, MU_SUN, AU
from core.integrators import rk4_step, rk45_adaptive
from core.lambert import solve_lambert
from core.launchers import get_launcher, LaunchVehicle


def solar_gravity_derivs(t: float, state: np.ndarray) -> np.ndarray:
    """Heliocentric point-mass gravity equations of motion."""
    r = state[0:3]
    v = state[3:6]
    a = -(MU_SUN / (np.linalg.norm(r) ** 3)) * r
    return np.concatenate([v, a])


def compute_porkchop(dep_dates: np.ndarray, arr_dates: np.ndarray, launcher: LaunchVehicle):
    """
    Computes C3 energy, arrival v_infinity, TOF, total Delta-V,
    and deliverable science payload mass across departure and arrival date grids.
    """
    n_dep = len(dep_dates)
    n_arr = len(arr_dates)

    c3_grid = np.full((n_arr, n_dep), np.nan)
    vinf_arr_grid = np.full((n_arr, n_dep), np.nan)
    tof_grid = np.full((n_arr, n_dep), np.nan)
    total_dv_grid = np.full((n_arr, n_dep), np.nan)
    payload_grid = np.full((n_arr, n_dep), np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for j, t_dep_mjd in enumerate(dep_dates):
            r_earth, v_earth = get_planet_state('earth', t_dep_mjd, frame='ecliptic')

            for i, t_arr_mjd in enumerate(arr_dates):
                tof_days = t_arr_mjd - t_dep_mjd
                tof_grid[i, j] = tof_days

                if tof_days < 80.0 or tof_days > 450.0:
                    continue

                try:
                    r_mars, v_mars = get_planet_state('mars', t_arr_mjd, frame='ecliptic')
                    v1_trans, v2_trans = solve_lambert(r_earth, r_mars, tof_days * 86400.0, MU_SUN, prograde=True)

                    v_dep_inf = np.linalg.norm(v1_trans - v_earth)
                    v_arr_inf = np.linalg.norm(v2_trans - v_mars)

                    c3 = v_dep_inf ** 2
                    tot_dv = v_dep_inf + v_arr_inf

                    # Soft-cap extreme energies to stabilize matplotlib contouring
                    if c3 < 1000.0:
                        c3_grid[i, j] = min(c3, 150.0)
                        vinf_arr_grid[i, j] = min(v_arr_inf, 25.0)
                        total_dv_grid[i, j] = min(tot_dv, 25.0)
                        payload_grid[i, j] = launcher.payload_for_c3(c3)

                except (ValueError, ZeroDivisionError):
                    continue

    return c3_grid, vinf_arr_grid, tof_grid, total_dv_grid, payload_grid


def run():
    print("=" * 75)
    print("   EARTH-TO-MARS MISSION DESIGN: PORKCHOP & LAUNCH VEHICLE PAYLOAD")
    print("=" * 75)

    # 1. Select Launch Vehicle Configuration
    launcher_key = "falcon_heavy_expendable"
    launcher = get_launcher(launcher_key)
    print(f"Selected Launch Vehicle: {launcher.name} ({launcher.operator})")

    # 2. Define Mission Date Search Space
    dep_start = datetime(2026, 9, 1)
    dep_end = datetime(2026, 12, 31)
    arr_start = datetime(2027, 4, 1)
    arr_end = datetime(2027, 12, 31)

    grid_resolution = 120

    dep_mjd_vals = np.linspace(datetime_to_jd(dep_start) - JD_J2000, datetime_to_jd(dep_end) - JD_J2000, grid_resolution)
    arr_mjd_vals = np.linspace(datetime_to_jd(arr_start) - JD_J2000, datetime_to_jd(arr_end) - JD_J2000, grid_resolution)

    print(f"Departure Window: {dep_start.strftime('%Y-%m-%d')} to {dep_end.strftime('%Y-%m-%d')}")
    print(f"Arrival Window:   {arr_start.strftime('%Y-%m-%d')} to {arr_end.strftime('%Y-%m-%d')}")
    print(f"Scanning {grid_resolution}x{grid_resolution} trajectory options...")

    # 3. Compute Trajectories & Payload Capacities
    c3_grid, vinf_arr, tof_grid, total_dv, payload_grid = compute_porkchop(dep_mjd_vals, arr_mjd_vals, launcher)

    # 4. Extract Optimality Points
    # Minimum Heliocentric Delta-V
    min_dv_idx = np.nanargmin(total_dv)
    arr_opt_idx, dep_opt_idx = np.unravel_index(min_dv_idx, total_dv.shape)

    # Maximum Delivered Payload
    max_payload_idx = np.nanargmax(payload_grid)
    arr_maxpay_idx, dep_maxpay_idx = np.unravel_index(max_payload_idx, payload_grid.shape)

    opt_dep_mjd = dep_mjd_vals[dep_opt_idx]
    opt_arr_mjd = arr_mjd_vals[arr_opt_idx]
    opt_dep_dt = jd_to_datetime(opt_dep_mjd + JD_J2000)
    opt_arr_dt = jd_to_datetime(opt_arr_mjd + JD_J2000)
    tof_opt_sec = (opt_arr_mjd - opt_dep_mjd) * 86400.0

    print("\n" + "-" * 75)
    print(" OPTIMAL TRANSFER OPPORTUNITY IDENTIFIED (MINIMUM TOTAL DELTA-V)")
    print("-" * 75)
    print(f"Optimal Launch Date:        {opt_dep_dt.strftime('%Y-%m-%d')}")
    print(f"Optimal Arrival Date:       {opt_arr_dt.strftime('%Y-%m-%d')}")
    print(f"Optimal Time of Flight:     {tof_grid[arr_opt_idx, dep_opt_idx]:.1f} days")
    print(f"Departure C3 Launch Energy: {c3_grid[arr_opt_idx, dep_opt_idx]:.2f} km^2/s^2")
    print(f"Mars Arrival v_infinity:    {vinf_arr[arr_opt_idx, dep_opt_idx]:.2f} km/s")
    print(f"Total Heliocentric dV:      {total_dv[arr_opt_idx, dep_opt_idx]:.2f} km/s")
    print(f"Delivered Science Payload:  {payload_grid[arr_opt_idx, dep_opt_idx]:.1f} kg")
    print("-" * 75)
    print(f"Max Delivered Payload Case: {payload_grid[arr_maxpay_idx, dep_maxpay_idx]:.1f} kg (Launch C3: {c3_grid[arr_maxpay_idx, dep_maxpay_idx]:.2f} km^2/s^2)")
    print("-" * 75)

    # 5. Integrate Precision Transfer Arc: RK4 vs. Adaptive RK45 Benchmark
    r_earth_dep, _ = get_planet_state('earth', opt_dep_mjd, frame='ecliptic')
    r_mars_arr, _ = get_planet_state('mars', opt_arr_mjd, frame='ecliptic')
    r_mars_dep, _ = get_planet_state('mars', opt_dep_mjd, frame='ecliptic')
    r_earth_arr, _ = get_planet_state('earth', opt_arr_mjd, frame='ecliptic')

    v_sc_dep, _ = solve_lambert(r_earth_dep, r_mars_arr, tof_opt_sec, MU_SUN, prograde=True)
    initial_state = np.concatenate([r_earth_dep, v_sc_dep])

    # 5A. Fixed-step RK4 Propagation (Baseline dt = 1800s)
    dt = 1800.0
    num_steps_rk4 = int(np.ceil(tof_opt_sec / dt))
    print(f"\n[1/2] Propagating via Fixed-Step RK4 (dt={dt/60:.0f}m, {num_steps_rk4:,} steps)...")

    t0_wall = time.perf_counter()
    states_rk4 = np.zeros((num_steps_rk4 + 1, 6))
    states_rk4[0] = initial_state
    curr_t = 0.0
    curr_s = initial_state.copy()

    for i in range(num_steps_rk4):
        h = min(dt, tof_opt_sec - curr_t)
        curr_s = rk4_step(solar_gravity_derivs, curr_t, curr_s, h)
        curr_t += h
        states_rk4[i + 1] = curr_s
    rk4_wall = time.perf_counter() - t0_wall
    miss_km_rk4 = np.linalg.norm(states_rk4[-1, 0:3] - r_mars_arr) / 1000.0
    print(f"      Completed in {rk4_wall:.4f} s | Mars Miss Distance: {miss_km_rk4:.2f} km")

    # 5B. Adaptive-step RK45 Propagation (Dormand-Prince)
    print("\n[2/2] Propagating via Adaptive RK45 (rtol=1e-9, atol=1e-12)...")
    t0_wall = time.perf_counter()
    times_rk45, states_rk45 = rk45_adaptive(
        derivs_func=solar_gravity_derivs,
        t_span=(0.0, tof_opt_sec),
        y0=initial_state,
        rtol=1e-9,
        atol=1e-12,
        h_init=3600.0,
        h_min=1.0,
        h_max=86400.0 * 5.0,  # Allow up to 5-day step size during smooth cruise
    )
    rk45_wall = time.perf_counter() - t0_wall
    num_steps_rk45 = len(times_rk45)
    miss_km_rk45 = np.linalg.norm(states_rk45[-1, 0:3] - r_mars_arr) / 1000.0
    print(f"      Completed in {rk45_wall:.4f} s ({num_steps_rk45:,} steps) | Mars Miss Distance: {miss_km_rk45:.2f} km")

    # Performance comparison printout
    speedup = rk4_wall / rk45_wall if rk45_wall > 0 else 0.0
    step_reduc = (1.0 - (num_steps_rk45 / num_steps_rk4)) * 100.0
    print("\n" + "-" * 75)
    print(f"{'Performance Metric':<28} | {'Fixed RK4 (dt=30m)':<20} | {'Adaptive RK45':<20}")
    print("-" * 75)
    print(f"{'Wall-Clock Runtime (s)':<28} | {rk4_wall:<20.4f} | {rk45_wall:<20.4f}")
    print(f"{'Total Steps Taken':<28} | {num_steps_rk4:<20,} | {num_steps_rk45:<20,}")
    print(f"{'Mars Arrival Miss (km)':<28} | {miss_km_rk4:<20.2f} | {miss_km_rk45:<20.2f}")
    print("-" * 75)
    print(f"Runtime Speedup        : {speedup:.2f}x faster")
    print(f"Step Count Reduction   : {step_reduc:.2f}% fewer integration evaluations")
    print("-" * 75)

    # Use RK45 trajectory states for plotting
    states = states_rk45

    # 6. Visualization
    dep_dt_list = [jd_to_datetime(m + JD_J2000) for m in dep_mjd_vals]
    arr_dt_list = [jd_to_datetime(m + JD_J2000) for m in arr_mjd_vals]
    dep_num_grid, arr_num_grid = np.meshgrid(mdates.date2num(dep_dt_list), mdates.date2num(arr_dt_list))

    # =========================================================================
    # WINDOW 1: Tri-Panel Mission Design (C3, Delivered Payload, Total dV)
    # =========================================================================
    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20.0, 6.5), num="Window 1: Earth-Mars Porkchop & Payload Capacity")

    # Panel 1: C3 Energy & TOF
    c3_levels = [9, 10, 12, 15, 20, 25, 30, 40, 50, 70]
    cs1 = ax1.contourf(dep_num_grid, arr_num_grid, c3_grid, levels=c3_levels, cmap='Blues_r', extend='max')
    cbar1 = plt.colorbar(cs1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label(r'Departure $C_3$ [$\mathrm{km}^2/\mathrm{s}^2$]', fontsize=9)

    lines1 = ax1.contour(dep_num_grid, arr_num_grid, c3_grid, levels=c3_levels, colors='navy', linewidths=0.8)
    ax1.clabel(lines1, inline=True, fontsize=8, fmt=r'$C_3$=%d')

    tof_levels = [150, 200, 250, 300, 350, 400]
    cs_tof = ax1.contour(dep_num_grid, arr_num_grid, tof_grid, levels=tof_levels, colors='dimgray', linestyles='--', linewidths=1.0)
    ax1.clabel(cs_tof, inline=True, fontsize=8, fmt='%d d')

    ax1.plot(mdates.date2num(opt_dep_dt), mdates.date2num(opt_arr_dt), 'r*', markersize=13, label='Min $\\Delta v$')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.yaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.set_xlabel('Earth Departure Date', fontweight='bold')
    ax1.set_ylabel('Mars Arrival Date', fontweight='bold')
    ax1.set_title(r'Departure Energy ($C_3$) & Flight Time', fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left')

    # Panel 2: Deliverable Science Payload Mass
    pay_valid = payload_grid[~np.isnan(payload_grid)]
    if len(pay_valid) > 0 and np.max(pay_valid) > 0:
        pay_max = np.max(pay_valid)
        pay_levels = np.linspace(max(0.0, pay_max * 0.4), pay_max, 10)
        cs2 = ax2.contourf(dep_num_grid, arr_num_grid, payload_grid, levels=pay_levels, cmap='viridis', extend='both')
        cbar2 = plt.colorbar(cs2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('Payload Mass [kg]', fontsize=9)

        lines2 = ax2.contour(dep_num_grid, arr_num_grid, payload_grid, levels=pay_levels, colors='white', linewidths=0.6)
        ax2.clabel(lines2, inline=True, fontsize=8, fmt='%d kg')

    ax2.plot(mdates.date2num(opt_dep_dt), mdates.date2num(opt_arr_dt), 'r*', markersize=13, label='Min $\\Delta v$')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.yaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.set_xlabel('Earth Departure Date', fontweight='bold')
    ax2.set_ylabel('Mars Arrival Date', fontweight='bold')
    ax2.set_title(f'Deliverable Payload ({launcher.name})', fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left')

    # Panel 3: Total Delta-V & Mars v_inf
    dv_levels = [5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0, 12.0]
    cs3 = ax3.contourf(dep_num_grid, arr_num_grid, total_dv, levels=dv_levels, cmap='magma_r', extend='max')
    cbar3 = plt.colorbar(cs3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label(r'Total $\Delta v$ (Dep + Arr) [km/s]', fontsize=9)

    lines3 = ax3.contour(dep_num_grid, arr_num_grid, total_dv, levels=dv_levels, colors='black', linewidths=0.8)
    ax3.clabel(lines3, inline=True, fontsize=8, fmt=r'%.1f')

    vinf_levels = [2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
    cs_vinf = ax3.contour(dep_num_grid, arr_num_grid, vinf_arr, levels=vinf_levels, colors='cyan', linestyles=':', linewidths=1.1)
    ax3.clabel(cs_vinf, inline=True, fontsize=8, fmt=r'$v_\infty$=%.1f')

    ax3.plot(mdates.date2num(opt_dep_dt), mdates.date2num(opt_arr_dt), 'b*', markersize=13, label='Min $\\Delta v$')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax3.yaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax3.set_xlabel('Earth Departure Date', fontweight='bold')
    ax3.set_ylabel('Mars Arrival Date', fontweight='bold')
    ax3.set_title('Total $\\Delta v$ Budget & Mars $v_\\infty$', fontsize=11, fontweight='bold')
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='upper left')

    fig1.suptitle(f'Earth-Mars Mission Optimization — {launcher.name}', fontsize=14, fontweight='bold')
    fig1.tight_layout()

    # =========================================================================
    # WINDOW 2: Propagated Heliocentric Trajectory Arc
    # =========================================================================
    fig2, ax4 = plt.subplots(figsize=(8.5, 8.0), num="Window 2: Heliocentric Trajectory Arc")

    t_earth_orbit = np.linspace(opt_dep_mjd, opt_dep_mjd + 365.25, 200)
    t_mars_orbit = np.linspace(opt_dep_mjd, opt_dep_mjd + 686.98, 200)
    earth_orbit = np.array([get_planet_state('earth', t)[0] for t in t_earth_orbit]) / AU
    mars_orbit = np.array([get_planet_state('mars', t)[0] for t in t_mars_orbit]) / AU

    sc_pos_au = states[:, 0:3] / AU

    ax4.plot(earth_orbit[:, 0], earth_orbit[:, 1], 'b--', alpha=0.5, label='Earth Orbit (1.0 AU)')
    ax4.plot(mars_orbit[:, 0], mars_orbit[:, 1], 'r--', alpha=0.5, label='Mars Orbit (1.52 AU)')
    ax4.plot(sc_pos_au[:, 0], sc_pos_au[:, 1], color='forestgreen', linewidth=2.5, label=f'RK45 Transfer Arc ({num_steps_rk45} pts)')

    ax4.scatter(0.0, 0.0, color='gold', s=200, edgecolors='black', label='Sun', zorder=5)
    ax4.scatter(r_earth_dep[0] / AU, r_earth_dep[1] / AU, color='dodgerblue', s=100, edgecolors='black', label='Earth @ Dep', zorder=5)
    ax4.scatter(r_mars_arr[0] / AU, r_mars_arr[1] / AU, color='crimson', s=100, edgecolors='black', label='Mars @ Arr', zorder=5)
    ax4.scatter(r_mars_dep[0] / AU, r_mars_dep[1] / AU, color='crimson', s=40, alpha=0.35, edgecolors='black', label='Mars @ Dep', zorder=4)
    ax4.scatter(r_earth_arr[0] / AU, r_earth_arr[1] / AU, color='dodgerblue', s=40, alpha=0.35, edgecolors='black', label='Earth @ Arr', zorder=4)

    ax4.set_xlim([-1.8, 1.8])
    ax4.set_ylim([-1.8, 1.8])
    ax4.set_aspect('equal', adjustable='box')
    ax4.set_xlabel('Heliocentric X [AU]', fontweight='bold')
    ax4.set_ylabel('Heliocentric Y [AU]', fontweight='bold')
    ax4.set_title(
        f"Targeted Earth-Mars Transfer Orbit Arc (RK45)\n"
        f"Launch: {opt_dep_dt.strftime('%Y-%m-%d')} | Arrival: {opt_arr_dt.strftime('%Y-%m-%d')} | TOF: {tof_grid[arr_opt_idx, dep_opt_idx]:.1f} d\n"
        f"Delivered Science Payload ({launcher.name}): {payload_grid[arr_opt_idx, dep_opt_idx]:.1f} kg",
        fontsize=10.5,
        fontweight='bold'
    )
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='lower left', fontsize=8.5, ncol=2)

    fig2.tight_layout()
    plt.show()


main = run

if __name__ == '__main__':
    run()
    