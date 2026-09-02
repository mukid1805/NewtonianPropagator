"""
Scenario 8: Multi-Leg Interplanetary Gravity Assist Optimizer.
Performs a 3D grid-search across Launch, Flyby, and Arrival dates to find
the absolute minimum Delta-V Earth-Venus-Mars (EVM) trajectory, and benchmarks
Fixed-Step RK4 against Adaptive RK45 (Dormand-Prince) on both heliocentric legs.
"""
import sys
import time
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

from core.time import datetime_to_jd, jd_to_datetime, JD_J2000
from core.ephemeris import get_planet_state, MU_SUN, AU
from core.lambert import solve_lambert
from core.flyby import evaluate_unpowered_flyby
from core.integrators import rk4_step, rk45_adaptive


def solar_gravity_derivs(t: float, state: np.ndarray) -> np.ndarray:
    """Heliocentric point-mass gravity equations of motion."""
    r = state[0:3]
    v = state[3:6]
    a = -(MU_SUN / (np.linalg.norm(r) ** 3)) * r
    return np.concatenate([v, a])


def propagate_leg_rk4(r0: np.ndarray, v0: np.ndarray, tof_sec: float, dt: float = 1800.0) -> np.ndarray:
    """Propagates a heliocentric transfer leg via fixed-step RK4."""
    num_steps = int(np.ceil(tof_sec / dt))
    states = np.zeros((num_steps + 1, 6), dtype=np.float64)
    states[0] = np.concatenate([r0, v0])

    curr_t = 0.0
    curr_state = states[0].copy()

    for i in range(num_steps):
        h = min(dt, tof_sec - curr_t)
        curr_state = rk4_step(solar_gravity_derivs, curr_t, curr_state, h)
        curr_t += h
        states[i + 1] = curr_state

    return states


def propagate_leg_rk45(
    r0: np.ndarray,
    v0: np.ndarray,
    tof_sec: float,
    rtol: float = 1e-9,
    atol: float = 1e-12
):
    """Propagates a heliocentric transfer leg via adaptive-step RK45 (Dormand-Prince)."""
    initial_state = np.concatenate([r0, v0])
    times, states = rk45_adaptive(
        derivs_func=solar_gravity_derivs,
        t_span=(0.0, tof_sec),
        y0=initial_state,
        rtol=rtol,
        atol=atol,
        h_init=3600.0,
        h_min=1.0,
        h_max=86400.0 * 5.0,  # Up to 5-day step size during smooth cruise
    )
    return times, states


def optimize_evm_trajectory():
    """
    Scans a 3D grid of Departure Dates, Leg 1 TOFs, and Leg 2 TOFs
    to find the minimum total Delta-V mission.
    """
    # 1. Define Search Grid Parameters
    dep_start_mjd = datetime_to_jd(datetime(2028, 5, 1)) - JD_J2000
    dep_end_mjd = datetime_to_jd(datetime(2028, 12, 31)) - JD_J2000
    dep_step = 5.0  # days

    tof1_min, tof1_max, tof1_step = 100.0, 200.0, 5.0
    tof2_min, tof2_max, tof2_step = 150.0, 400.0, 5.0

    dep_mjds = np.arange(dep_start_mjd, dep_end_mjd + dep_step, dep_step)
    tof1_days = np.arange(tof1_min, tof1_max + tof1_step, tof1_step)
    tof2_days = np.arange(tof2_min, tof2_max + tof2_step, tof2_step)

    total_combinations = len(dep_mjds) * len(tof1_days) * len(tof2_days)
    print(f"Search Grid Size: {total_combinations} trajectories...")

    ephem_cache = {'earth': {}, 'venus': {}, 'mars': {}}

    best_cost = np.inf
    best_mission = None
    count = 0

    t_start = time.time()

    # 2. Execute Grid Search
    for mjd_dep in dep_mjds:
        if mjd_dep not in ephem_cache['earth']:
            ephem_cache['earth'][mjd_dep] = get_planet_state('earth', mjd_dep)
        r_earth, v_earth = ephem_cache['earth'][mjd_dep]

        for t1 in tof1_days:
            mjd_flyby = mjd_dep + t1
            if mjd_flyby not in ephem_cache['venus']:
                ephem_cache['venus'][mjd_flyby] = get_planet_state('venus', mjd_flyby)
            r_venus, v_venus = ephem_cache['venus'][mjd_flyby]

            try:
                v1_leg1, v2_leg1 = solve_lambert(r_earth, r_venus, t1 * 86400.0, MU_SUN, prograde=True)
            except ValueError:
                count += len(tof2_days)
                continue

            v_inf_dep = np.linalg.norm(v1_leg1 - v_earth)
            v_inf_venus_in = v2_leg1 - v_venus

            for t2 in tof2_days:
                count += 1
                if count % 2000 == 0:
                    sys.stdout.write(f"\rOptimizing... {count}/{total_combinations} ({(count/total_combinations)*100:.1f}%)")
                    sys.stdout.flush()

                mjd_arr = mjd_flyby + t2
                if mjd_arr not in ephem_cache['mars']:
                    ephem_cache['mars'][mjd_arr] = get_planet_state('mars', mjd_arr)
                r_mars, v_mars = ephem_cache['mars'][mjd_arr]

                try:
                    v1_leg2, v2_leg2 = solve_lambert(r_venus, r_mars, t2 * 86400.0, MU_SUN, prograde=True)
                except ValueError:
                    continue

                v_inf_venus_out = v1_leg2 - v_venus
                v_inf_arr = np.linalg.norm(v2_leg2 - v_mars)

                flyby_res = evaluate_unpowered_flyby(v_inf_venus_in, v_inf_venus_out, 'venus', min_altitude_km=300.0)

                if not flyby_res['is_feasible']:
                    continue

                total_dv = v_inf_dep + v_inf_arr + flyby_res['dv_powered']

                if total_dv < best_cost:
                    best_cost = total_dv
                    best_mission = {
                        'mjd_dep': mjd_dep,
                        'mjd_flyby': mjd_flyby,
                        'mjd_arr': mjd_arr,
                        'tof1': t1,
                        'tof2': t2,
                        'flyby_res': flyby_res,
                        'v_inf_dep': v_inf_dep,
                        'v_inf_arr': v_inf_arr,
                        'total_dv': total_dv,
                        'r_earth': r_earth, 'r_venus': r_venus, 'r_mars': r_mars,
                        'v1_leg1': v1_leg1, 'v1_leg2': v1_leg2
                    }

    t_end = time.time()
    sys.stdout.write(f"\rOptimization Complete in {t_end - t_start:.2f} seconds!            \n")
    return best_mission


def run():
    print("=" * 72)
    print("   AUTOMATED MULTI-LEG GRAVITY ASSIST OPTIMIZER (EARTH-VENUS-MARS)")
    print("=" * 72)

    # 1. Run Optimizer
    mission = optimize_evm_trajectory()

    if mission is None:
        print("ERROR: No valid free-return gravity assist found in the search window.")
        return

    dep_dt = jd_to_datetime(mission['mjd_dep'] + JD_J2000)
    flyby_dt = jd_to_datetime(mission['mjd_flyby'] + JD_J2000)
    arr_dt = jd_to_datetime(mission['mjd_arr'] + JD_J2000)

    print("\n" + "=" * 72)
    print("   OPTIMAL 'FREE' GRAVITY ASSIST IDENTIFIED")
    print("=" * 72)
    print(f"Earth Departure: {dep_dt.strftime('%Y-%m-%d')}")
    print(f"Venus Flyby:     {flyby_dt.strftime('%Y-%m-%d')}  (Leg 1 TOF: {mission['tof1']:.1f} d)")
    print(f"Mars Arrival:    {arr_dt.strftime('%Y-%m-%d')}  (Leg 2 TOF: {mission['tof2']:.1f} d)")
    print("-" * 72)

    fres = mission['flyby_res']
    c3 = mission['v_inf_dep'] ** 2

    print(f"Earth Launch C3 Energy:           {c3:.2f} km^2/s^2  (v_inf = {mission['v_inf_dep']:.2f} km/s)")
    print(f"Venus Inbound v_inf:             {fres['v_inf_in_mag']:.2f} km/s")
    print(f"Venus Outbound v_inf:            {fres['v_inf_out_mag']:.2f} km/s")
    print(f"Venus Hyperbolic Turn Angle (d): {fres['delta_angle_deg']:.2f} deg")
    print(f"Venus Flyby Periapsis Altitude:  {fres['h_p']:.1f} km (Min Safe: 300 km)")
    print(f"Venus Powered Correction dV:     {fres['dv_powered']:.3f} km/s")
    print(f"Mars Arrival v_inf:              {mission['v_inf_arr']:.2f} km/s")
    print(f"Total Mission Flight Time:       {mission['tof1'] + mission['tof2']:.1f} days")
    print("=" * 72)

    # -------------------------------------------------------------------------
    # 2. Numerical Trajectory Verification & RK4 vs. RK45 Benchmark
    # -------------------------------------------------------------------------
    tof1_sec = mission['tof1'] * 86400.0
    tof2_sec = mission['tof2'] * 86400.0
    dt_rk4 = 1800.0  # 30-minute fixed step

    print("\n--- Numerical Verification & Integrator Benchmark ---")

    # Leg 1: Fixed RK4
    t0 = time.perf_counter()
    states_leg1_rk4 = propagate_leg_rk4(mission['r_earth'], mission['v1_leg1'], tof1_sec, dt=dt_rk4)
    time_leg1_rk4 = time.perf_counter() - t0
    miss_venus_rk4 = np.linalg.norm(states_leg1_rk4[-1, 0:3] - mission['r_venus']) / 1000.0

    # Leg 1: Adaptive RK45
    t0 = time.perf_counter()
    times_leg1_rk45, states_leg1_rk45 = propagate_leg_rk45(mission['r_earth'], mission['v1_leg1'], tof1_sec)
    time_leg1_rk45 = time.perf_counter() - t0
    miss_venus_rk45 = np.linalg.norm(states_leg1_rk45[-1, 0:3] - mission['r_venus']) / 1000.0

    # Leg 2: Fixed RK4
    t0 = time.perf_counter()
    states_leg2_rk4 = propagate_leg_rk4(mission['r_venus'], mission['v1_leg2'], tof2_sec, dt=dt_rk4)
    time_leg2_rk4 = time.perf_counter() - t0
    miss_mars_rk4 = np.linalg.norm(states_leg2_rk4[-1, 0:3] - mission['r_mars']) / 1000.0

    # Leg 2: Adaptive RK45
    t0 = time.perf_counter()
    times_leg2_rk45, states_leg2_rk45 = propagate_leg_rk45(mission['r_venus'], mission['v1_leg2'], tof2_sec)
    time_leg2_rk45 = time.perf_counter() - t0
    miss_mars_rk45 = np.linalg.norm(states_leg2_rk45[-1, 0:3] - mission['r_mars']) / 1000.0

    total_time_rk4 = time_leg1_rk4 + time_leg2_rk4
    total_time_rk45 = time_leg1_rk45 + time_leg2_rk45
    total_steps_rk4 = len(states_leg1_rk4) + len(states_leg2_rk4)
    total_steps_rk45 = len(times_leg1_rk45) + len(times_leg2_rk45)
    speedup = total_time_rk4 / total_time_rk45 if total_time_rk45 > 0 else 0.0
    step_reduc = (1.0 - (total_steps_rk45 / total_steps_rk4)) * 100.0

    print(f"{'Metric':<30} | {'Fixed RK4 (dt=30m)':<20} | {'Adaptive RK45':<20}")
    print("-" * 75)
    print(f"{'Leg 1 (Earth->Venus) Runtime':<30} | {time_leg1_rk4:<20.4f} | {time_leg1_rk45:<20.4f}")
    print(f"{'Leg 1 Step Count':<30} | {len(states_leg1_rk4):<20,} | {len(times_leg1_rk45):<20,}")
    print(f"{'Venus Intercept Miss (km)':<30} | {miss_venus_rk4:<20.2f} | {miss_venus_rk45:<20.2f}")
    print("-" * 75)
    print(f"{'Leg 2 (Venus->Mars) Runtime':<30} | {time_leg2_rk4:<20.4f} | {time_leg2_rk45:<20.4f}")
    print(f"{'Leg 2 Step Count':<30} | {len(states_leg2_rk4):<20,} | {len(times_leg2_rk45):<20,}")
    print(f"{'Mars Intercept Miss (km)':<30} | {miss_mars_rk4:<20.2f} | {miss_mars_rk45:<20.2f}")
    print("-" * 75)
    print(f"{'Total Propagation Time (s)':<30} | {total_time_rk4:<20.4f} | {total_time_rk45:<20.4f}")
    print(f"{'Total Steps (Both Legs)':<30} | {total_steps_rk4:<20,} | {total_steps_rk45:<20,}")
    print(f"Overall Speedup                : {speedup:.2f}x faster")
    print(f"Step Count Reduction           : {step_reduc:.2f}% fewer evaluations")
    print("=" * 72)

    # Use RK45 states for trajectory plotting
    states_leg1 = states_leg1_rk45
    states_leg2 = states_leg2_rk45

    # -------------------------------------------------------------------------
    # 3. Plotting Multi-Leg Transfer Trajectory
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.0, 9.0), num="Scenario 8: Optimized EVM Gravity Assist")

    t_orb = np.linspace(0, 700, 300)
    venus_orb = np.array([get_planet_state('venus', mission['mjd_dep'] + t)[0] for t in t_orb]) / AU
    earth_orb = np.array([get_planet_state('earth', mission['mjd_dep'] + t)[0] for t in t_orb]) / AU
    mars_orb = np.array([get_planet_state('mars', mission['mjd_dep'] + t)[0] for t in t_orb]) / AU

    ax.plot(venus_orb[:, 0], venus_orb[:, 1], color='orange', linestyle=':', alpha=0.5, label='Venus Orbit (0.72 AU)')
    ax.plot(earth_orb[:, 0], earth_orb[:, 1], color='dodgerblue', linestyle='--', alpha=0.5, label='Earth Orbit (1.00 AU)')
    ax.plot(mars_orb[:, 0], mars_orb[:, 1], color='crimson', linestyle='--', alpha=0.5, label='Mars Orbit (1.52 AU)')

    pos1_au = states_leg1[:, 0:3] / AU
    pos2_au = states_leg2[:, 0:3] / AU
    ax.plot(pos1_au[:, 0], pos1_au[:, 1], color='darkcyan', linewidth=2.5,
            label=f'Leg 1: Earth -> Venus (RK45: {len(times_leg1_rk45)} pts)')
    ax.plot(pos2_au[:, 0], pos2_au[:, 1], color='purple', linewidth=2.5,
            label=f'Leg 2: Venus -> Mars (RK45: {len(times_leg2_rk45)} pts)')

    ax.scatter(0, 0, color='gold', s=200, edgecolors='black', label='Sun', zorder=5)
    ax.scatter(mission['r_earth'][0]/AU, mission['r_earth'][1]/AU, color='dodgerblue', s=100, edgecolors='black', label='Earth @ Launch', zorder=5)
    ax.scatter(mission['r_venus'][0]/AU, mission['r_venus'][1]/AU, color='orange', s=110, edgecolors='black', label='Venus @ Flyby', zorder=5)
    ax.scatter(mission['r_mars'][0]/AU, mission['r_mars'][1]/AU, color='crimson', s=100, edgecolors='black', label='Mars @ Arrival', zorder=5)

    ax.set_xlim([-1.8, 1.8])
    ax.set_ylim([-1.8, 1.8])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Heliocentric X [AU]', fontweight='bold')
    ax.set_ylabel('Heliocentric Y [AU]', fontweight='bold')

    title_str = (
        f"Optimized Earth-Venus-Mars Gravity Assist (Adaptive RK45)\n"
        f"Dep: {dep_dt.strftime('%b %d, %Y')} | Flyby: {flyby_dt.strftime('%b %d, %Y')} | Arr: {arr_dt.strftime('%b %d, %Y')}\n"
        f"Powered Flyby dV: {fres['dv_powered']:.3f} km/s | Alt: {fres['h_p']:.0f} km"
    )
    ax.set_title(title_str, fontsize=11, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower left', fontsize=8.5, ncol=2)

    plt.tight_layout()
    plt.show()


main = run

if __name__ == '__main__':
    run()
