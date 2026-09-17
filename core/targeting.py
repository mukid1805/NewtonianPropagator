r"""Orbital Targeting, Differential Corrections, and Relative Motion Subsystems.

Provides:
- `solve_complete_rendezvous`: Two-impulse targeting solver using STM differential correction.
- `eci_to_lvlh`: Kinematic coordinate conversion from ECI to Target-centred Hill/LVLH frame.
- `plot_rendezvous_3d`: 3D Cartesian visualizer for transfer trajectories and burn quivers.
- `plot_lvlh_docking`: 2D multi-panel relative approach trajectory visualizer.
"""

from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np

from core.constants import R_EARTH
from core.propagator import SpacecraftPropagator
from core.lambert import solve_lambert


def solve_complete_rendezvous(
    engine: SpacecraftPropagator,
    r0_chaser: np.ndarray,
    v0_chaser: np.ndarray,
    r0_target: np.ndarray,
    v0_target: np.ndarray,
    tof: float,
    tol_m: float = 1e-2,
    max_iter: int = 15,
    method: str = "rk4",
    dt: float = 10.0,
) -> Dict[str, object]:
    r"""Compute departure and arrival burns for a complete two-impulse rendezvous sequence.

    Solves the two-point boundary value problem by computing the departure
    impulse $\Delta \mathbf{v}_0$ via Newton-Raphson differential correction on
    the upper-right block $\Phi_{rv} = \partial \mathbf{r}(t_f)/\partial \mathbf{v}_0$:

    $$\Delta \mathbf{v}^{(k+1)} = \Delta \mathbf{v}^{(k)} + \Phi_{rv}^{-1} \left(\mathbf{r}_t(t_f) - \mathbf{r}_c^{(k)}(t_f)\right)$$

    seeded with an unperturbed Lambert boundary-value warm-start, and the
    terminal braking/matching manoeuvre:

    $$\Delta \mathbf{v}_f = \mathbf{v}_t(t_f) - \mathbf{v}_c^-(t_f)$$

    Args:
        engine: SpacecraftPropagator instance with `propagate_with_stm` support.
        r0_chaser: Chaser initial position vector in the ECI frame of shape `(3,)` in metres.
        v0_chaser: Chaser initial velocity vector in the ECI frame of shape `(3,)` in m/s.
        r0_target: Target initial position vector in the ECI frame of shape `(3,)` in metres.
        v0_target: Target initial velocity vector in the ECI frame of shape `(3,)` in m/s.
        tof: Planned transfer time-of-flight $\Delta t = t_f - t_0$ in seconds.
        tol_m: Position convergence tolerance in metres. Defaults to 0.01 (1 cm).
        max_iter: Maximum permitted Newton-Raphson correction iterations. Defaults to 15.
        method: Numerical integration routine ('rk4' or 'rk45'). Defaults to 'rk4'.
        dt: Integration time step size in seconds. Defaults to 10.0.

    Returns:
        Dict containing:
            - 'delta_v0' (np.ndarray): Departure burn vector in the ECI frame [m/s].
            - 'delta_v0_mag' (float): Magnitude of departure impulse [m/s].
            - 'delta_vf' (np.ndarray): Arrival braking burn vector in the ECI frame [m/s].
            - 'delta_vf_mag' (float): Magnitude of arrival braking impulse [m/s].
            - 'delta_v_total' (float): Total mission budget $\|\Delta \mathbf{v}_0\| + \|\Delta \mathbf{v}_f\|$ [m/s].
            - 'final_miss_m' (float): Terminal position miss distance in metres.
            - 'relative_vf_mag' (float): Residual relative velocity magnitude after docking burn [m/s].
            - 'iterations' (int): Total iterations required to achieve convergence.
            - 't_transfer' (np.ndarray): Discrete solution time points of shape `(N,)` [s].
            - 'chaser_transfer_states' (np.ndarray): Chaser integrated states of shape `(N, 6)` [m, m/s].
            - 'target_states' (np.ndarray): Target propagated states of shape `(N, 6)` [m, m/s].

    Raises:
        RuntimeError: If differential correction diverges, encounters a caustic singularity,
            or exceeds max_iter without reaching tolerance.
    """
    r0_c = np.asarray(r0_chaser, dtype=np.float64)
    v0_c = np.asarray(v0_chaser, dtype=np.float64)
    r0_t = np.asarray(r0_target, dtype=np.float64)
    v0_t = np.asarray(v0_target, dtype=np.float64)

    # 1. Propagate Target unperturbed to determine rendezvous waypoint
    t_target, target_states = engine.propagate(
        r0=r0_t,
        v0=v0_t,
        t_span=tof,
        dt=dt,
        method=method,
    )
    rf_target = target_states[-1, 0:3]
    vf_target = target_states[-1, 3:6]

    # 2. Initialise departure velocity using two-body Lambert warm-start
    try:
        v0_lambert, _ = solve_lambert(r0_c, rf_target, tof, mu=engine.mu, prograde=True)
        v0_guess = v0_lambert.copy()
    except Exception:
        v0_guess = v0_c.copy()

    # 3. Newton-Raphson differential correction on departure velocity
    miss_distance = float("inf")
    converged = False
    final_states = None
    final_times = None

    for iteration in range(1, max_iter + 1):
        times, states, stms = engine.propagate_with_stm(
            r0=r0_c,
            v0=v0_guess,
            t_span=tof,
            dt=dt,
            method=method,
        )

        rf_chaser = states[-1, 0:3]
        error_pos = rf_target - rf_chaser
        miss_distance = float(np.linalg.norm(error_pos))

        if miss_distance <= tol_m:
            converged = True
            final_states = states
            final_times = times
            break

        phi_rv = stms[-1, 0:3, 3:6]  # Upper-right 3x3 block: dr(tf) / dv(t0)

        cond = np.linalg.cond(phi_rv)
        if cond > 1e12:
            raise RuntimeError(
                f"Ill-conditioned Phi_rv matrix (condition number = {cond:.2e}). "
                "Time of flight is near an orbital half-period caustic singularity."
            )

        # Apply correction update
        dv_corr = np.linalg.solve(phi_rv, error_pos)
        v0_guess += dv_corr

    if not converged:
        raise RuntimeError(
            f"Targeting solver failed to converge after {max_iter} iterations. "
            f"Final miss: {miss_distance:.3e} m (tolerance: {tol_m:.3e} m)."
        )

    # 4. Compute departure impulse
    delta_v0 = v0_guess - v0_c
    delta_v0_mag = float(np.linalg.norm(delta_v0))

    # 5. Compute arrival braking impulse
    vf_chaser_arrival = final_states[-1, 3:6]
    delta_vf = vf_target - vf_chaser_arrival
    delta_vf_mag = float(np.linalg.norm(delta_vf))

    residual_v = (vf_chaser_arrival + delta_vf) - vf_target

    return {
        "delta_v0": delta_v0,
        "delta_v0_mag": delta_v0_mag,
        "delta_vf": delta_vf,
        "delta_vf_mag": delta_vf_mag,
        "delta_v_total": delta_v0_mag + delta_vf_mag,
        "final_miss_m": miss_distance,
        "relative_vf_mag": float(np.linalg.norm(residual_v)),
        "iterations": iteration,
        "t_transfer": final_times,
        "chaser_transfer_states": final_states,
        "target_states": target_states,
    }

def eci_to_lvlh(
    r_target: np.ndarray,
    v_target: np.ndarray,
    r_chaser: np.ndarray,
    v_chaser: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Transform instantaneous Chaser state into Target-centred Hill/LVLH rotating coordinates.

    The Local-Vertical Local-Horizontal (LVLH) orthonormal triad is defined by:

    $$\hat{\mathbf{e}}_r = \frac{\mathbf{r}_t}{\|\mathbf{r}_t\|}, \quad \hat{\mathbf{e}}_c = \frac{\mathbf{r}_t \times \mathbf{v}_t}{\|\mathbf{r}_t \times \mathbf{v}_t\|}, \quad \hat{\mathbf{e}}_t = \hat{\mathbf{e}}_c \times \hat{\mathbf{e}}_r$$

    Args:
        r_target: Target position vector array in ECI of shape `(N, 3)` or `(3,)` [m].
        v_target: Target velocity vector array in ECI of shape `(N, 3)` or `(3,)` [m/s].
        r_chaser: Chaser position vector array in ECI of shape `(N, 3)` or `(3,)` [m].
        v_chaser: Chaser velocity vector array in ECI of shape `(N, 3)` or `(3,)` [m/s].

    Returns:
        Tuple containing:
            - r_lvlh (np.ndarray): Relative position $[x_{\text{radial}}, y_{\text{in-track}}, z_{\text{cross-track}}]^T$ [m].
            - v_lvlh (np.ndarray): Relative velocity with respect to the rotating frame [m/s].
    """
    rt = np.atleast_2d(r_target)
    vt = np.atleast_2d(v_target)
    rc = np.atleast_2d(r_chaser)
    vc = np.atleast_2d(v_chaser)

    n_points = rt.shape[0]
    r_lvlh = np.zeros((n_points, 3), dtype=np.float64)
    v_lvlh = np.zeros((n_points, 3), dtype=np.float64)

    for i in range(n_points):
        r_t_i = rt[i]
        v_t_i = vt[i]
        r_c_i = rc[i]
        v_c_i = vc[i]

        r_mag = np.linalg.norm(r_t_i)
        h_vec = np.cross(r_t_i, v_t_i)
        h_mag = np.linalg.norm(h_vec)

        # Orthonormal basis unit vectors
        e_r = r_t_i / r_mag
        e_c = h_vec / h_mag
        e_t = np.cross(e_c, e_r)

        # Rotation matrix ECI -> LVLH
        c_i_to_lvlh = np.vstack((e_r, e_t, e_c))

        # Relative position
        dr_eci = r_c_i - r_t_i
        r_lvlh[i] = c_i_to_lvlh @ dr_eci

        # Frame angular velocity omega = (r x v) / r^2
        omega_vec = h_vec / (r_mag**2)

        # Transport theorem: v_rel = C * (dv_eci - omega x dr_eci)
        dv_eci = v_c_i - v_t_i
        v_lvlh[i] = c_i_to_lvlh @ (dv_eci - np.cross(omega_vec, dr_eci))

    if r_target.ndim == 1:
        return r_lvlh[0], v_lvlh[0]
    return r_lvlh, v_lvlh


def plot_rendezvous_3d(
    chaser_states: np.ndarray,
    target_states: np.ndarray,
    delta_v0: np.ndarray,
    delta_vf: np.ndarray,
    burn_scale: float = 20.0,
    title: str = "Two-Impulse Orbital Rendezvous Trajectory",
) -> None:
    r"""Plot the 3D transfer trajectory, target orbit, and burn quivers in ECI space.

    Args:
        chaser_states: Chaser transfer state trajectory array of shape `(N, 6)` in metres and m/s.
        target_states: Target orbit state trajectory array of shape `(N, 6)` in metres and m/s.
        delta_v0: Initial departure burn vector in the ECI frame of shape `(3,)` in m/s.
        delta_vf: Final arrival braking burn vector in the ECI frame of shape `(3,)` in m/s.
        burn_scale: Display scale factor to convert velocity [m/s] into visible vector arrows [km]. Defaults to 20.0.
        title: Matplotlib figure title string. Defaults to 'Two-Impulse Orbital Rendezvous Trajectory'.
    """
    c_arr = np.asarray(chaser_states, dtype=np.float64)
    t_arr = np.asarray(target_states, dtype=np.float64)

    x_c, y_c, z_c = c_arr[:, 0] / 1000.0, c_arr[:, 1] / 1000.0, c_arr[:, 2] / 1000.0
    x_t, y_t, z_t = t_arr[:, 0] / 1000.0, t_arr[:, 1] / 1000.0, t_arr[:, 2] / 1000.0

    fig = plt.figure(figsize=(11.0, 9.0))
    ax = fig.add_subplot(111, projection="3d")

    # Earth wireframe sphere
    r_earth_km = R_EARTH / 1000.0
    u, v = np.mgrid[0 : 2 * np.pi : 30j, 0 : np.pi : 15j]
    ax.plot_wireframe(
        r_earth_km * np.cos(u) * np.sin(v),
        r_earth_km * np.sin(u) * np.sin(v),
        r_earth_km * np.cos(v),
        color="royalblue",
        alpha=0.20,
        linewidth=0.7,
        label="Earth",
    )

    # Target trajectory
    ax.plot(x_t, y_t, z_t, color="darkorange", linestyle="--", linewidth=1.8, label="Target Orbit")
    ax.scatter(x_t[0], y_t[0], z_t[0], color="darkorange", marker="o", s=40, label="Target (t0)")

    # Chaser transfer trajectory
    ax.plot(x_c, y_c, z_c, color="crimson", linewidth=2.0, label="Chaser Transfer Arc")
    ax.scatter(x_c[0], y_c[0], z_c[0], color="forestgreen", marker="^", s=70, label="Chaser Ignition (t0)")
    ax.scatter(x_c[-1], y_c[-1], z_c[-1], color="magenta", marker="*", s=140, label="Rendezvous (tf)")

    # Departure burn quiver
    dv0_km = (delta_v0 * burn_scale) / 1000.0
    ax.quiver(
        x_c[0], y_c[0], z_c[0],
        dv0_km[0], dv0_km[1], dv0_km[2],
        color="limegreen",
        linewidth=2.0,
        arrow_length_ratio=0.25,
        label=r"$\Delta \mathbf{v}_0$ Departure Burn",
    )

    # Arrival burn quiver
    dvf_km = (delta_vf * burn_scale) / 1000.0
    ax.quiver(
        x_c[-1], y_c[-1], z_c[-1],
        dvf_km[0], dvf_km[1], dvf_km[2],
        color="deepskyblue",
        linewidth=2.0,
        arrow_length_ratio=0.25,
        label=r"$\Delta \mathbf{v}_f$ Arrival Burn",
    )

    all_x = np.concatenate([x_c, x_t])
    all_y = np.concatenate([y_c, y_t])
    all_z = np.concatenate([z_c, z_t])

    max_span = max(all_x.max() - all_x.min(), all_y.max() - all_y.min(), all_z.max() - all_z.min()) * 0.5
    mid_x = (all_x.max() + all_x.min()) * 0.5
    mid_y = (all_y.max() + all_y.min()) * 0.5
    mid_z = (all_z.max() + all_z.min()) * 0.5

    ax.set_xlim(mid_x - max_span, mid_x + max_span)
    ax.set_ylim(mid_y - max_span, mid_y + max_span)
    ax.set_zlim(mid_z - max_span, mid_z + max_span)

    ax.set_xlabel("ECI X [km]", labelpad=8)
    ax.set_ylabel("ECI Y [km]", labelpad=8)
    ax.set_zlabel("ECI Z [km]", labelpad=8)
    ax.set_title(title, pad=15, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.85)
    plt.tight_layout()
    plt.show()


def plot_lvlh_docking(
    r_lvlh: np.ndarray,
    title: str = "Relative Rendezvous Motion in Target LVLH Frame",
) -> None:
    r"""Plot multi-panel relative motion profiles in the Target-centred Hill/LVLH frame.

    Args:
        r_lvlh: Relative position array $[x_{\text{radial}}, y_{\text{in-track}}, z_{\text{cross-track}}]^T$
            of shape `(N, 3)` in metres.
        title: Figure title string. Defaults to 'Relative Rendezvous Motion in Target LVLH Frame'.
    """
    pos = np.asarray(r_lvlh, dtype=np.float64)
    x_radial = pos[:, 0] / 1000.0       # Radial [km]
    y_intrack = pos[:, 1] / 1000.0      # In-track [km]
    z_crosstrack = pos[:, 2] / 1000.0   # Cross-track [km]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 6.0))

    # Panel 1: In-track vs Radial (Orbital Plane)
    ax1.plot(y_intrack, x_radial, color="crimson", linewidth=1.8, label="Chaser Approach Arc")
    ax1.scatter([0.0], [0.0], color="darkorange", marker="o", s=80, label="Target Center (0,0)")
    ax1.scatter(y_intrack[0], x_radial[0], color="forestgreen", marker="^", s=70, label="Departure (t0)")
    ax1.scatter(y_intrack[-1], x_radial[-1], color="magenta", marker="*", s=130, label="Arrival (tf)")

    ax1.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
    ax1.axvline(0.0, color="gray", linestyle=":", linewidth=0.8)
    ax1.set_xlabel("In-Track (V-Bar) [km]")
    ax1.set_ylabel("Radial (R-Bar) [km]")
    ax1.set_title("In-Plane Motion (V-Bar vs R-Bar)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Panel 2: In-track vs Cross-Track (Out-of-Plane)
    ax2.plot(y_intrack, z_crosstrack, color="royalblue", linewidth=1.8, label="Chaser Approach Arc")
    ax2.scatter([0.0], [0.0], color="darkorange", marker="o", s=80, label="Target Center (0,0)")
    ax2.scatter(y_intrack[0], z_crosstrack[0], color="forestgreen", marker="^", s=70, label="Departure (t0)")
    ax2.scatter(y_intrack[-1], z_crosstrack[-1], color="magenta", marker="*", s=130, label="Arrival (tf)")

    ax2.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
    ax2.axvline(0.0, color="gray", linestyle=":", linewidth=0.8)
    ax2.set_xlabel("In-Track (V-Bar) [km]")
    ax2.set_ylabel("Cross-Track (H-Bar) [km]")
    ax2.set_title("Out-of-Plane Motion (V-Bar vs H-Bar)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()