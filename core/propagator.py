r"""
Unified Spacecraft Numerical Propagation Engine.

Supports 6-DOF $(\mathbf{r}, \mathbf{v})$ and 7-DOF $(\mathbf{r}, \mathbf{v}, m)$
orbital trajectory simulation under primary two-body gravity, zonal harmonics
($J_2$–$J_4$), third-body lunar perturbations, atmospheric drag, solar radiation
pressure (SRP), and continuous or impulsive thrust.

Provides selectable numerical integration backends:
- `'rk4'`: Fixed-step 4th-order Runge-Kutta.
- `'rk45'`: Adaptive step-size Dormand-Prince 5(4) with embedded LTE control.
"""

from typing import Callable, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt

from core.constants import G0, G_EARTH, R_EARTH
from core.forces import (
    accel_atmospheric_drag,
    accel_earth_gravity,
    accel_electric_prograde,
    accel_fixed_thrust,
    accel_j2_perturbation,
    accel_j3_perturbation,
    accel_j4_perturbation,
    accel_lunar_gravity,
    accel_solar_radiation_pressure,
)
from core.integrators import rk4, rk4_step, rk45_adaptive


class SpacecraftPropagator:
    r"""Unified orbital propagation engine supporting 6-DOF and 7-DOF dynamics."""

    def __init__(
        self,
        mass: float = 500.0,
        drag_area: float = 2.0,
        cd: float = 2.2,
        srp_area: float = 4.0,
        cr: float = 1.2,
        isp: float = 1800.0,
        use_j2: bool = True,
        use_j3: bool = False,
        use_j4: bool = False,
        use_lunar: bool = False,
        use_drag: bool = False,
        use_srp: bool = False,
        use_thrust: bool = False,
        mu: float = G_EARTH,
        r_body: float = R_EARTH,
        area_drag: Optional[float] = None,
        area_srp: Optional[float] = None,
        thrust_mag: float = 0.0,
        thrust_steering_law: Optional[Callable[[float, np.ndarray, np.ndarray, float], np.ndarray]] = None,
        **kwargs,
    ):
        self.mass = float(mass)
        self.cd = float(cd)
        self.cr = float(cr)
        self.isp = float(isp)
        self.mu = float(mu)
        self.r_body = float(r_body)

        # Handle alias compatibility between old and new names
        self.drag_area = float(area_drag if area_drag is not None else drag_area)
        self.area_drag = self.drag_area
        self.srp_area = float(area_srp if area_srp is not None else srp_area)
        self.area_srp = self.srp_area

        # Perturbation flags
        self.use_j2 = use_j2
        self.use_j3 = use_j3
        self.use_j4 = use_j4
        self.use_lunar = use_lunar
        self.use_drag = use_drag
        self.use_srp = use_srp
        self.use_thrust = use_thrust

        # Thrust modes & steering
        self.thrust_mode = "none"
        self.thrust_params = {}
        self.thrust_mag = float(thrust_mag)
        self.thrust_steering_law = thrust_steering_law

    def configure_fixed_burn(
        self,
        start_t: float,
        duration: float,
        thrust_vec: np.ndarray,
        isp: Optional[float] = None,
    ) -> "SpacecraftPropagator":
        r"""Configure an inertial directional burn over $[t_{\text{start}}, t_{\text{start}} + \Delta t]$."""
        thrust_arr = np.asarray(thrust_vec, dtype=np.float64)
        self.use_thrust = True
        self.thrust_mode = "fixed"
        self.thrust_params = {
            "start_t": float(start_t),
            "duration": float(duration),
            "vec": thrust_arr,
        }
        self.thrust_mag = float(np.linalg.norm(thrust_arr))
        if isp is not None:
            self.isp = float(isp)

        t_end = float(start_t + duration)

        def _fixed_burn_steering(t: float, r: np.ndarray, v: np.ndarray, mass: float) -> np.ndarray:
            if start_t <= t <= t_end and mass > 20.0:
                return thrust_arr / mass
            return np.zeros(3, dtype=np.float64)

        self.thrust_steering_law = _fixed_burn_steering
        return self

    def configure_electric_burn(
        self,
        thrust_magnitude: Optional[float] = None,
        thrust_mag: Optional[float] = None,
        isp: Optional[float] = None,
        steering_law: Optional[Callable[[float, np.ndarray, np.ndarray, float], np.ndarray]] = None,
    ) -> "SpacecraftPropagator":
        r"""Configure continuous prograde low thrust ($T$ in Newtons)."""
        mag = thrust_magnitude if thrust_magnitude is not None else (thrust_mag if thrust_mag is not None else 0.0)
        self.use_thrust = True
        self.thrust_mode = "electric_prograde"
        self.thrust_mag = float(mag)
        self.thrust_params = {"thrust": float(mag)}
        if isp is not None:
            self.isp = float(isp)
        if steering_law is not None:
            self.thrust_steering_law = steering_law
        return self

    def configure_thrust(
        self,
        thrust_mag: Optional[float] = None,
        thrust_magnitude: Optional[float] = None,
        isp: Optional[float] = None,
        steering_law: Optional[Callable[[float, np.ndarray, np.ndarray, float], np.ndarray]] = None,
    ) -> "SpacecraftPropagator":
        r"""Unified wrapper for low-thrust configuration."""
        return self.configure_electric_burn(
            thrust_magnitude=thrust_magnitude,
            thrust_mag=thrust_mag,
            isp=isp,
            steering_law=steering_law,
        )

    def derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        r"""Evaluate total state derivative vector $[d\mathbf{r}/dt, d\mathbf{v}/dt (, dm/dt)]^T$."""
        r = state[0:3]
        v = state[3:6]
        current_mass = float(state[6]) if len(state) >= 7 else self.mass

        # Central Newtonian gravity
        a_total = accel_earth_gravity(r)

        # Superposition of Perturbations
        if self.use_j2:
            a_total = a_total + accel_j2_perturbation(r)
        if self.use_j3:
            a_total = a_total + accel_j3_perturbation(r)
        if self.use_j4:
            a_total = a_total + accel_j4_perturbation(r)
        if self.use_lunar:
            a_total = a_total + accel_lunar_gravity(r, t)
        if self.use_drag:
            a_total = a_total + accel_atmospheric_drag(r, v, self.cd, self.drag_area, current_mass)
        if self.use_srp:
            a_total = a_total + accel_solar_radiation_pressure(r, self.cr, self.srp_area, current_mass)

        # Thrust & Mass Depletion
        m_dot = 0.0
        if self.use_thrust or self.thrust_mode != "none":
            if self.thrust_steering_law is not None:
                a_total = a_total + self.thrust_steering_law(t, r, v, current_mass)
                if self.thrust_mag > 0.0 and current_mass > 20.0:
                    m_dot = -self.thrust_mag / (G0 * self.isp)
            elif self.thrust_mode == "fixed":
                a_total = a_total + accel_fixed_thrust(
                    t,
                    self.thrust_params["start_t"],
                    self.thrust_params["duration"],
                    self.thrust_params["vec"],
                    current_mass,
                )
                start_t = self.thrust_params["start_t"]
                t_end = start_t + self.thrust_params["duration"]
                if start_t <= t <= t_end and current_mass > 20.0:
                    m_dot = -self.thrust_mag / (G0 * self.isp)
            elif self.thrust_mode == "electric_prograde":
                t_mag = self.thrust_params.get("thrust", self.thrust_mag)
                if current_mass > 20.0 and t_mag > 0.0:
                    a_total = a_total + accel_electric_prograde(v, t_mag, current_mass)
                    m_dot = -t_mag / (G0 * self.isp)

        if len(state) >= 7:
            return np.concatenate((v, a_total, [m_dot]))
        return np.concatenate((v, a_total))

    def propagate(
        self,
        r0: Optional[np.ndarray] = None,
        v0: Optional[np.ndarray] = None,
        t_span: float = 86400.0,
        dt: float = 10.0,
        track_mass: bool = False,
        mass0: Optional[float] = None,
        method: str = "rk4",
        rtol: float = 1e-8,
        atol: float = 1e-10,
        h_min: float = 1e-4,
        h_max: float = 86400.0,
        state0: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Propagate state vectors across the designated duration."""
        if state0 is not None:
            state0_arr = np.asarray(state0, dtype=np.float64)
            r0_arr = state0_arr[0:3]
            v0_arr = state0_arr[3:6]
            if len(state0_arr) >= 7:
                track_mass = True
                mass0 = float(state0_arr[6])
        else:
            if r0 is None or v0 is None:
                raise ValueError("Must provide either (r0, v0) or combined state0.")
            r0_arr = np.asarray(r0, dtype=np.float64)
            v0_arr = np.asarray(v0, dtype=np.float64)

        is_7dof = track_mass or (mass0 is not None)
        initial_mass = float(mass0) if mass0 is not None else self.mass

        if is_7dof:
            initial_state = np.concatenate((r0_arr, v0_arr, [initial_mass]))
        else:
            initial_state = np.concatenate((r0_arr, v0_arr))

        chosen_method = (method or "rk4").strip().lower()

        if chosen_method == "rk4":
            num_steps = int(round(float(t_span) / float(dt))) + 1
            times = np.linspace(0.0, float(t_span), num_steps)
            states = np.zeros((num_steps, len(initial_state)), dtype=np.float64)
            states[0] = initial_state

            actual_dt = times[1] - times[0] if num_steps > 1 else float(dt)
            for i in range(1, num_steps):
                states[i] = rk4_step(
                    self.derivatives, times[i - 1], states[i - 1], actual_dt
                )

            return times, states

        elif chosen_method in ("rk45", "dopri5"):
            return rk45_adaptive(
                derivs_func=self.derivatives,
                t_span=(0.0, float(t_span)),
                y0=initial_state,
                rtol=float(rtol),
                atol=float(atol),
                h_init=float(dt),
                h_min=float(h_min),
                h_max=float(h_max),
            )
        else:
            raise ValueError(f"Unsupported numerical integration method '{method}'. Choose 'rk4' or 'rk45'.")

    @staticmethod
    def plot_3d(states: np.ndarray, title: str = "Trajectory") -> None:
        r"""Plot the computed 3D orbit trajectory around a scaled Earth sphere."""
        states_arr = np.asarray(states)
        x_km = states_arr[:, 0] / 1000.0
        y_km = states_arr[:, 1] / 1000.0
        z_km = states_arr[:, 2] / 1000.0

        fig = plt.figure(figsize=(10.0, 8.0))
        ax = fig.add_subplot(111, projection="3d")

        # Earth wireframe sphere
        r_earth_km = R_EARTH / 1000.0
        u, v = np.mgrid[0 : 2 * np.pi : 30j, 0 : np.pi : 15j]
        ax.plot_wireframe(
            r_earth_km * np.cos(u) * np.sin(v),
            r_earth_km * np.sin(u) * np.sin(v),
            r_earth_km * np.cos(v),
            color="dodgerblue",
            alpha=0.25,
            label="Earth",
        )

        # Orbit trajectory
        ax.plot(x_km, y_km, z_km, color="crimson", linewidth=1.5, label="Trajectory")
        ax.scatter(x_km[0], y_km[0], z_km[0], color="forestgreen", s=50, label="Start")
        ax.scatter(x_km[-1], y_km[-1], z_km[-1], color="black", s=50, label="End")

        # Equal aspect ratio scaling
        max_r = np.array([x_km.max() - x_km.min(), y_km.max() - y_km.min(), z_km.max() - z_km.min()]).max() / 2.0
        mid_x = (x_km.max() + x_km.min()) * 0.5
        mid_y = (y_km.max() + y_km.min()) * 0.5
        mid_z = (z_km.max() + z_km.min()) * 0.5
        ax.set_xlim(mid_x - max_r, mid_x + max_r)
        ax.set_ylim(mid_y - max_r, mid_y + max_r)
        ax.set_zlim(mid_z - max_r, mid_z + max_r)

        ax.set_xlabel("ECI X (km)")
        ax.set_ylabel("ECI Y (km)")
        ax.set_zlabel("ECI Z (km)")
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        plt.show()
