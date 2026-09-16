r"""Unified Spacecraft Numerical Propagation Engine.

Supports 6-DOF $(\mathbf{r}, \mathbf{v})$ and 7-DOF $(\mathbf{r}, \mathbf{v}, m)$
orbital trajectory simulation under primary Newtonian central-body gravitation,
geopotential zonal harmonics ($J_2$-$J_4$), third-body lunar perturbations,
exponential atmospheric drag, cannonball Solar Radiation Pressure (SRP), and
continuous or impulsive propulsive maneuvers.

Provides selectable numerical integration backends:
- `'rk4'`: Fixed-step classical 4th-order Runge-Kutta.
- `'rk45'`: Adaptive step-size Dormand-Prince 5(4) with embedded LTE control.
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np

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
from core.integrators import rk4_step, rk45_adaptive

# ============================================================================
# PROPAGATION ENGINE CONSTANTS & THRESHOLDS
# ============================================================================
REENTRY_ALTITUDE_THRESHOLD: float = 80_000.0  # Atmospheric re-entry floor [m]
MINIMUM_PROPULSION_MASS: float = 20.0        # Dry structural cutoff mass [kg]


class SpacecraftPropagator:
    r"""Unified orbital propagation engine supporting 6-DOF and 7-DOF dynamics.

    Propagates Cartesian state vectors in the Earth-Centered Inertial (ECI)
    frame by assembling the total acceleration vector:

    $$\ddot{\mathbf{r}} = \mathbf{a}_{\text{grav}} + \mathbf{a}_{\text{pert}} + \mathbf{a}_{\text{thrust}}$$

    and mass depletion dynamics:

    $$\dot{m} = -\frac{\|\mathbf{F}_{\text{thrust}}\|}{I_{\text{sp}} g_0}$$
    """

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
        **kwargs: Any,
    ) -> None:
        r"""Initialize spacecraft properties, perturbation toggles, and steering laws.

        Args:
            mass: Dry or wet initial spacecraft mass in kilograms. Defaults to 500.0.
            drag_area: Frontal aerodynamic drag reference area in square meters. Defaults to 2.0.
            cd: Dimensionless aerodynamic drag coefficient. Defaults to 2.2.
            srp_area: Solar radiation pressure illuminated area in square meters. Defaults to 4.0.
            cr: Dimensionless radiation pressure reflectivity coefficient. Defaults to 1.2.
            isp: Specific impulse in seconds. Defaults to 1800.0.
            use_j2: Enable J2 oblateness harmonic perturbation. Defaults to True.
            use_j3: Enable J3 pear-shaped harmonic perturbation. Defaults to False.
            use_j4: Enable J4 second-order oblateness harmonic perturbation. Defaults to False.
            use_lunar: Enable third-body lunar gravity perturbation. Defaults to False.
            use_drag: Enable atmospheric drag acceleration. Defaults to False.
            use_srp: Enable solar radiation pressure with cylindrical shadowing. Defaults to False.
            use_thrust: Enable active thruster acceleration. Defaults to False.
            mu: Central body gravitational parameter in m^3/s^2. Defaults to G_EARTH.
            r_body: Central body mean radius in meters. Defaults to R_EARTH.
            area_drag: Parameter alias for drag_area. Defaults to None.
            area_srp: Parameter alias for srp_area. Defaults to None.
            thrust_mag: Continuous thrust magnitude in Newtons. Defaults to 0.0.
            thrust_steering_law: Optional callable returning thrust acceleration vector. Defaults to None.
            **kwargs: Backward-compatibility keyword arguments.

        Raises:
            ValueError: If mass, isp, mu, or r_body are non-positive.
        """
        if float(mass) <= 0.0:
            raise ValueError(f"Spacecraft initial mass must be positive, got {mass} kg.")
        if float(isp) <= 0.0:
            raise ValueError(f"Specific impulse must be positive, got {isp} s.")
        if float(mu) <= 0.0:
            raise ValueError(f"Gravitational parameter mu must be positive, got {mu}.")
        if float(r_body) <= 0.0:
            raise ValueError(f"Central body radius must be positive, got {r_body} m.")

        self.mass: float = float(mass)
        self.cd: float = float(cd)
        self.cr: float = float(cr)
        self.isp: float = float(isp)
        self.mu: float = float(mu)
        self.r_body: float = float(r_body)

        # Handle alias compatibility between area parameter names
        self.drag_area: float = float(area_drag if area_drag is not None else drag_area)
        self.area_drag: float = self.drag_area
        self.srp_area: float = float(area_srp if area_srp is not None else srp_area)
        self.area_srp: float = self.srp_area

        # Force perturbation flags
        self.use_j2: bool = bool(use_j2)
        self.use_j3: bool = bool(use_j3)
        self.use_j4: bool = bool(use_j4)
        self.use_lunar: bool = bool(use_lunar)
        self.use_drag: bool = bool(use_drag)
        self.use_srp: bool = bool(use_srp)
        self.use_thrust: bool = bool(use_thrust)

        # Propulsion state, guidance configuration, and status monitors
        self.thrust_mode: str = "none"
        self.thrust_params: Dict[str, Union[float, np.ndarray]] = {}
        self.thrust_mag: float = float(thrust_mag)
        self.thrust_steering_law: Optional[Callable[[float, np.ndarray, np.ndarray, float], np.ndarray]] = thrust_steering_law
        self.reentry_detected: bool = False

    # ========================================================================
    # PROPULSION CONFIGURATION
    # ========================================================================

    def configure_fixed_burn(
        self,
        start_t: float,
        duration: float,
        thrust_vec: np.ndarray,
        isp: Optional[float] = None,
    ) -> "SpacecraftPropagator":
        r"""Configure a fixed inertial directional burn over $[t_{\text{start}}, t_{\text{start}} + \Delta t]$.

        Args:
            start_t: Burn ignition epoch in seconds.
            duration: Total continuous burn duration in seconds.
            thrust_vec: Applied inertial thrust vector of shape `(3,)` in Newtons.
            isp: Optional override for specific impulse in seconds. Defaults to None.

        Returns:
            SpacecraftPropagator: Self reference for method chaining.

        Raises:
            ValueError: If duration is non-positive or thrust_vec does not have 3 elements.
        """
        thrust_arr = np.asarray(thrust_vec, dtype=np.float64)
        if thrust_arr.shape != (3,):
            raise ValueError(f"thrust_vec must be a 3D vector, got shape {thrust_arr.shape}.")
        if float(duration) <= 0.0:
            raise ValueError(f"Burn duration must be positive, got {duration} s.")

        self.use_thrust = True
        self.thrust_mode = "fixed"
        self.thrust_params = {
            "start_t": float(start_t),
            "duration": float(duration),
            "vec": thrust_arr,
        }
        self.thrust_mag = float(np.linalg.norm(thrust_arr))
        if isp is not None:
            if float(isp) <= 0.0:
                raise ValueError(f"Override specific impulse must be positive, got {isp} s.")
            self.isp = float(isp)

        t_end = float(start_t + duration)

        def _fixed_burn_steering(t: float, r: np.ndarray, v: np.ndarray, mass: float) -> np.ndarray:
            if start_t <= t <= t_end and mass > MINIMUM_PROPULSION_MASS:
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
        r"""Configure continuous prograde low thrust aligned with instantaneous velocity.

        $$\mathbf{a}_{\text{thrust}} = \frac{T}{m} \hat{\mathbf{v}}$$

        Args:
            thrust_magnitude: Continuous thrust force magnitude in Newtons. Defaults to None.
            thrust_mag: Parameter alias for thrust_magnitude. Defaults to None.
            isp: Optional override for specific impulse in seconds. Defaults to None.
            steering_law: Optional directional override steering function callback. Defaults to None.

        Returns:
            SpacecraftPropagator: Self reference for method chaining.

        Raises:
            ValueError: If thrust magnitude is negative or override isp is non-positive.
        """
        mag = thrust_magnitude if thrust_magnitude is not None else (thrust_mag if thrust_mag is not None else 0.0)
        if float(mag) < 0.0:
            raise ValueError(f"Thrust magnitude must be non-negative, got {mag} N.")
        if isp is not None and float(isp) <= 0.0:
            raise ValueError(f"Override specific impulse must be positive, got {isp} s.")

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
        r"""Unified wrapper for continuous low-thrust propulsion configuration.

        Args:
            thrust_mag: Parameter alias for continuous thrust magnitude in Newtons. Defaults to None.
            thrust_magnitude: Explicit continuous thrust magnitude in Newtons. Defaults to None.
            isp: Optional override for specific impulse in seconds. Defaults to None.
            steering_law: Custom steering law callback. Defaults to None.

        Returns:
            SpacecraftPropagator: Self reference for method chaining.

        Raises:
            ValueError: If thrust magnitude is negative or override isp is non-positive.
        """
        return self.configure_electric_burn(
            thrust_magnitude=thrust_magnitude,
            thrust_mag=thrust_mag,
            isp=isp,
            steering_law=steering_law,
        )

    # ========================================================================
    # STATE VECTOR DERIVATIVE JUNCTION
    # ========================================================================

    def derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        r"""Evaluate total state derivative vector $[d\mathbf{r}/dt, d\mathbf{v}/dt (, dm/dt)]^T$.

        Args:
            t: Current epoch time in seconds.
            state: Integrated state vector array of shape `(6,)` or `(7,)` in SI units.

        Returns:
            np.ndarray: First-order derivatives of shape `(6,)` or `(7,)` matching input state dimension.

        Raises:
            ValueError: If state array has fewer than 6 elements.
        """
        if len(state) < 6:
            raise ValueError(f"State vector must contain at least 6 elements, got length {len(state)}.")

        r = state[0:3]
        v = state[3:6]
        current_mass = float(state[6]) if len(state) >= 7 else self.mass

        # Atmospheric Re-entry / Ground Impact Floor (80 km boundary interface)
        r_mag = np.linalg.norm(r)
        if r_mag <= (self.r_body + REENTRY_ALTITUDE_THRESHOLD):
            self.reentry_detected = True
            return np.zeros_like(state)

        # Primary Newtonian two-body gravitation
        a_total = accel_earth_gravity(r)

        # Superposition of Geopotential & Environmental Perturbations
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

        # Propulsive Thrust Acceleration & Mass Depletion
        m_dot = 0.0
        if self.use_thrust or self.thrust_mode != "none":
            if self.thrust_steering_law is not None:
                a_total = a_total + self.thrust_steering_law(t, r, v, current_mass)
                if self.thrust_mag > 0.0 and current_mass > MINIMUM_PROPULSION_MASS:
                    m_dot = -self.thrust_mag / (G0 * self.isp)
            elif self.thrust_mode == "fixed":
                a_total = a_total + accel_fixed_thrust(
                    t,
                    self.thrust_params["start_t"],
                    self.thrust_params["duration"],
                    self.thrust_params["vec"],
                    current_mass,
                )
                start_t = float(self.thrust_params["start_t"])
                t_end = start_t + float(self.thrust_params["duration"])
                if start_t <= t <= t_end and current_mass > MINIMUM_PROPULSION_MASS:
                    m_dot = -self.thrust_mag / (G0 * self.isp)
            elif self.thrust_mode == "electric_prograde":
                t_mag = float(self.thrust_params.get("thrust", self.thrust_mag))
                if current_mass > MINIMUM_PROPULSION_MASS and t_mag > 0.0:
                    a_total = a_total + accel_electric_prograde(v, t_mag, current_mass)
                    m_dot = -t_mag / (G0 * self.isp)

        if len(state) >= 7:
            return np.concatenate((v, a_total, [m_dot]))
        return np.concatenate((v, a_total))

    # ========================================================================
    # NUMERICAL INTEGRATION
    # ========================================================================

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
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Propagate Cartesian state vectors forward across the designated time duration.

        Args:
            r0: Initial position vector in the ECI frame of shape `(3,)` in meters. Defaults to None.
            v0: Initial velocity vector in the ECI frame of shape `(3,)` in m/s. Defaults to None.
            t_span: Total propagation duration in seconds. Defaults to 86400.0.
            dt: Fixed integration step size or initial candidate step size in seconds. Defaults to 10.0.
            track_mass: If True, tracks fuel consumption via 7-DOF dynamic equations. Defaults to False.
            mass0: Initial vehicle mass override in kilograms. Defaults to None.
            method: Numerical integrator backend selection ('rk4' or 'rk45'). Defaults to 'rk4'.
            rtol: Relative error tolerance for adaptive Dormand-Prince ('rk45'). Defaults to 1e-8.
            atol: Absolute error tolerance for adaptive Dormand-Prince ('rk45'). Defaults to 1e-10.
            h_min: Minimum allowable step size for adaptive integration in seconds. Defaults to 1e-4.
            h_max: Maximum allowable step size for adaptive integration in seconds. Defaults to 86400.0.
            state0: Pre-assembled initial state vector array `(r0, v0 [, m0])`. Defaults to None.
            **kwargs: Additional backward-compatibility keyword arguments.

        Returns:
            Tuple containing:
                - times (np.ndarray): Discrete solution epochs of shape `(N,)` in seconds.
                - states (np.ndarray): Propagated trajectory states of shape `(N, 6)` or `(N, 7)`.

        Raises:
            ValueError: If neither state0 nor (r0, v0) is provided, if t_span or dt is non-positive,
                or if an unsupported integration method is requested.
        """
        if float(t_span) <= 0.0:
            raise ValueError(f"Propagation duration t_span must be positive, got {t_span} s.")
        if float(dt) <= 0.0:
            raise ValueError(f"Integration time step dt must be positive, got {dt} s.")

        if state0 is not None:
            state0_arr = np.asarray(state0, dtype=np.float64)
            if len(state0_arr) < 6:
                raise ValueError(f"state0 must contain at least 6 elements, got shape {state0_arr.shape}.")
            r0_arr = state0_arr[0:3]
            v0_arr = state0_arr[3:6]
            if len(state0_arr) >= 7:
                track_mass = True
                mass0 = float(state0_arr[6])
        else:
            if r0 is None or v0 is None:
                raise ValueError("Must provide either (r0, v0) state vectors or combined state0.")
            r0_arr = np.asarray(r0, dtype=np.float64)
            v0_arr = np.asarray(v0, dtype=np.float64)
            if r0_arr.shape != (3,) or v0_arr.shape != (3,):
                raise ValueError(f"r0 and v0 must each have shape (3,), got r0={r0_arr.shape} and v0={v0_arr.shape}.")

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
            self.reentry_detected = False

            actual_dt = times[1] - times[0] if num_steps > 1 else float(dt)
            for i in range(1, num_steps):
                states[i] = rk4_step(
                    self.derivatives, times[i - 1], states[i - 1], actual_dt
                )
                r_step_mag = np.linalg.norm(states[i, 0:3])
                if r_step_mag <= (self.r_body + REENTRY_ALTITUDE_THRESHOLD):
                    self.reentry_detected = True
                    altitude_km = (r_step_mag - self.r_body) / 1000.0
                    print(
                        f"\n[WARNING] Simulation terminated early at t = {times[i]:.1f} s ({times[i] / 86400.0:.2f} days): "
                        f"spacecraft experienced atmospheric re-entry (altitude = {altitude_km:.2f} km <= {REENTRY_ALTITUDE_THRESHOLD / 1000.0:.1f} km)."
                    )
                    return times[: i + 1], states[: i + 1]

            return times, states

        elif chosen_method in ("rk45", "dopri5"):
            self.reentry_detected = False
            t_hist, y_hist = rk45_adaptive(
                derivs_func=self.derivatives,
                t_span=(0.0, float(t_span)),
                y0=initial_state,
                rtol=float(rtol),
                atol=float(atol),
                h_init=float(dt),
                h_min=float(h_min),
                h_max=float(h_max),
            )
            r_hist_mag = np.linalg.norm(y_hist[:, 0:3], axis=1)
            below_floor = np.where(r_hist_mag <= (self.r_body + REENTRY_ALTITUDE_THRESHOLD))[0]
            if len(below_floor) > 0:
                self.reentry_detected = True
                first_impact = below_floor[0]
                t_hist = t_hist[: first_impact + 1]
                y_hist = y_hist[: first_impact + 1]
                altitude_km = (r_hist_mag[first_impact] - self.r_body) / 1000.0
                print(
                    f"\n[WARNING] Simulation terminated early at t = {t_hist[-1]:.1f} s ({t_hist[-1] / 86400.0:.2f} days): "
                    f"spacecraft experienced atmospheric re-entry (altitude = {altitude_km:.2f} km <= {REENTRY_ALTITUDE_THRESHOLD / 1000.0:.1f} km)."
                )
            return t_hist, y_hist
        else:
            raise ValueError(f"Unsupported numerical integration method '{method}'. Choose 'rk4' or 'rk45'.")

    # ========================================================================
    # TRAJECTORY VISUALIZATION
    # ========================================================================

    @staticmethod
    def plot_3d(states: np.ndarray, title: str = "Trajectory") -> None:
        r"""Plot the computed 3D Cartesian orbit trajectory around a scaled Earth wireframe sphere.

        Args:
            states: Position and velocity state vector history array of shape `(N, 6)` or `(N, 7)`.
            title: Matplotlib figure title string. Defaults to 'Trajectory'.

        Raises:
            ValueError: If states array does not have 2 dimensions or fewer than 3 columns.
        """
        states_arr = np.asarray(states, dtype=np.float64)
        if states_arr.ndim != 2 or states_arr.shape[1] < 3:
            raise ValueError(f"states must have shape (N, M) with M >= 3, got shape {states_arr.shape}.")

        x_km = states_arr[:, 0] / 1000.0
        y_km = states_arr[:, 1] / 1000.0
        z_km = states_arr[:, 2] / 1000.0

        fig = plt.figure(figsize=(10.0, 8.0))
        ax = fig.add_subplot(111, projection="3d")

        # Earth wireframe sphere reference
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

        # Orbit trajectory curve
        ax.plot(x_km, y_km, z_km, color="crimson", linewidth=1.5, label="Trajectory")
        ax.scatter(x_km[0], y_km[0], z_km[0], color="forestgreen", s=50, label="Start")
        ax.scatter(x_km[-1], y_km[-1], z_km[-1], color="black", s=50, label="End")

        # Equal aspect ratio scaling across all 3 Cartesian axes
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