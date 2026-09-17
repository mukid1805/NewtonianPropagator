r"""Unified Spacecraft Numerical Propagation Engine.

Supports 6-DOF $(\mathbf{r}, \mathbf{v})$, 7-DOF $(\mathbf{r}, \mathbf{v}, m)$,
and 42-DOF $(\mathbf{r}, \mathbf{v}, \Phi)$ trajectory simulation under primary
Newtonian gravitation, zonal harmonics ($J_2$–$J_4$), third-body lunar
perturbations, exponential atmospheric drag, cannonball Solar Radiation
Pressure (SRP), and propulsive manoeuvres.

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
    gravity_gradient_tensor,
)
from core.integrators import rk4_step, rk45_adaptive

# ============================================================================
# PROPAGATION ENGINE CONSTANTS & THRESHOLDS
# ============================================================================
REENTRY_ALTITUDE_THRESHOLD: float = 80_000.0  # Atmospheric re-entry boundary interface [m]
MINIMUM_PROPULSION_MASS: float = 20.0        # Structural cutoff limit [kg]


class SpacecraftPropagator:
    r"""Unified orbital propagation engine supporting 6-DOF, 7-DOF, and 42-DOF dynamics.

    Propagates Cartesian state vectors in the Earth-Centred Inertial (ECI)
    frame by assembling the acceleration vector:

    $$\ddot{\mathbf{r}} = \mathbf{a}_{\text{grav}} + \mathbf{a}_{\text{pert}} + \mathbf{a}_{\text{thrust}}$$

    coupled variational state equations:

    $$\dot{\Phi}(t, t_0) = \mathbf{A}(t) \Phi(t, t_0)$$

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
        r"""Initialise spacecraft parameters, perturbation toggles, and steering laws.

        Args:
            mass: Dry or wet initial spacecraft mass in kilograms. Defaults to 500.0.
            drag_area: Frontal aerodynamic drag reference area in square metres. Defaults to 2.0.
            cd: Dimensionless aerodynamic drag coefficient. Defaults to 2.2.
            srp_area: Solar radiation pressure illuminated area in square metres. Defaults to 4.0.
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
            r_body: Central body mean radius in metres. Defaults to R_EARTH.
            area_drag: Parameter alias for drag_area. Defaults to None.
            area_srp: Parameter alias for srp_area. Defaults to None.
            thrust_mag: Continuous thrust magnitude in Newtons. Defaults to 0.0.
            thrust_steering_law: Optional callable returning thrust acceleration vector. Defaults to None.
            kwargs: Supplementary backward-compatibility keyword arguments.

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

        # Propulsion state, guidance configuration, and monitors
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
    # STATE VECTOR DERIVATIVE JUNCTIONS
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

        # Atmospheric Re-entry / Ground Impact Boundary (80 km altitude floor)
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

    def _stm_derivatives(self, t: float, state_aug: np.ndarray) -> np.ndarray:
        r"""Evaluate coupled 42-state dynamics: $[d\mathbf{r}/dt, d\mathbf{v}/dt, \text{vec}(d\Phi/dt)]^T$."""
        x = state_aug[0:6]
        phi = state_aug[6:42].reshape((6, 6))

        r = x[0:3]
        if np.linalg.norm(r) <= (self.r_body + REENTRY_ALTITUDE_THRESHOLD):
            self.reentry_detected = True
            return np.zeros_like(state_aug)

        # Baseline kinematics and physical dynamics
        dx = self.derivatives(t, x)[0:6]

        # Evaluate system plant Jacobian A(t)
        g_mat = gravity_gradient_tensor(r, mu=self.mu, use_j2=self.use_j2, r_body=self.r_body)
        a_mat = np.zeros((6, 6), dtype=np.float64)
        a_mat[0:3, 3:6] = np.eye(3, dtype=np.float64)
        a_mat[3:6, 0:3] = g_mat

        # Variational equation dPhi/dt = A(t) @ Phi(t)
        dphi = a_mat @ phi
        return np.concatenate((dx, dphi.ravel()))

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
            r0: Initial position vector in the ECI frame of shape `(3,)` in metres. Defaults to None.
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
            kwargs: Supplementary backward-compatibility keyword arguments.

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
                states[i] = rk4_step(self.derivatives, times[i - 1], states[i - 1], actual_dt)
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

    def propagate_with_stm(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        t_span: float = 86400.0,
        dt: float = 10.0,
        method: str = "rk4",
        rtol: float = 1e-8,
        atol: float = 1e-10,
        h_min: float = 1e-4,
        h_max: float = 86400.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Propagate orbital states alongside the 6x6 State Transition Matrix $\Phi(t, t_0)$.

        Args:
            r0: Initial position vector in the ECI frame of shape `(3,)` in metres.
            v0: Initial velocity vector in the ECI frame of shape `(3,)` in m/s.
            t_span: Total integration duration in seconds. Defaults to 86400.0.
            dt: Fixed integration step size or initial candidate step size in seconds. Defaults to 10.0.
            method: Numerical integrator backend selection ('rk4' or 'rk45'). Defaults to 'rk4'.
            rtol: Relative error tolerance for adaptive Dormand-Prince ('rk45'). Defaults to 1e-8.
            atol: Absolute error tolerance for adaptive Dormand-Prince ('rk45'). Defaults to 1e-10.
            h_min: Minimum allowable step size for adaptive integration in seconds. Defaults to 1e-4.
            h_max: Maximum allowable step size for adaptive integration in seconds. Defaults to 86400.0.

        Returns:
            Tuple containing:
                - times (np.ndarray): Discrete solution epochs of shape `(N,)` in seconds.
                - states (np.ndarray): Orbit states of shape `(N, 6)` in metres and m/s.
                - stms (np.ndarray): State Transition Matrices $\Phi(t_k, t_0)$ of shape `(N, 6, 6)`.

        Raises:
            ValueError: If position or velocity arrays do not have shape `(3,)`, or if t_span is non-positive.
        """
        r0_arr = np.asarray(r0, dtype=np.float64)
        v0_arr = np.asarray(v0, dtype=np.float64)
        if r0_arr.shape != (3,) or v0_arr.shape != (3,):
            raise ValueError(f"r0 and v0 must each have shape (3,), got r0={r0_arr.shape} and v0={v0_arr.shape}.")
        if float(t_span) <= 0.0:
            raise ValueError(f"t_span must be positive, got {t_span} s.")

        x0 = np.concatenate((r0_arr, v0_arr))
        phi0 = np.eye(6, dtype=np.float64).ravel()
        aug0 = np.concatenate((x0, phi0))

        chosen_method = (method or "rk4").strip().lower()

        if chosen_method == "rk4":
            num_steps = int(round(float(t_span) / float(dt))) + 1
            times = np.linspace(0.0, float(t_span), num_steps)
            aug_states = np.zeros((num_steps, 42), dtype=np.float64)
            aug_states[0] = aug0
            self.reentry_detected = False

            actual_dt = times[1] - times[0] if num_steps > 1 else float(dt)
            for i in range(1, num_steps):
                aug_states[i] = rk4_step(self._stm_derivatives, times[i - 1], aug_states[i - 1], actual_dt)
                r_step_mag = np.linalg.norm(aug_states[i, 0:3])
                if r_step_mag <= (self.r_body + REENTRY_ALTITUDE_THRESHOLD):
                    self.reentry_detected = True
                    return times[: i + 1], aug_states[: i + 1, 0:6], aug_states[: i + 1, 6:42].reshape((-1, 6, 6))

            return times, aug_states[:, 0:6], aug_states[:, 6:42].reshape((-1, 6, 6))

        elif chosen_method in ("rk45", "dopri5"):
            self.reentry_detected = False
            t_hist, y_hist = rk45_adaptive(
                derivs_func=self._stm_derivatives,
                t_span=(0.0, float(t_span)),
                y0=aug0,
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

            return t_hist, y_hist[:, 0:6], y_hist[:, 6:42].reshape((-1, 6, 6))
        else:
            raise ValueError(f"Unsupported numerical integration method '{method}'. Choose 'rk4' or 'rk45'.")

    def propagate_covariance(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        cov0: np.ndarray,
        t_span: float = 86400.0,
        dt: float = 10.0,
        method: str = "rk4",
        rtol: float = 1e-8,
        atol: float = 1e-10,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Propagate a 6x6 state covariance matrix forward under linear STM mapping.

        $$\mathbf{P}(t_k) = \Phi(t_k, t_0) \mathbf{P}_0 \Phi(t_k, t_0)^T$$

        Args:
            r0: Initial position vector in the ECI frame of shape `(3,)` in metres.
            v0: Initial velocity vector in the ECI frame of shape `(3,)` in m/s.
            cov0: Initial symmetric 6x6 state covariance matrix $\mathbf{P}_0$ in m^2 and (m/s)^2.
            t_span: Total propagation duration in seconds. Defaults to 86400.0.
            dt: Time step size in seconds. Defaults to 10.0.
            method: Integrator backend ('rk4' or 'rk45'). Defaults to 'rk4'.
            rtol: Relative tolerance for Dormand-Prince integrator. Defaults to 1e-8.
            atol: Absolute tolerance for Dormand-Prince integrator. Defaults to 1e-10.

        Returns:
            Tuple containing:
                - times (np.ndarray): Solution epochs of shape `(N,)` in seconds.
                - states (np.ndarray): Orbit state trajectory of shape `(N, 6)` in metres and m/s.
                - covariances (np.ndarray): Evolved covariance matrices of shape `(N, 6, 6)`.

        Raises:
            ValueError: If initial covariance matrix does not have shape `(6, 6)`.
        """
        cov0_arr = np.asarray(cov0, dtype=np.float64)
        if cov0_arr.shape != (6, 6):
            raise ValueError(f"Initial covariance matrix must have shape (6, 6), got {cov0_arr.shape}.")

        times, states, stms = self.propagate_with_stm(
            r0=r0,
            v0=v0,
            t_span=t_span,
            dt=dt,
            method=method,
            rtol=rtol,
            atol=atol,
        )

        n_points = len(times)
        cov_history = np.zeros((n_points, 6, 6), dtype=np.float64)

        for i in range(n_points):
            phi = stms[i]
            cov_history[i] = phi @ cov0_arr @ phi.T

        return times, states, cov_history

    # ========================================================================
    # TRAJECTORY VISUALISATION
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
