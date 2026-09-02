"""
Unified Spacecraft Numerical Propagation Engine.

Supports 6-DOF (r, v) and 7-DOF (r, v, m) orbital trajectory simulation
under primary two-body gravity, zonal harmonics (J2-J4), third-body lunar perturbations,
atmospheric drag, solar radiation pressure (SRP), and continuous thrust.

Provides selectable numerical integration backends:
- 'rk4': Fixed-step 4th-order Runge-Kutta.
- 'rk45': Adaptive step-size Dormand-Prince 5(4) with embedded LTE control.
"""

from typing import Tuple, Optional, Callable
import numpy as np

from core.constants import G_EARTH, R_EARTH, G0
from core.forces import (
    accel_earth_gravity,
    accel_j2_perturbation,
    accel_j3_perturbation,
    accel_j4_perturbation,
    accel_lunar_gravity,
    accel_atmospheric_drag,
    accel_solar_radiation_pressure,
    accel_electric_prograde,
)
from core.integrators import rk4, rk45_adaptive


class SpacecraftPropagator:
    """
    Unified orbital propagation engine with modular force additions
    and multiple numerical integration backends.
    """

    def __init__(
        self,
        mu: float = G_EARTH,
        r_body: float = R_EARTH,
        use_j2: bool = True,
        use_j3: bool = False,
        use_j4: bool = False,
        use_lunar: bool = False,
        use_drag: bool = False,
        use_srp: bool = False,
        use_thrust: bool = False,
        cd: float = 2.2,
        area_drag: float = 1.0,
        cr: float = 1.8,
        area_srp: float = 1.0,
        thrust_mag: float = 0.0,
        isp: float = 3000.0,
        thrust_steering_law: Optional[Callable[[float, np.ndarray, np.ndarray, float], np.ndarray]] = None,
    ):
        self.mu = mu
        self.r_body = r_body
        self.use_j2 = use_j2
        self.use_j3 = use_j3
        self.use_j4 = use_j4
        self.use_lunar = use_lunar
        self.use_drag = use_drag
        self.use_srp = use_srp
        self.use_thrust = use_thrust

        self.cd = cd
        self.area_drag = area_drag
        self.cr = cr
        self.area_srp = area_srp
        self.thrust_mag = thrust_mag
        self.isp = isp
        self.thrust_steering_law = thrust_steering_law

    def _derivatives_6dof(self, t: float, state: np.ndarray, mass: float) -> np.ndarray:
        """Evaluates time derivative for 6-DOF state [x, y, z, vx, vy, vz]."""
        r = state[0:3]
        v = state[3:6]

        # Primary central point-mass acceleration
        acc = accel_earth_gravity(r)

        # Geopotential zonal harmonics
        if self.use_j2:
            acc = acc + accel_j2_perturbation(r)
        if self.use_j3:
            acc = acc + accel_j3_perturbation(r)
        if self.use_j4:
            acc = acc + accel_j4_perturbation(r)

        # Third-body lunar gravity
        if self.use_lunar:
            acc = acc + accel_lunar_gravity(r, t)

        # Atmospheric drag
        if self.use_drag:
            acc = acc + accel_atmospheric_drag(r, v, cd=self.cd, area=self.area_drag, mass=mass)

        # Solar Radiation Pressure (cannonball model with cylindrical shadow)
        if self.use_srp:
            acc = acc + accel_solar_radiation_pressure(r, cr=self.cr, area=self.area_srp, mass=mass)

        # Continuous thrust acceleration
        if self.use_thrust:
            if self.thrust_steering_law is not None:
                acc = acc + self.thrust_steering_law(t, r, v, mass)
            elif self.thrust_mag > 0.0:
                acc = acc + accel_electric_prograde(v, thrust_mag=self.thrust_mag, mass=mass)

        return np.concatenate([v, acc])

    def _derivatives_7dof(self, t: float, state: np.ndarray) -> np.ndarray:
        """Evaluates time derivative for 7-DOF state [x, y, z, vx, vy, vz, m]."""
        mass = float(max(state[6], 1e-3))
        derivs_6dof = self._derivatives_6dof(t, state[:6], mass=mass)

        # Mass depletion rate: m_dot = - Thrust / (Isp * g0)
        if self.use_thrust and self.thrust_mag > 0.0:
            m_dot = -self.thrust_mag / (self.isp * G0)
        else:
            m_dot = 0.0

        return np.append(derivs_6dof, m_dot)

    def propagate(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        t_span: float,
        dt: float = 10.0,
        mass0: Optional[float] = None,
        method: str = "rk4",
        rtol: float = 1e-8,
        atol: float = 1e-10,
        h_min: float = 1e-4,
        h_max: float = 86400.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagates spacecraft state vectors over the specified duration.

        Parameters
        ----------
        r0 : np.ndarray
            Initial position vector [x, y, z] in meters.
        v0 : np.ndarray
            Initial velocity vector [vx, vy, vz] in meters/second.
        t_span : float
            Total propagation duration in seconds (0 to t_span).
        dt : float, optional
            Fixed time step for 'rk4' or initial step candidate for 'rk45' (default: 10.0 s).
        mass0 : Optional[float], optional
            Initial spacecraft mass in kg. If provided, integrates as 7-DOF [r, v, m].
            If None, integrates as 6-DOF [r, v] using 1000 kg for drag/SRP calculations.
        method : str, optional
            Integration routine: 'rk4' (fixed step) or 'rk45' (adaptive Dormand-Prince).
            Default is 'rk4'.
        rtol : float, optional
            Relative error tolerance for 'rk45' (default: 1e-8).
        atol : float, optional
            Absolute error tolerance for 'rk45' (default: 1e-10).
        h_min : float, optional
            Minimum time step for 'rk45' in seconds (default: 1e-4 s).
        h_max : float, optional
            Maximum time step for 'rk45' in seconds (default: 86400.0 s).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - times: 1D array of time stamps (seconds).
            - states: 2D array of state vectors shape (N, 6) or (N, 7).
        """
        r0_arr = np.asarray(r0, dtype=np.float64)
        v0_arr = np.asarray(v0, dtype=np.float64)

        if mass0 is not None:
            initial_state = np.concatenate([r0_arr, v0_arr, [float(mass0)]])
            derivs = self._derivatives_7dof
        else:
            initial_state = np.concatenate([r0_arr, v0_arr])
            constant_mass = 1000.0
            derivs = lambda t, y: self._derivatives_6dof(t, y, mass=constant_mass)

        chosen_method = (method or "rk4").strip().lower()

        if chosen_method == "rk4":
            return rk4(
                derivs_func=derivs,
                t_span=(0.0, float(t_span)),
                y0=initial_state,
                dt=float(dt),
            )
        elif chosen_method in ("rk45", "dopri5"):
            return rk45_adaptive(
                derivs_func=derivs,
                t_span=(0.0, float(t_span)),
                y0=initial_state,
                rtol=float(rtol),
                atol=float(atol),
                h_init=float(dt),
                h_min=float(h_min),
                h_max=float(h_max),
            )
        else:
            raise ValueError(
                f"Unsupported numerical integration method '{method}'. Choose 'rk4' or 'rk45'."
            )
