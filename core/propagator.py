"""
Unified Spacecraft Propagator engine supporting 6-DOF (r, v) and 7-DOF (r, v, m) dynamics.
"""
import numpy as np
import matplotlib.pyplot as plt

from core.constants import R_EARTH, G0
from core.integrators import rk4_step
from core.forces import (
    accel_earth_gravity,
    accel_j2_perturbation,
    accel_j3_perturbation,
    accel_j4_perturbation,
    accel_lunar_gravity,
    accel_atmospheric_drag,
    accel_solar_radiation_pressure,
    accel_fixed_thrust,
    accel_electric_prograde,
    rv_to_keplerian
)


class SpacecraftPropagator:
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
        use_srp: bool = False
    ):
        self.mass = mass
        self.drag_area = drag_area
        self.cd = cd
        self.srp_area = srp_area
        self.cr = cr
        self.isp = isp

        # Perturbation flags
        self.use_j2 = use_j2
        self.use_j3 = use_j3
        self.use_j4 = use_j4
        self.use_lunar = use_lunar
        self.use_drag = use_drag
        self.use_srp = use_srp

        # Thrust configuration
        self.thrust_mode = "none"
        self.thrust_params = {}

    def configure_fixed_burn(self, start_t: float, duration: float, thrust_vec: np.ndarray):
        """Configure a directional thrust burn (thrust_vec in Newtons)."""
        self.thrust_mode = "fixed"
        self.thrust_params = {"start_t": start_t, "duration": duration, "vec": thrust_vec}

    def configure_electric_burn(self, thrust_magnitude: float):
        """Configure continuous prograde low thrust (thrust_magnitude in Newtons)."""
        self.thrust_mode = "electric_prograde"
        self.thrust_params = {"thrust": thrust_magnitude}

    def derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Summation Junction:
        Computes total derivative vector [dr/dt, dv/dt, dm/dt].
        """
        r = state[0:3]
        v = state[3:6]
        current_mass = state[6] if len(state) == 7 else self.mass

        # Central Newtonian gravity
        a_total = accel_earth_gravity(r)

        # Superposition of Perturbations
        if self.use_j2:
            a_total += accel_j2_perturbation(r)
        if self.use_j3:
            a_total += accel_j3_perturbation(r)
        if self.use_j4:
            a_total += accel_j4_perturbation(r)
        if self.use_lunar:
            a_total += accel_lunar_gravity(r, t)
        if self.use_drag:
            a_total += accel_atmospheric_drag(r, v, self.cd, self.drag_area, current_mass)
        if self.use_srp:
            a_total += accel_solar_radiation_pressure(r, self.cr, self.srp_area, current_mass)

        # Thrust & Mass Depletion
        m_dot = 0.0
        if self.thrust_mode == "fixed":
            a_total += accel_fixed_thrust(
                t,
                self.thrust_params["start_t"],
                self.thrust_params["duration"],
                self.thrust_params["vec"],
                current_mass
            )
        elif self.thrust_mode == "electric_prograde":
            t_mag = self.thrust_params["thrust"]
            if current_mass > 20.0:  # Minimum dry mass bound
                a_total += accel_electric_prograde(v, t_mag, current_mass)
                m_dot = -t_mag / (G0 * self.isp)

        if len(state) == 7:
            return np.concatenate((v, a_total, [m_dot]))
        return np.concatenate((v, a_total))

    def propagate(self, r0: np.ndarray, v0: np.ndarray, t_span: float, dt: float, track_mass: bool = False):
        """Propagate state vector across time span."""
        num_steps = int(t_span / dt)
        times = np.linspace(0, t_span, num_steps)
        state_dim = 7 if track_mass else 6
        states = np.zeros((num_steps, state_dim))

        states[0] = np.concatenate((r0, v0, [self.mass])) if track_mass else np.concatenate((r0, v0))

        for i in range(1, num_steps):
            states[i] = rk4_step(self.derivatives, times[i - 1], states[i - 1], dt)

        return times, states

    @staticmethod
    def plot_3d(states: np.ndarray, title: str = "Trajectory"):
        """Plot the computed 3D orbit trajectory around a scaled Earth sphere."""
        x_km = states[:, 0] / 1000.0
        y_km = states[:, 1] / 1000.0
        z_km = states[:, 2] / 1000.0

        fig = plt.figure(figsize=(10.0, 8.0))
        ax = fig.add_subplot(111, projection='3d')

        # Draw Earth
        r_earth_km = R_EARTH / 1000.0
        u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:15j]
        ax.plot_wireframe(
            r_earth_km * np.cos(u) * np.sin(v),
            r_earth_km * np.sin(u) * np.sin(v),
            r_earth_km * np.cos(v),
            color='dodgerblue', alpha=0.25, label='Earth'
        )

        # Plot trajectory
        ax.plot(x_km, y_km, z_km, color='crimson', linewidth=1.5, label='Trajectory')
        ax.scatter(x_km[0], y_km[0], z_km[0], color='forestgreen', s=50, label='Start')
        ax.scatter(x_km[-1], y_km[-1], z_km[-1], color='black', s=50, label='End')

        # Equal aspect ratio scaling
        max_r = np.array([x_km.max() - x_km.min(), y_km.max() - y_km.min(), z_km.max() - z_km.min()]).max() / 2.0
        mid_x = (x_km.max() + x_km.min()) * 0.5
        mid_y = (y_km.max() + y_km.min()) * 0.5
        mid_z = (z_km.max() + z_km.min()) * 0.5
        ax.set_xlim(mid_x - max_r, mid_x + max_r)
        ax.set_ylim(mid_y - max_r, mid_y + max_r)
        ax.set_zlim(mid_z - max_r, mid_z + max_r)

        ax.set_xlabel('ECI X (km)')
        ax.set_ylabel('ECI Y (km)')
        ax.set_zlabel('ECI Z (km)')
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        plt.show()