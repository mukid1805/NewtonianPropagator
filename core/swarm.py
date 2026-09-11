r"""Multi-Agent Swarm Propagator for constellation & formation flying.

Facilitates simultaneous constellation trajectory integration, Chief-Deputy
relative tracking, and Local-Vertical/Local-Horizontal (LVLH) coordinate transformations.
"""

from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

from core.forces import eci_to_lvlh
from core.propagator import SpacecraftPropagator


class SwarmPropagator:
    r"""Multi-satellite formation trajectory propagator.

    Args:
        chief_propagator: Configured baseline propagator engine assigned to the lead (Chief) spacecraft.
    """

    def __init__(self, chief_propagator: SpacecraftPropagator):
        self.chief_prop = chief_propagator
        self.deputies: List[Dict] = []

    def add_deputy(
        self,
        name: str,
        propagator: SpacecraftPropagator,
        r0: np.ndarray,
        v0: np.ndarray
    ) -> None:
        r"""Register a deputy satellite agent within the constellation.

        Args:
            name: Unique deputy agent identifier.
            propagator: Dedicated propagator instance configured with specific force models for this agent.
            r0: Initial inertial position vector $[x, y, z]$ in meters.
            v0: Initial inertial velocity vector $[v_x, v_y, v_z]$ in meters/second.
        """
        self.deputies.append({
            "name": name,
            "prop": propagator,
            "r0": r0,
            "v0": v0
        })

    def propagate_swarm(
        self,
        r0_chief: np.ndarray,
        v0_chief: np.ndarray,
        t_span: float,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        r"""Propagate Chief and all Deputy spacecraft simultaneously.

        Args:
            r0_chief: Initial Chief position vector $[x, y, z]$ in meters.
            v0_chief: Initial Chief velocity vector $[v_x, v_y, v_z]$ in meters/second.
            t_span: Total propagation duration in seconds.
            dt: Fixed integration step size in seconds.

        Returns:
            Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]: A tuple containing:
                - **times** (`np.ndarray`): 1D array of time steps of shape `(N,)` [$\text{s}$].
                - **chief_states** (`np.ndarray`): Array of Chief ECI states of shape `(N, 6)`.
                - **relative_positions** (`Dict[str, np.ndarray]`): Mapping of deputy names
                  to relative LVLH track coordinates of shape `(N, 3)` [$\text{m}$].
        """
        times, chief_states = self.chief_prop.propagate(r0_chief, v0_chief, t_span, dt)

        relative_tracks = {}
        for dep in self.deputies:
            _, dep_states = dep["prop"].propagate(dep["r0"], dep["v0"], t_span, dt)

            num_steps = len(times)
            rel_lvlh = np.zeros((num_steps, 3))

            for i in range(num_steps):
                r_c = chief_states[i, 0:3]
                v_c = chief_states[i, 3:6]
                r_d = dep_states[i, 0:3]
                rel_lvlh[i] = eci_to_lvlh(r_c, v_c, r_d)

            relative_tracks[dep["name"]] = rel_lvlh

        return times, chief_states, relative_tracks

    @staticmethod
    def plot_relative_motion(
        relative_tracks: Dict[str, np.ndarray],
        title: str = "Swarm Relative LVLH Motion"
    ) -> None:
        r"""Render a 3D visualization in the Chief-centered rotating LVLH frame.

        Args:
            relative_tracks: Dictionary mapping deputy identifiers to `(N, 3)` relative coordinate arrays.
            title: Display title for the generated figure. Defaults to `'Swarm Relative LVLH Motion'`.
        """
        fig = plt.figure(figsize=(10.0, 8.0))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(0, 0, 0, color='gold', s=120, edgecolors='black', label='Chief (Origin)', zorder=10)

        colors = ['dodgerblue', 'crimson', 'forestgreen', 'darkviolet', 'darkorange']
        for idx, (name, track) in enumerate(relative_tracks.items()):
            color = colors[idx % len(colors)]
            ax.plot(track[:, 1], track[:, 0], track[:, 2], color=color, linewidth=1.5, label=f'{name} Path')
            ax.scatter(track[0, 1], track[0, 0], track[0, 2], color=color, marker='o', s=40, label=f'{name} Start')
            ax.scatter(track[-1, 1], track[-1, 0], track[-1, 2], color=color, marker='x', s=50, label=f'{name} End')

        ax.set_xlabel('Along-Track / In-Track [y] (m)')
        ax.set_ylabel('Radial [x] (m)')
        ax.set_zlabel('Cross-Track [z] (m)')
        ax.set_title(title)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        plt.tight_layout()
        plt.show()
