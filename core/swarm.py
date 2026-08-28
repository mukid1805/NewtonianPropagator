"""
Multi-Agent Swarm Propagator for simultaneous constellation and formation flying simulations.
"""
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt

from core.propagator import SpacecraftPropagator
from core.forces import eci_to_lvlh


class SwarmPropagator:
    def __init__(self, chief_propagator: SpacecraftPropagator):
        self.chief_prop = chief_propagator
        self.deputies: List[Dict] = []

    def add_deputy(self, name: str, propagator: SpacecraftPropagator, r0: np.ndarray, v0: np.ndarray):
        """Add a deputy agent with custom force models and initial conditions."""
        self.deputies.append({
            "name": name,
            "prop": propagator,
            "r0": r0,
            "v0": v0
        })

    def propagate_swarm(self, r0_chief: np.ndarray, v0_chief: np.ndarray, t_span: float, dt: float):
        """
        Propagate Chief and all Deputy spacecraft simultaneously.
        Returns:
            times: 1D array of time steps
            chief_states: (N, 6) array of Chief ECI states
            relative_positions: dict mapping deputy names to (N, 3) relative LVLH coordinates [m]
        """
        # 1. Propagate Chief
        times, chief_states = self.chief_prop.propagate(r0_chief, v0_chief, t_span, dt)

        # 2. Propagate each Deputy and compute relative LVLH tracks
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
    def plot_relative_motion(relative_tracks: Dict[str, np.ndarray], title: str = "Swarm Relative LVLH Motion"):
        """3D plot in the Chief-centered rotating LVLH reference frame."""
        fig = plt.figure(figsize=(10.0, 8.0))
        ax = fig.add_subplot(111, projection='3d')

        # Chief at origin (0, 0, 0)
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