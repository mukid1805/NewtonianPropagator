"""
Scenario 5: Multi-Agent Satellite Swarm Propagation & LVLH Frame Relative Motion.
Simulates bounded Clohessy-Wiltshire station-keeping ellipses and along-track drift.
"""
import numpy as np
from core.propagator import SpacecraftPropagator
from core.swarm import SwarmPropagator
from core.constants import G_EARTH, R_EARTH


def run():
    # 1. Chief Orbital Setup (~500 km circular LEO at 45 deg inclination)
    r_chief_mag = R_EARTH + 500_000.0
    v_chief_mag = float(np.sqrt(G_EARTH / r_chief_mag))
    inc = np.radians(45.0)

    r0_chief = np.array([r_chief_mag, 0.0, 0.0])
    v0_chief = np.array([0.0, v_chief_mag * np.cos(inc), v_chief_mag * np.sin(inc)])

    # 2. Frame vectors for initializing deputies in inertial frame
    e_r = r0_chief / r_chief_mag
    h_vec = np.cross(r0_chief, v0_chief)
    e_h = h_vec / np.linalg.norm(h_vec)
    e_theta = np.cross(e_h, e_r)

    # Mean motion n = sqrt(mu / a^3)
    n = np.sqrt(G_EARTH / (r_chief_mag ** 3))

    # 3. Deputy 1 Setup: Projected Circular Orbit (PCO - 2x1 relative ellipse of 200m radius)
    # Clohessy-Wiltshire condition: x_0 = rho/2, vy_0 = -2*n*x_0
    rho = 200.0  # meters
    dr_lvlh_1 = np.array([rho / 2.0, 0.0, rho])  # [radial, along-track, cross-track]
    dv_lvlh_1 = np.array([0.0, -2.0 * n * (rho / 2.0), 0.0])

    r0_dep1 = r0_chief + dr_lvlh_1[0] * e_r + dr_lvlh_1[1] * e_theta + dr_lvlh_1[2] * e_h
    v0_dep1 = v0_chief + dv_lvlh_1[0] * e_r + dv_lvlh_1[1] * e_theta + dv_lvlh_1[2] * e_h

    # 4. Deputy 2 Setup: Along-track follower offset by +500 m
    dr_lvlh_2 = np.array([0.0, 500.0, 0.0])
    r0_dep2 = r0_chief + dr_lvlh_2[0] * e_r + dr_lvlh_2[1] * e_theta + dr_lvlh_2[2] * e_h
    v0_dep2 = v0_chief.copy()

    # 5. Build Swarm with J2 perturbation enabled on all agents
    chief_engine = SpacecraftPropagator(use_j2=True)
    swarm = SwarmPropagator(chief_propagator=chief_engine)

    swarm.add_deputy("Deputy 1 (PCO Ellipse)", SpacecraftPropagator(use_j2=True), r0_dep1, v0_dep1)
    swarm.add_deputy("Deputy 2 (Along-Track)", SpacecraftPropagator(use_j2=True), r0_dep2, v0_dep2)

    # 6. Propagate over 4 orbital periods
    period = 2.0 * np.pi / n
    t_span = float(4.0 * period)
    dt = 5.0

    print("Propagating Multi-Agent Swarm over 4 orbits...")
    times, chief_states, relative_tracks = swarm.propagate_swarm(r0_chief, v0_chief, t_span=t_span, dt=dt)
    print(f"Swarm simulation complete across {len(times)} timesteps.")

    # 7. Visualize relative motion in LVLH frame
    SwarmPropagator.plot_relative_motion(relative_tracks, title="Swarm 3D Relative Motion in Chief LVLH Frame")


if __name__ == '__main__':
    run()