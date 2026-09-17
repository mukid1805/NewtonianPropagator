r"""Unit and Validation Tests for State Transition Matrix and Targeting Solvers."""

import unittest
import numpy as np

from core.constants import G_EARTH, R_EARTH
from core.propagator import SpacecraftPropagator
from core.targeting import eci_to_lvlh, solve_complete_rendezvous


class TestStmAndTargeting(unittest.TestCase):
    r"""Test verification suite for STM symplectic invariants and targeting convergence."""

    def setUp(self) -> None:
        r"""Set up standard circular reference orbits and propagator engines."""
        self.r_orbit = R_EARTH + 400_000.0
        self.v_circ = np.sqrt(G_EARTH / self.r_orbit)
        self.period = 2.0 * np.pi * np.sqrt(self.r_orbit**3 / G_EARTH)

        self.engine_kepler = SpacecraftPropagator(use_j2=False)
        self.engine_j2 = SpacecraftPropagator(use_j2=True)

    def test_stm_symplectic_determinant(self) -> None:
        r"""Verify Liouville's theorem that det(Phi(t, t0)) = 1 under conservative gravity."""
        r0 = np.array([self.r_orbit, 0.0, 0.0])
        v0 = np.array([0.0, self.v_circ, 0.0])

        # Propagate over half an orbital period
        t_span = 0.5 * self.period
        times, states, stms = self.engine_kepler.propagate_with_stm(r0, v0, t_span=t_span, dt=10.0)

        # Confirm identity at initial epoch
        np.testing.assert_allclose(stms[0], np.eye(6), atol=1e-12)

        # Confirm unity determinant along trajectory steps
        for i in range(len(times)):
            det_phi = np.linalg.det(stms[i])
            self.assertAlmostEqual(det_phi, 1.0, places=4)

    def test_covariance_propagation_symmetry(self) -> None:
        r"""Verify that propagated covariance P(t) preserves symmetry and positive definiteness."""
        r0 = np.array([self.r_orbit, 0.0, 0.0])
        v0 = np.array([0.0, self.v_circ, 0.0])

        # Diagonal 1-sigma initial covariance
        p0 = np.diag([100.0, 100.0, 100.0, 0.01, 0.01, 0.01])

        times, states, covs = self.engine_kepler.propagate_covariance(
            r0=r0,
            v0=v0,
            cov0=p0,
            t_span=1800.0,
            dt=30.0,
        )

        p_final = covs[-1]
        # Confirm symmetry P = P^T
        np.testing.assert_allclose(p_final, p_final.T, atol=1e-8)

        # Confirm positive definiteness (eigenvalues > 0)
        eigenvalues = np.linalg.eigvals(p_final)
        self.assertTrue(np.all(eigenvalues > 0.0))

    def test_two_impulse_rendezvous_convergence(self) -> None:
        r"""Verify sub-centimetre targeting convergence under Earth J2 geopotential."""
        r0_c = np.array([self.r_orbit - 15_000.0, 0.0, 0.0])
        v0_c = np.array([0.0, np.sqrt(G_EARTH / (self.r_orbit - 15_000.0)), 0.0])

        # Target offset by 0.8 degrees true anomaly
        theta = np.radians(0.8)
        r0_t = np.array([self.r_orbit * np.cos(theta), self.r_orbit * np.sin(theta), 0.0])
        v0_t = np.array([-self.v_circ * np.sin(theta), self.v_circ * np.cos(theta), 0.0])

        tof = 1.25 * self.period

        sol = solve_complete_rendezvous(
            engine=self.engine_j2,
            r0_chaser=r0_c,
            v0_chaser=v0_c,
            r0_target=r0_t,
            v0_target=v0_t,
            tof=tof,
            tol_m=0.01,
            max_iter=10,
            dt=15.0,
        )

        self.assertTrue(sol["iterations"] <= 10)
        self.assertLessEqual(sol["final_miss_m"], 0.01)
        self.assertLess(sol["relative_vf_mag"], 1e-6)

    def test_rk4_vs_rk45_targeting_delta_v_agreement(self) -> None:
        r"""Verify that fixed RK4 and adaptive RK45 solve for identical delta-V within 1 mm/s."""
        # Chaser in lower phasing orbit (-15 km altitude)
        r0_c = np.array([self.r_orbit - 15_000.0, 0.0, 0.0], dtype=np.float64)
        v0_c = np.array(
            [0.0, np.sqrt(G_EARTH / (self.r_orbit - 15_000.0)), 0.0], dtype=np.float64
        )

        # Target offset ahead by 0.5 degrees true anomaly
        theta = np.radians(0.5)
        r0_t = np.array(
            [self.r_orbit * np.cos(theta), self.r_orbit * np.sin(theta), 0.0],
            dtype=np.float64,
        )
        v0_t = np.array(
            [-self.v_circ * np.sin(theta), self.v_circ * np.cos(theta), 0.0],
            dtype=np.float64,
        )

        tof = 1.25 * self.period

        # Solve targeting via fixed-step RK4
        sol_rk4 = solve_complete_rendezvous(
            engine=self.engine_j2,
            r0_chaser=r0_c,
            v0_chaser=v0_c,
            r0_target=r0_t,
            v0_target=v0_t,
            tof=tof,
            tol_m=0.01,
            max_iter=15,
            method="rk4",
            dt=10.0,
        )

        # Solve targeting via adaptive-step RK45
        sol_rk45 = solve_complete_rendezvous(
            engine=self.engine_j2,
            r0_chaser=r0_c,
            v0_chaser=v0_c,
            r0_target=r0_t,
            v0_target=v0_t,
            tof=tof,
            tol_m=0.01,
            max_iter=15,
            method="rk45",
            dt=10.0,
        )

        # Both backends must converge to sub-centimetre precision
        self.assertLessEqual(sol_rk4["final_miss_m"], 0.01)
        self.assertLessEqual(sol_rk45["final_miss_m"], 0.01)

        # Total Delta-V budget must agree within 1 mm/s (0.001 m/s)
        self.assertAlmostEqual(
            sol_rk4["delta_v_total"],
            sol_rk45["delta_v_total"],
            delta=1e-3,
        )

        # Individual impulse vector components must agree within 1 mm/s
        np.testing.assert_allclose(sol_rk4["delta_v0"], sol_rk45["delta_v0"], atol=1e-3)
        np.testing.assert_allclose(sol_rk4["delta_vf"], sol_rk45["delta_vf"], atol=1e-3)

    def test_eci_to_lvlh_frame_origin(self) -> None:
        r"""Verify that target position evaluated in its own LVLH frame is identically zero."""
        r_t = np.array([self.r_orbit, 0.0, 0.0])
        v_t = np.array([0.0, self.v_circ, 0.0])

        r_rel, v_rel = eci_to_lvlh(r_t, v_t, r_t, v_t)
        np.testing.assert_allclose(r_rel, np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(v_rel, np.zeros(3), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
