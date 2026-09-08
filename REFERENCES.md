# References & Further Reading

This document outlines the foundational literature, theoretical background, and extended reading material supporting the astrodynamic models, numerical routines, and trajectory design tools in **NewtonianPropagator**.

---

## Endnotes & Implementation Notes

1. **Circular Restricted Three-Body Problem (CR3BP):** The equations of motion implemented in `core/cr3bp.py` model a massless spacecraft influenced by two primary orbiting bodies revolving around their common barycenter in circular orbits.
2. **Lambert’s Problem:** Handled in `core/lambert.py` to determine the Keplerian orbit connecting two position vectors over a specified time-of-flight, commonly used for impulsive orbital transfers and targeting.
3. **Orbital Perturbations:** Modeled in `core/forces.py`, accounting for non-spherical gravitational harmonics (such as $J_2$, $J_3$, and $J_4$), atmospheric drag, and Solar Radiation Pressure (SRP).
4. **Coordinate Frames & Formations:** Relative motion dynamics and satellite swarms in `core/swarm.py` are formulated using the Local-Vertical/Local-Horizontal (LVLH) frame.

---

## Primary References

* **Battin, R. H.** (1999). *An Introduction to the Mathematics and Methods of Astrodynamics* (Revised ed.). AIAA Education Series.
  * *Focus:* Rigorous mathematical foundations for the two-body problem, universal variable formulation of Kepler's and Lambert's problems, patched-conic approximations, and celestial mechanics.
* **Bate, R. R., Mueller, D. D., & White, J. E.** (1971). *Fundamentals of Astrodynamics*. Dover Publications.
  * *Focus:* Foundational theory for two-body orbital propagation, Kepler's problem, and impulsive orbital maneuvers.
* **Curtis, H. D.** (2020). *Orbital Mechanics for Engineering Students* (4th ed.). Butterworth-Heinemann.
  * *Focus:* Core algorithms for Lambert targeters, patched-conic trajectory approximations, planetary flybys, and gravity assists.
* **Vallado, D. A.** (2013). *Fundamentals of Astrodynamics and Applications* (4th ed.). Microcosm Press & Springer.
  * *Focus:* Standard implementations of ephemeris time frames, coordinate conversions, and environmental perturbations (drag, solar pressure, and zonal harmonics).
* **Szebehely, V.** (1967). *Theory of Orbits: The Restricted Problem of Three Bodies*. Academic Press.
  * *Focus:* Mathematical formulation of the Circular Restricted Three-Body Problem (CR3BP), equilibrium Lagrange points, and Jacobi energy conservation.

---

## Further Reading

### Interplanetary Trajectory Design & Gravity Assists
* **Prussing, J. E., & Conway, B. A.** (2012). *Orbital Mechanics*. Oxford University Press.
  * Derivations of patched-conic approximations, hyperbolic capture, and gravity-assist flybys.
  * *Repository context:* Accompanies the interactive tutorials in `notebooks/01_interplanetary_mission_design.ipynb` and `notebooks/02_gravity_assist_and_flyby_mechanics.ipynb`.

### Low-Thrust & Electric Propulsion Trajectories
* **Chobotov, V. A.** (2002). *Orbital Mechanics* (3rd ed.). AIAA Education Series.
  * Analytical and numerical treatment of continuous low-thrust spiral orbit-raising.
  * *Repository context:* Complements the continuous thrust simulation in `examples/ex03_electric_orbit_raising.py`.

### Formation Flying & Relative Motion
* **Alfriend, K. T., Vadali, S. R., Gurfil, P., How, J. P., & Breger, L.** (2010). *Spacecraft Formation Flying: Dynamics, Control and Navigation*. Butterworth-Heinemann.
  * Derivations of the Hill-Clohessy-Wiltshire (HCW) equations, curvilinear LVLH dynamics, and bounded swarm orbits.
  * *Repository context:* Theoretical basis for `core/swarm.py` and `examples/ex05_satellite_swarm_lvlh.py`.

### Numerical Integration in Astrodynamics
* **Hairer, E., Nørsett, S. P., & Wanner, G.** (1993). *Solving Ordinary Differential Equations I: Nonstiff Problems*. Springer-Verlag.
  * Analysis of explicit Runge-Kutta pairs and step-size control.
* **Hairer, E., Lubich, C., & Wanner, G.** (2006). *Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations*. Springer.
  * Study of symplectic integrators and geometric methods for conserving orbital energy and invariants over long propagation horizons.