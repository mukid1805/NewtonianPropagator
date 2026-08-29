# NewtonianPropagator: Multi-Body Orbital Dynamics & Swarm Simulation Engine

[![CI Test Suite](https://github.com/mukid1805/NewtonianPropagator/actions/workflows/tests.yml/badge.svg)](https://github.com/mukid1805/NewtonianPropagator/actions/workflows/tests.yml)



[![Release](https://img.shields.io/github/v/release/mukid1805/NewtonianPropagator?include_prereleases&color=blue&logo=github)](https://github.com/mukid1805/NewtonianPropagator/releases)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Last Commit](https://img.shields.io/github/last-commit/mukid1805/NewtonianPropagator?logo=github)](https://github.com/mukid1805/NewtonianPropagator/commits/main/)


---

A high-fidelity, modular astrodynamics simulation suite written in Python. It supports non-linear 6-DOF/7-DOF orbital mechanics, environmental perturbation superposition, launch vehicle $C_3$ injection modeling, multi-agent relative motion (LVLH / Hill's frame), Circular Restricted Three-Body Problem (CR3BP) cislunar dynamics, and interplanetary multi-leg gravity assist trajectory optimization.

---

## Key Capabilities


* **Launch Vehicle Injection & Mission Energetics:**
  * Empirical and analytical payload mass capacity curves as a function of characteristic energy ($C_3$).
  * Built-in launch vehicle performance profiles across multiple configurations (e.g., Falcon 9, Falcon Heavy, Atlas V, Vulcan Centaur, SLS).
  * Direct coupling between interplanetary Lambert departure energy ( $C_3$ ) and net deliverable spacecraft mass.
* **Multi-Leg Trajectories & Planetary Gravity Assists:**
  * Hyperbolic turning angle ($\delta$) and excess velocity ($\mathbf{v}_\infty$) vector matching in planetary flyby frames.
  * Automated flyby feasibility checks (periapsis altitude $h_p \ge h_{\text{safe}}$ and powered $\Delta v$ deficit computation). 
  * Patched-conic multi-leg chaining across interplanetary bodies (e.g., Earth $\to$ Venus $\to$ Mars). 
  * 3D calendar grid-search optimizer across departure, flyby, and arrival epochs for minimum - $\Delta v$ trajectory windows.
* **Interplanetary Mission Design:**
  * Universal Variable Lambert solver (Bate-Mueller-White formulation) for robust two-point boundary value targeting (elliptical, parabolic, and hyperbolic transfers).
  * Automated Porkchop Plot generation for launch window optimization scanning characteristic energy ($C_3$), arrival excess velocity ($v_\infty$), and total $\Delta v$.
  * Analytical planetary ephemerides and precision astronomical time conversions (JD, MJD, J2000 offsets).
* **Environmental Perturbation Superposition:**
  * Nonspherical Earth Geopotential Harmonics ($J_2$, $J_3$, $J_4$).
  * Atmospheric Drag with diurnal Earth rotation velocity coupling and exponential density scale height.
  * Cannonball Solar Radiation Pressure (SRP) with cylindrical Earth umbra eclipse modeling.
  * Third-Body Gravitational Perturbations (Moon direct and indirect acceleration).
* **Propulsion & Mass Depletion:**
  * Timed fixed-thrust impulsive maneuvers.
  * Continuous low-thrust electric propulsion (e.g., Hall effect thrusters) with dynamic mass flow:
    $$\dot{m} = -\frac{T}{g_0 I_{\text{sp}}}$$
* **Multi-Agent Formation Dynamics:**
  * Multi-satellite swarm propagation across identical or disparate perturbation models.
  * Real-time ECI to rotating Local-Vertical Local-Horizontal (LVLH / Hill's) frame coordinate transformations for relative motion and proximity operations.
* **Cislunar Three-Body Mechanics (CR3BP):**
  * Rotating (synodic) frame formulation for the Earth-Moon system.
  * Root-finding solvers for collinear ($L_1, L_2, L_3$) and triangular ($L_4, L_5$) equilibrium Lagrange points.
  * Free-return cislunar trajectory propagation with Jacobi Integral ($\mathcal{C}$) conservation tracking.

---

## Mathematical Formulations

### Newtonian Acceleration Superposition Junction
The equation of motion in the Earth-Centered Inertial (ECI) frame sums all central and non-spherical forces:

$$\mathbf{a}_{\text{total}} = \mathbf{a}_{\text{grav}} + \mathbf{a}_{J2} + \mathbf{a}_{J3} + \mathbf{a}_{J4} + \mathbf{a}_{\text{lunar}} + \mathbf{a}_{\text{drag}} + \mathbf{a}_{\text{SRP}} + \mathbf{a}_{\text{thrust}}$$

* **Central Body Gravity:**

$$\mathbf{a}_{\text{grav}} = -\frac{\mu_{\text{E}}}{r^3} \mathbf{r}$$

* **Atmospheric Drag:**

$$\mathbf{a}_{\text{drag}} = -\frac{1}{2} \rho(h) \frac{C_D A_{\text{drag}}}{m} \|\mathbf{v}_{\text{rel}}\| \mathbf{v}_{\text{rel}}, \quad \mathbf{v}_{\text{rel}} = \mathbf{v} - (\vec{\omega}_{\text{E}} \times \mathbf{r})$$

* **Solar Radiation Pressure (SRP):**

$$\mathbf{a}_{\text{SRP}} = P_{\text{sun}} C_R \frac{A_{\text{SRP}}}{m} \hat{\mathbf{r}}_{\text{sun}} \quad (\text{zero in Earth umbra})$$

### Interplanetary Characteristic Energy

* **Departure Characteristic Energy ($C_3$):**

$$C_3 = v_\infty^2 = \Vert{}\mathbf{v}_{\text{dep}} - \mathbf{v}_{\text{planet}}\Vert{}^2$$


* **Hyperbolic Turning Angle ($\delta$):**

$$\sin\left(\frac{\delta}{2}\right) = \frac{1}{1 + \frac{r_p v_\infty^2}{\mu_p}}$$


---
## Repository Architecture

```text
NewtonianPropagator/
├── .github/
│   └── workflows/
│       └── tests.yml                  # Automated CI test suite
├── core/
│   ├── __init__.py                    # Core package definitions
│   ├── constants.py                   # Planetary, gravitational, and astronomical constants
│   ├── cr3bp.py                       # CR3BP synodic dynamics, Lagrange solvers, frame transforms
│   ├── ephemeris.py                   # Analytical planetary ephemerides and state vectors
│   ├── flyby.py                       # Hyperbolic gravity assist mechanics and turning angle analysis
│   ├── forces.py                      # Vectorized acceleration models (Gravity, J2-J4, Drag, SRP, Thrust)
│   ├── integrators.py                 # Classical 4th-Order Runge-Kutta numerical solver
│   ├── lambert.py                     # Universal Variable Lambert problem solver
│   ├── launchers.py                   # Launch vehicle models and C3 payload capacity curves
│   ├── propagator.py                  # Unified 6-DOF / 7-DOF spacecraft propagation engine
│   ├── swarm.py                       # Multi-agent constellation and relative motion engine
│   └── time.py                        # Astronomical time conversions (JD, MJD, J2000 offsets)
├── customscripts/
│   ├── __init__.py
│   ├── template_custom_orbit.py             # Parametric sandbox for custom mission design
│   └── template_interplanetary_mission.py   # Starter template for Lambert targeting & launch analysis
├── examples/
│   ├── __init__.py
│   ├── ex01_lunar_impulsive_transfer.py
│   ├── ex02_drag_and_srp_decay.py
│   ├── ex03_electric_orbit_raising.py
│   ├── ex04_frozen_orbit_j3_j4.py
│   ├── ex05_satellite_swarm_lvlh.py
│   ├── ex06_cislunar_free_return.py
│   ├── ex07_earth_mars_transfer.py
│   └── ex08_gravity_assist_transfer.py
├── tests/
│   ├── __init__.py
│   ├── test_energy_conservation.py    # Symplectic energy drift assertions
│   ├── test_flyby.py                  # Gravity assist turning angle & Delta-V validation
│   ├── test_lambert.py                # Validation of BVP solver boundary conditions
│   ├── test_launchers.py              # Launch vehicle capacity & C3 performance curve assertions
│   └── test_time.py                   # Epoch and temporal conversion assertions
├── .gitignore                         # Git exclusion rules
├── environment.yml                    # Conda environment definition
├── main.py                            # Interactive CLI menu
├── QUICKSTART.md                      # Rapid deployment instructions
├── README.md                          # Project documentation
└── requirements.txt                   # Pip requirements
```
---
