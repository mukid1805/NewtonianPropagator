# NewtonianPropagator: Multi-Body Orbital Dynamics & Swarm Simulation Engine

[![CI Test Suite](https://github.com/mukid1805/NewtonianPropagator/actions/workflows/tests.yml/badge.svg)](https://github.com/mukid1805/NewtonianPropagator/actions/workflows/tests.yml)

A high-fidelity, modular astrodynamics simulation suite written in Python. It supports non-linear 6-DOF/7-DOF orbital mechanics, environmental perturbation superposition, multi-agent relative motion (LVLH / Hill's frame), and Circular Restricted Three-Body Problem (CR3BP) cislunar dynamics.

---

## Key Capabilities

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

---
## Repository Architecture

```text
NewtonianPropagator/
├── .github/
│   └── workflows/
│       └── tests.yml      # Automated CI test suite
├── core/
│   ├── __init__.py        # Core package definitions
│   ├── constants.py       # Planetary, gravitational, and astronomical constants
│   ├── cr3bp.py           # CR3BP synodic dynamics, Lagrange solvers, frame transforms
│   ├── forces.py          # Vectorized acceleration models (Gravity, J2-J4, Drag, SRP, Thrust)
│   ├── integrators.py     # Classical 4th-Order Runge-Kutta numerical solver
│   ├── propagator.py      # Unified 6-DOF / 7-DOF spacecraft propagation engine
│   └── swarm.py           # Multi-agent constellation and relative motion engine
├── customscripts/
│   ├── __init__.py
│   └── template_custom_orbit.py  # Parametric sandbox for custom mission design
├── examples/
│   ├── __init__.py
│   ├── ex01_lunar_impulsive_transfer.py
│   ├── ex02_drag_and_srp_decay.py
│   ├── ex03_electric_orbit_raising.py
│   ├── ex04_frozen_orbit_j3_j4.py
│   ├── ex05_satellite_swarm_lvlh.py
│   └── ex06_cislunar_free_return.py
├── tests/
│   ├── __init__.py
│   └── test_energy_conservation.py # Numerical verification of RK4 energy conservation
├── .gitignore             # Git exclusion rules
├── environment.yml        # Conda environment definition
├── main.py                # Interactive CLI menu
├── QUICKSTART.md          # Rapid deployment instructions
├── README.md              # Project documentation
└── requirements.txt       # Pip requirements