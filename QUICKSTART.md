# Quickstart Guide

A step-by-step operational guide to configuring the environment, installing dependencies, and executing baseline orbital mechanics simulations with **NewtonianPropagator**.

---

## 1. Prerequisites

Ensure you have either **Conda / Mamba** (recommended) or **Python 3.10+** installed.

---

## 2. Clone & Environment Setup

### Option A: Using Conda (Recommended)
```bash
# Clone the repository
git clone [https://github.com/](https://github.com/)<your-username>/NewtonianPropagator.git
cd NewtonianPropagator

# Create and activate the conda environment
conda env create -f environment.yml
conda activate astrodynamics
```

### Option B: Using Pip & Virtual Environment
```bash
# Clone the repository
git clone [https://github.com/](https://github.com/)<your-username>/NewtonianPropagator.git
cd NewtonianPropagator

# Create and activate a venv
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
---

## 3. Verify the Installation
Run the automated numerical validation suite to verify the RK4 integrator's symplectic-like energy conservation:
```bash
python -m unittest tests/test_energy_conservation.py
```
Expected output:
```text
--- Energy Conservation Verification ---
Number of Orbits: 5
Initial Specific Energy: -28975901.5995 J/kg
Maximum Relative Energy Drift: 4.52e-12
.
----------------------------------------------------------------------
Ran 1 test in 0.342s

OK
```
---

## 4. Run Interactive CLI
Launch the main interactive selector to run any scenario and generate 3D/2D plots:
```bash
python main.py
```
Expected output:
```text
=================================================================
      NEWTONIAN MULTI-BODY & SWARM DYNAMICS PROPAGATOR
=================================================================
1. Scenario 1: Lunar 3rd-Body & Impulsive Directional Burn
2. Scenario 2: LEO Trajectory with Drag, SRP & J2 Precession
3. Scenario 3: 30-Day Electric Low-Thrust Orbit Raising Spiral
4. Scenario 4: Higher-Order Harmonics (J3, J4) & Frozen Orbit
5. Scenario 5: Multi-Agent Satellite Swarm & LVLH Relative Motion
6. Scenario 6: Cislunar Free-Return Trajectory & Lagrange Points
7. Exit
=================================================================
Select a scenario to run [1-7]:
```
---

## 5. Run Individual Scenarios Directly
You can execute any scenario directly using Python's `-m` module flag:
```bash
# Continuous low-thrust spiral with mass depletion
python -m examples.ex03_electric_orbit_raising

# Multi-satellite formation flying in the LVLH frame
python -m examples.ex05_satellite_swarm_lvlh

# Apollo-style Earth-Moon figure-8 free-return & Lagrange points
python -m examples.ex06_cislunar_free_return
```
---

## 6. Minimal Python Code Example
You can write custom simulations inside `customscripts/template_custom_orbit.py` or script them directly:

```python
import numpy as np
from core.propagator import SpacecraftPropagator
from core.constants import R_EARTH, G_EARTH

# 1. Define initial state (500 km altitude circular orbit)
r0 = np.array([R_EARTH + 500_000.0, 0.0, 0.0])
v_mag = np.sqrt(G_EARTH / np.linalg.norm(r0))
v0 = np.array([0.0, v_mag, 0.0])

# 2. Configure propagator with perturbations
engine = SpacecraftPropagator(
    mass=500.0,
    use_j2=True,
    use_j3=True,
    use_drag=True,
    drag_area=2.0,
    cd=2.2
)

# 3. Propagate for 3 orbits (~5 hours)
period = 2.0 * np.pi * np.sqrt(np.linalg.norm(r0)**3 / G_EARTH)
times, states = engine.propagate(r0, v0, t_span=3 * period, dt=5.0)

# 4. Render 3D orbit plot
SpacecraftPropagator.plot_3d(states, title="Custom 500 km LEO Orbit with Drag & J2/J3")
```
---
