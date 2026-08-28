"""
Numerical Ordinary Differential Equation (ODE) solvers.
"""
from typing import Callable
import numpy as np


def rk4_step(func: Callable[[float, np.ndarray], np.ndarray], t: float, state: np.ndarray, dt: float) -> np.ndarray:
    """
    Classical 4th-Order Runge-Kutta numerical integration step.
    Advances the state vector by dt.
    """
    k1 = func(t, state)
    k2 = func(t + 0.5 * dt, state + 0.5 * dt * k1)
    k3 = func(t + 0.5 * dt, state + 0.5 * dt * k2)
    k4 = func(t + dt, state + dt * k3)

    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)