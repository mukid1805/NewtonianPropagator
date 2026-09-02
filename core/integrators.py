"""
Numerical Integration Routines for Spacecraft Trajectory Propagation.

Provides:
- rk4_step: Standard fixed-step 4th-order Runge-Kutta single step.
- rk4: Fixed-step Runge-Kutta numerical propagator.
- rk45_adaptive: Adaptive step-size Dormand-Prince (DOPRI5) integrator
  with embedded local truncation error (LTE) control and FSAL optimization.
"""

from typing import Callable, Tuple, Optional
import numpy as np


# ============================================================================
# DORMAND-PRINCE 5(4) BUTCHER TABLEAU COEFFICIENTS
# ============================================================================
# Nodes (c_i)
C2 = 1.0 / 5.0
C3 = 3.0 / 10.0
C4 = 4.0 / 5.0
C5 = 8.0 / 9.0
C6 = 1.0
C7 = 1.0

# Runge-Kutta Matrix (a_ij)
A21 = 1.0 / 5.0

A31 = 3.0 / 40.0
A32 = 9.0 / 40.0

A41 = 44.0 / 45.0
A42 = -56.0 / 15.0
A43 = 32.0 / 9.0

A51 = 19372.0 / 6561.0
A52 = -25360.0 / 2187.0
A53 = 64448.0 / 6561.0
A54 = -212.0 / 729.0

A61 = 9017.0 / 3168.0
A62 = -355.0 / 33.0
A63 = 46732.0 / 5247.0
A64 = 49.0 / 176.0
A65 = -5103.0 / 18656.0

A71 = 35.0 / 384.0
A72 = 0.0
A73 = 500.0 / 1113.0
A74 = 125.0 / 192.0
A75 = -2187.0 / 6784.0
A76 = 11.0 / 84.0

# 5th-Order Weights (b_i) -> Matches row 7 of A for FSAL property
B1 = 35.0 / 384.0
B2 = 0.0
B3 = 500.0 / 1113.0
B4 = 125.0 / 192.0
B5 = -2187.0 / 6784.0
B6 = 11.0 / 84.0
B7 = 0.0

# 4th-Order Embedded Weights (b*_i)
BH1 = 5179.0 / 57600.0
BH2 = 0.0
BH3 = 7571.0 / 16695.0
BH4 = 393.0 / 640.0
BH5 = -92097.0 / 339200.0
BH6 = 187.0 / 2100.0
BH7 = 1.0 / 40.0

# Error Estimator Coefficients (E_i = b_i - b*_i)
E1 = B1 - BH1  #  71.0 / 57600.0
E2 = 0.0
E3 = B3 - BH3  # -71.0 / 16695.0
E4 = B4 - BH4  #  71.0 / 1920.0
E5 = B5 - BH5  # -17253.0 / 339200.0
E6 = B6 - BH6  #  22.0 / 525.0
E7 = B7 - BH7  # -1.0 / 40.0


# ============================================================================
# FIXED-STEP RK4 IMPLEMENTATIONS
# ============================================================================

def rk4_step(
    derivs_func: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    y: np.ndarray,
    dt: float
) -> np.ndarray:
    """
    Computes a single classical 4th-order Runge-Kutta step.

    Parameters
    ----------
    derivs_func : Callable[[float, np.ndarray], np.ndarray]
        Right-hand side vector derivative function f(t, y).
    t : float
        Current time in seconds.
    y : np.ndarray
        Current state vector.
    dt : float
        Time step size in seconds.

    Returns
    -------
    np.ndarray
        State vector advanced to t + dt.
    """
    k1 = derivs_func(t, y)
    k2 = derivs_func(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = derivs_func(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = derivs_func(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4(
    derivs_func: Callable[[float, np.ndarray], np.ndarray],
    t_span: Tuple[float, float],
    y0: np.ndarray,
    dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrates an ODE initial-value problem using fixed-step RK4.

    Parameters
    ----------
    derivs_func : Callable[[float, np.ndarray], np.ndarray]
        Right-hand side derivative function f(t, y).
    t_span : Tuple[float, float]
        Integration bounds (t_start, t_final) in seconds.
    y0 : np.ndarray
        Initial state vector of shape (n,).
    dt : float
        Fixed step size in seconds.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - t_eval: Array of time points shape (N,).
        - y_eval: Array of state vectors shape (N, n).
    """
    t_start, t_final = t_span
    t_eval = np.arange(t_start, t_final + dt, dt)
    num_steps = len(t_eval)
    state_dim = len(y0)

    y_eval = np.zeros((num_steps, state_dim), dtype=np.float64)
    y_eval[0] = np.asarray(y0, dtype=np.float64)

    for i in range(num_steps - 1):
        y_eval[i + 1] = rk4_step(derivs_func, t_eval[i], y_eval[i], dt)

    return t_eval, y_eval


# ============================================================================
# ADAPTIVE DORMAND-PRINCE RK45 IMPLEMENTATION
# ============================================================================

def rk45_adaptive(
    derivs_func: Callable[[float, np.ndarray], np.ndarray],
    t_span: Tuple[float, float],
    y0: np.ndarray,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    h_init: Optional[float] = None,
    h_min: float = 1e-6,
    h_max: float = 86400.0,
    max_steps: int = 1_000_000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrates an ODE initial-value problem using adaptive step-size Dormand-Prince 5(4).

    Features:
    - Embedded 5th-order solution with 4th-order local truncation error estimation.
    - FSAL (First-Same-As-Last) reuse: avoids redundant f(t, y) evaluation on accepted steps.
    - Proportional-Integral step-size scaling with safety factors and min/max clamps.

    Parameters
    ----------
    derivs_func : Callable[[float, np.ndarray], np.ndarray]
        Right-hand side derivative function f(t, y).
    t_span : Tuple[float, float]
        Integration span (t_start, t_final) in seconds.
    y0 : np.ndarray
        Initial state vector shape (n,).
    rtol : float, optional
        Relative error tolerance (default: 1e-8).
    atol : float, optional
        Absolute error tolerance (default: 1e-10).
    h_init : Optional[float], optional
        Initial candidate step size in seconds. If None, estimated automatically.
    h_min : float, optional
        Minimum allowable time step in seconds (default: 1e-6 s).
    h_max : float, optional
        Maximum allowable time step in seconds (default: 86400.0 s).
    max_steps : int, optional
        Maximum number of integration steps permitted before aborting.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - t_history: Array of dynamic time points shape (M,).
        - y_history: Array of state vectors shape (M, n).
    """
    t_start, t_final = float(t_span[0]), float(t_span[1])
    direction = 1.0 if t_final >= t_start else -1.0

    t = t_start
    y = np.asarray(y0, dtype=np.float64).copy()
    dim = len(y)

    # Initial derivative evaluation
    k1 = np.asarray(derivs_func(t, y), dtype=np.float64)

    # Automatic initial step-size estimation if not provided
    if h_init is None:
        d0 = np.linalg.norm(y / (atol + np.abs(y) * rtol))
        d1 = np.linalg.norm(k1 / (atol + np.abs(y) * rtol))
        if d0 < 1e-5 or d1 < 1e-5:
            h = 1e-4
        else:
            h = 0.01 * (d0 / d1)
        h = max(h_min, min(h, h_max, abs(t_final - t_start)))
    else:
        h = float(h_init)

    # Pre-allocate dynamic tracking lists
    t_list = [t]
    y_list = [y.copy()]

    # Safety parameters
    safety = 0.90
    fac_min = 0.2
    fac_max = 5.0
    step_count = 0

    while (direction * (t_final - t)) > 1e-12:
        if step_count > max_steps:
            raise RuntimeError(
                f"Adaptive RK45 exceeded maximum allowed steps ({max_steps}). "
                f"Current time: {t:.4f} s, Target: {t_final:.4f} s."
            )

        # Truncate step if approaching final time boundary
        if (direction * (t + direction * h)) > (direction * t_final):
            h_step = abs(t_final - t)
        else:
            h_step = h

        h_signed = direction * h_step

        # --- Dormand-Prince Stages ---
        k2 = derivs_func(t + C2 * h_signed, y + h_signed * (A21 * k1))
        k3 = derivs_func(t + C3 * h_signed, y + h_signed * (A31 * k1 + A32 * k2))
        k4 = derivs_func(t + C4 * h_signed, y + h_signed * (A41 * k1 + A42 * k2 + A43 * k3))
        k5 = derivs_func(t + C5 * h_signed, y + h_signed * (A51 * k1 + A52 * k2 + A53 * k3 + A54 * k4))
        k6 = derivs_func(t + C6 * h_signed, y + h_signed * (A61 * k1 + A62 * k2 + A63 * k3 + A64 * k4 + A65 * k5))

        # 5th-Order candidate solution
        y_next = y + h_signed * (A71 * k1 + A73 * k3 + A74 * k4 + A75 * k5 + A76 * k6)

        # Stage 7 evaluation for FSAL and error estimation
        k7 = derivs_func(t + h_signed, y_next)

        # Local Truncation Error vector (difference between 5th and 4th order)
        error_vec = h_signed * (E1 * k1 + E3 * k3 + E4 * k4 + E5 * k5 + E6 * k6 + E7 * k7)

        # Error scale normalization against state-dependent tolerances
        scale = atol + np.maximum(np.abs(y), np.abs(y_next)) * rtol
        error_norm = np.sqrt(np.mean((error_vec / scale) ** 2))

        # --- Step Acceptance Logic ---
        if error_norm <= 1.0:
            # Step accepted: advance state and time
            t += h_signed
            y = y_next
            step_count += 1

            t_list.append(t)
            y_list.append(y.copy())

            # FSAL Property: k1 for the next step is k7 of this step
            k1 = k7

            # Optimal step size growth factor for next step
            if error_norm == 0.0:
                scale_factor = fac_max
            else:
                scale_factor = min(fac_max, max(fac_min, safety * (error_norm ** -0.2)))
            h = min(h_max, max(h_min, h_step * scale_factor))
        else:
            # Step rejected: reduce step size and retry without advancing time
            scale_factor = max(fac_min, safety * (error_norm ** -0.2))
            h = max(h_min, h_step * scale_factor)
            if h <= h_min and h_step <= h_min:
                # Force minimum step advance to avoid deadlocks in stiff regimes
                t += h_signed
                y = y_next
                step_count += 1
                t_list.append(t)
                y_list.append(y.copy())
                k1 = derivs_func(t, y)

    return np.array(t_list, dtype=np.float64), np.array(y_list, dtype=np.float64)
