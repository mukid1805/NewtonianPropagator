"""
Starter Template for Interplanetary Trajectory Design & Lambert Solvers.
Location: customscripts/template_lambert_transfer.py
"""

from pathlib import Path
import sys

# Add project root directory to sys.path for standalone script execution
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.ephemeris import MU_SUN, get_planet_state
from core.lambert import solve_lambert
import matplotlib.pyplot as plt
import numpy as np


def run_mission_analysis():
  print('=' * 60)
  print('      INTERPLANETARY LAMBERT TRANSFER EXPERIMENT')
  print('=' * 60)

  # ---------------------------------------------------------
  # 1. MISSION DATES & EPHEMERIDES (MJD2000)
  # ---------------------------------------------------------
  dep_epoch = 9812.0
  arr_epoch = 10122.0
  tof_days = arr_epoch - dep_epoch
  tof_sec = tof_days * 86400.0

  # Retrieve planetary states in heliocentric frame
  r_earth, v_earth = get_planet_state('earth', mjd2000=dep_epoch)
  r_mars, v_mars = get_planet_state('mars', mjd2000=arr_epoch)

  print(f'Departure Epoch:   {dep_epoch:.1f} MJD2000')
  print(f'Arrival Epoch:     {arr_epoch:.1f} MJD2000')
  print(f'Time of Flight:    {tof_days:.1f} days ({tof_sec / 1e6:.2f} Ms)')

  # ---------------------------------------------------------
  # 2. SOLVE LAMBERT BOUNDARY VALUE PROBLEM
  # ---------------------------------------------------------
  v_sc_dep, v_sc_arr = solve_lambert(
      r_earth, r_mars, tof_sec, MU_SUN, prograde=True
  )

  # ---------------------------------------------------------
  # 3. ENERGETICS & DELTA-V COMPUTATION
  # ---------------------------------------------------------
  # Hyperbolic excess velocity at departure (Earth)
  v_inf_dep = v_sc_dep - v_earth
  c3_dep = float(np.linalg.norm(v_inf_dep) ** 2)

  # Hyperbolic excess velocity at arrival (Mars)
  v_inf_arr = v_sc_arr - v_mars
  v_inf_arr_mag = float(np.linalg.norm(v_inf_arr))

  print('\n' + '-' * 40)
  print(f'Departure C3 Energy:         {c3_dep:.2f} km^2/s^2')
  print(f'Departure |v_inf|:           {np.sqrt(c3_dep):.3f} km/s')
  print(f'Arrival |v_inf| at Mars:     {v_inf_arr_mag:.3f} km/s')
  print('-' * 40)


if __name__ == '__main__':
  run_mission_analysis()