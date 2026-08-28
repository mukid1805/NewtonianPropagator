"""
Newtonian Orbital Dynamics Propagator - Main Entry Point
"""
import sys
from examples import (
    ex01_lunar_impulsive_transfer as ex1,
    ex02_drag_and_srp_decay as ex2,
    ex03_electric_orbit_raising as ex3,
    ex04_frozen_orbit_j3_j4 as ex4,
    ex05_satellite_swarm_lvlh as ex5,
    ex06_cislunar_free_return as ex6,
    ex07_earth_mars_transfer as ex7
)


def print_menu():
    print("=" * 65)
    print("      NEWTONIAN MULTI-BODY & SWARM DYNAMICS PROPAGATOR")
    print("=" * 65)
    print("1. Scenario 1: Lunar 3rd-Body & Impulsive Directional Burn")
    print("2. Scenario 2: LEO Trajectory with Drag, SRP & J2 Precession")
    print("3. Scenario 3: 30-Day Electric Low-Thrust Orbit Raising Spiral")
    print("4. Scenario 4: Higher-Order Harmonics (J3, J4) & Frozen Orbit")
    print("5. Scenario 5: Multi-Agent Satellite Swarm & LVLH Relative Motion")
    print("6. Scenario 6: Cislunar Free-Return Trajectory & Lagrange Points")
    print("7. Scenario 7: Earth-to-Mars Interplanetary Mission Design & Flight Dynamics")
    print("8. Exit")
    print("=" * 65)


def main():
    while True:
        print_menu()
        choice = input("Select a scenario to run [1-8]: ").strip()
        if choice == '1':
            ex1.run()
        elif choice == '2':
            ex2.run()
        elif choice == '3':
            ex3.run()
        elif choice == '4':
            ex4.run()
        elif choice == '5':
            ex5.run()
        elif choice == '6':
            ex6.run()
        elif choice == '7':
            ex7.run()
        elif choice == '8':
            print("Exiting.")
            sys.exit(0)
        else:
            print("Invalid selection. Please enter a digit from 1 to 7.\n")


if __name__ == '__main__':
    main()
