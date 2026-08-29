"""
core/launchers.py - Launch Vehicle Performance & Payload Injection Capacity Engine.
Provides empirical C3 vs. payload curves, stage mass properties, Tsiolkovsky rocket
equation calculations, and catalog specifications for active and legacy rockets.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from core.constants import G0


@dataclass
class StageSpec:
    """Represents an individual propulsion stage."""
    name: str
    propellant_mass_kg: float
    dry_mass_kg: float
    isp_vac_s: float
    thrust_vac_n: float

    @property
    def total_mass_kg(self) -> float:
        return self.propellant_mass_kg + self.dry_mass_kg

    def stage_delta_v(self, payload_mass_kg: float) -> float:
        """Calculates ideal vacuum Delta-V (m/s) via the Tsiolkovsky Rocket Equation."""
        m_initial = self.total_mass_kg + payload_mass_kg
        m_final = self.dry_mass_kg + payload_mass_kg
        if m_final <= 0 or m_initial <= m_final:
            return 0.0
        return float(G0 * self.isp_vac_s * np.log(m_initial / m_final))


@dataclass
class LaunchVehicle:
    """
    Launch vehicle model with payload capabilities, stage breakdowns,
    and high-energy C3 injection polynomial curves.
    """
    name: str
    operator: str
    status: str  # 'active' or 'legacy'
    leo_capacity_kg: float
    gto_capacity_kg: float
    tli_capacity_kg: Optional[float] = None
    max_c3_km2_s2: float = 100.0
    # Quadratic fit: mass(C3) = c0 + c1*C3 + c2*C3^2 (where C3 is in km^2/s^2, mass in kg)
    # Stored as [c2, c1, c0] for standard numpy.polyval evaluation
    c3_poly_coeffs: Optional[Tuple[float, float, float]] = None
    stages: List[StageSpec] = field(default_factory=list)

    def payload_for_c3(self, c3_km2_s2: float) -> float:
        """
        Computes maximum deliverable payload mass (kg) for a given characteristic energy C3.

        Parameters
        ----------
        c3_km2_s2 : float
            Characteristic launch energy (v_infinity^2) in km^2/s^2.

        Returns
        -------
        float
            Payload mass in kg (returns 0.0 if C3 exceeds vehicle capability).
        """
        if self.c3_poly_coeffs is None:
            raise NotImplementedError(f"C3 polynomial model is not defined for {self.name}.")

        if c3_km2_s2 < 0.0:
            c3_km2_s2 = 0.0

        if c3_km2_s2 > self.max_c3_km2_s2:
            return 0.0

        mass = float(np.polyval(self.c3_poly_coeffs, c3_km2_s2))
        return float(max(0.0, mass))

    def total_vehicle_delta_v(self, payload_mass_kg: float) -> float:
        """
        Computes ideal multi-stage Delta-V (m/s) through sequential stage burn/jettison.
        """
        if not self.stages:
            return 0.0

        total_dv = 0.0
        current_payload = payload_mass_kg

        # Evaluate stages from upper (last) to booster (first)
        for i in reversed(range(len(self.stages))):
            stage = self.stages[i]
            dv_stage = stage.stage_delta_v(current_payload)
            total_dv += dv_stage
            # Prior stages must carry this stage's entire stack
            current_payload += stage.total_mass_kg

        return total_dv


# =============================================================================
# LAUNCH VEHICLE CATALOG (Empirical Performance & Fit Models)
# =============================================================================

LAUNCH_VEHICLE_CATALOG: Dict[str, LaunchVehicle] = {
    # --- ISRO (INDIAN SPACE RESEARCH ORGANISATION) ---
    "isro_lvm3": LaunchVehicle(
        name="LVM3 / GSLV Mk III (Chandrayaan / Gaganyaan)",
        operator="ISRO",
        status="active",
        leo_capacity_kg=10000.0,
        gto_capacity_kg=4300.0,
        tli_capacity_kg=2200.0,
        max_c3_km2_s2=45.0,
        c3_poly_coeffs=(0.32, -68.0, 2350.0),
        stages=[
            StageSpec("2x S200 Solid Boosters", propellant_mass_kg=414000.0, dry_mass_kg=62000.0, isp_vac_s=274.5, thrust_vac_n=10300e3),
            StageSpec("L110 Core (2x Vikas)", propellant_mass_kg=115000.0, dry_mass_kg=10500.0, isp_vac_s=293.0, thrust_vac_n=1598e3),
            StageSpec("C25 Cryogenic (1x CE-20)", propellant_mass_kg=28000.0, dry_mass_kg=5000.0, isp_vac_s=442.0, thrust_vac_n=200e3),
        ]
    ),
    "isro_pslv_xl": LaunchVehicle(
        name="PSLV-XL (MOM / Chandrayaan-1 / Aditya-L1)",
        operator="ISRO",
        status="active",
        leo_capacity_kg=3800.0,
        gto_capacity_kg=1425.0,
        tli_capacity_kg=1300.0,
        max_c3_km2_s2=25.0,
        c3_poly_coeffs=(0.14, -38.0, 1380.0),
        stages=[
            StageSpec("PS1 (S139 + 6x PS0M-XL)", propellant_mass_kg=210000.0, dry_mass_kg=30200.0, isp_vac_s=269.0, thrust_vac_n=4800e3),
            StageSpec("PS2 (1x Vikas)", propellant_mass_kg=41000.0, dry_mass_kg=5300.0, isp_vac_s=293.0, thrust_vac_n=800e3),
            StageSpec("PS3 (HPS3 Solid)", propellant_mass_kg=7650.0, dry_mass_kg=1050.0, isp_vac_s=295.0, thrust_vac_n=240e3),
            StageSpec("PS4 (Liquid Upper)", propellant_mass_kg=2500.0, dry_mass_kg=420.0, isp_vac_s=308.0, thrust_vac_n=14.6e3),
        ]
    ),
    "isro_gslv_mk2": LaunchVehicle(
        name="GSLV Mk II (CUS / NISAR)",
        operator="ISRO",
        status="active",
        leo_capacity_kg=5000.0,
        gto_capacity_kg=2250.0,
        tli_capacity_kg=1500.0,
        max_c3_km2_s2=35.0,
        c3_poly_coeffs=(0.22, -48.0, 1650.0),
        stages=[
            StageSpec("GS1 (S139 + 4x L40 Vikas)", propellant_mass_kg=300000.0, dry_mass_kg=48000.0, isp_vac_s=278.0, thrust_vac_n=7500e3),
            StageSpec("GS2 (L37.5 Vikas)", propellant_mass_kg=39500.0, dry_mass_kg=5000.0, isp_vac_s=295.0, thrust_vac_n=800e3),
            StageSpec("GS3 (Cryogenic CUS / CE-7.5)", propellant_mass_kg=12800.0, dry_mass_kg=2600.0, isp_vac_s=454.0, thrust_vac_n=73.5e3),
        ]
    ),
    "isro_sslv": LaunchVehicle(
        name="SSLV (Small Satellite Launch Vehicle)",
        operator="ISRO",
        status="active",
        leo_capacity_kg=500.0,
        gto_capacity_kg=0.0,
        tli_capacity_kg=0.0,
        max_c3_km2_s2=5.0,
        c3_poly_coeffs=(0.0, -10.0, 50.0),
        stages=[
            StageSpec("SS1 (Solid)", propellant_mass_kg=87000.0, dry_mass_kg=9000.0, isp_vac_s=260.0, thrust_vac_n=2600e3),
            StageSpec("SS2 (Solid)", propellant_mass_kg=7700.0, dry_mass_kg=1000.0, isp_vac_s=275.0, thrust_vac_n=250e3),
            StageSpec("SS3 (Solid)", propellant_mass_kg=4500.0, dry_mass_kg=600.0, isp_vac_s=285.0, thrust_vac_n=160e3),
            StageSpec("VTM (Velocity Trimming Module)", propellant_mass_kg=50.0, dry_mass_kg=25.0, isp_vac_s=300.0, thrust_vac_n=0.5e3),
        ]
    ),

    # --- SPACEX ---
    "falcon_9_reusable": LaunchVehicle(
        name="Falcon 9 (Drone Ship ASDS Recovery)",
        operator="SpaceX",
        status="active",
        leo_capacity_kg=17500.0,
        gto_capacity_kg=5500.0,
        tli_capacity_kg=2200.0,
        max_c3_km2_s2=45.0,
        c3_poly_coeffs=(0.32, -88.5, 3100.0),
        stages=[
            StageSpec("Stage 1 (ASDS)", propellant_mass_kg=418000.0, dry_mass_kg=22200.0, isp_vac_s=311.0, thrust_vac_n=7607e3),
            StageSpec("Stage 2 (M1D-Vac)", propellant_mass_kg=107500.0, dry_mass_kg=4000.0, isp_vac_s=348.0, thrust_vac_n=981e3),
        ]
    ),
    "falcon_9_expendable": LaunchVehicle(
        name="Falcon 9 (Fully Expendable)",
        operator="SpaceX",
        status="active",
        leo_capacity_kg=22800.0,
        gto_capacity_kg=8300.0,
        tli_capacity_kg=4020.0,
        max_c3_km2_s2=65.0,
        c3_poly_coeffs=(0.45, -112.0, 4350.0),
        stages=[
            StageSpec("Stage 1 (Exp)", propellant_mass_kg=418000.0, dry_mass_kg=20000.0, isp_vac_s=311.0, thrust_vac_n=7607e3),
            StageSpec("Stage 2 (M1D-Vac)", propellant_mass_kg=107500.0, dry_mass_kg=4000.0, isp_vac_s=348.0, thrust_vac_n=981e3),
        ]
    ),
    "falcon_heavy_expendable": LaunchVehicle(
        name="Falcon Heavy (Fully Expendable)",
        operator="SpaceX",
        status="active",
        leo_capacity_kg=63800.0,
        gto_capacity_kg=26700.0,
        tli_capacity_kg=15500.0,
        max_c3_km2_s2=120.0,
        c3_poly_coeffs=(0.62, -210.0, 16800.0),
        stages=[
            StageSpec("Booster Cores (3x)", propellant_mass_kg=1254000.0, dry_mass_kg=65000.0, isp_vac_s=311.0, thrust_vac_n=22819e3),
            StageSpec("Stage 2 (M1D-Vac)", propellant_mass_kg=107500.0, dry_mass_kg=4000.0, isp_vac_s=348.0, thrust_vac_n=981e3),
        ]
    ),

    # --- ROCKET LAB ---
    "electron": LaunchVehicle(
        name="Electron (Kick Stage / Curie)",
        operator="Rocket Lab",
        status="active",
        leo_capacity_kg=300.0,
        gto_capacity_kg=80.0,
        tli_capacity_kg=35.0,
        max_c3_km2_s2=25.0,
        c3_poly_coeffs=(0.04, -2.1, 42.0),
        stages=[
            StageSpec("Stage 1 (9x Rutherford)", propellant_mass_kg=9200.0, dry_mass_kg=950.0, isp_vac_s=311.0, thrust_vac_n=224e3),
            StageSpec("Stage 2 (1x Rutherford Vac)", propellant_mass_kg=2150.0, dry_mass_kg=250.0, isp_vac_s=343.0, thrust_vac_n=25.8e3),
            StageSpec("Curie Kick Stage", propellant_mass_kg=45.0, dry_mass_kg=12.0, isp_vac_s=313.0, thrust_vac_n=120.0),
        ]
    ),

    # --- ULA & NASA ---
    "atlas_v_551": LaunchVehicle(
        name="Atlas V 551 (Centaur)",
        operator="ULA",
        status="active",
        leo_capacity_kg=18810.0,
        gto_capacity_kg=8900.0,
        tli_capacity_kg=5200.0,
        max_c3_km2_s2=160.0,
        c3_poly_coeffs=(0.18, -62.0, 5400.0),
        stages=[
            StageSpec("CCB + 5x SRB", propellant_mass_kg=325000.0, dry_mass_kg=35000.0, isp_vac_s=311.3, thrust_vac_n=10800e3),
            StageSpec("Centaur III (RL10C-1)", propellant_mass_kg=20830.0, dry_mass_kg=2247.0, isp_vac_s=448.5, thrust_vac_n=99.2e3),
        ]
    ),
    "sls_block_1": LaunchVehicle(
        name="Space Launch System (SLS Block 1)",
        operator="NASA",
        status="active",
        leo_capacity_kg=95000.0,
        gto_capacity_kg=42000.0,
        tli_capacity_kg=27000.0,
        max_c3_km2_s2=100.0,
        c3_poly_coeffs=(0.85, -340.0, 31000.0),
        stages=[
            StageSpec("Core + 2x 5-Seg SRBs", propellant_mass_kg=2100000.0, dry_mass_kg=240000.0, isp_vac_s=360.0, thrust_vac_n=39100e3),
            StageSpec("ICPS (1x RL10B-2)", propellant_mass_kg=28500.0, dry_mass_kg=3500.0, isp_vac_s=465.5, thrust_vac_n=110e3),
        ]
    ),

    # --- HISTORIC / LEGACY ---
    "saturn_v": LaunchVehicle(
        name="Saturn V (Apollo Benchmark)",
        operator="NASA",
        status="legacy",
        leo_capacity_kg=140000.0,
        gto_capacity_kg=60000.0,
        tli_capacity_kg=48600.0,
        max_c3_km2_s2=75.0,
        c3_poly_coeffs=(1.2, -620.0, 52000.0),
        stages=[
            StageSpec("S-IC (5x F-1)", propellant_mass_kg=2150000.0, dry_mass_kg=130000.0, isp_vac_s=304.0, thrust_vac_n=35100e3),
            StageSpec("S-II (5x J-2)", propellant_mass_kg=450000.0, dry_mass_kg=38000.0, isp_vac_s=421.0, thrust_vac_n=5100e3),
            StageSpec("S-IVB (1x J-2)", propellant_mass_kg=107000.0, dry_mass_kg=11300.0, isp_vac_s=421.0, thrust_vac_n=1020e3),
        ]
    ),
    "delta_ii_7925": LaunchVehicle(
        name="Delta II 7925 (PAM-D Upper Stage)",
        operator="ULA / McDonnell Douglas",
        status="legacy",
        leo_capacity_kg=6100.0,
        gto_capacity_kg=1820.0,
        tli_capacity_kg=1280.0,
        max_c3_km2_s2=80.0,
        c3_poly_coeffs=(0.08, -24.0, 1310.0),
        stages=[
            StageSpec("Thor / Extra-Extended Core + 9x GEM", propellant_mass_kg=118000.0, dry_mass_kg=10200.0, isp_vac_s=302.0, thrust_vac_n=3500e3),
            StageSpec("Delta K (AJ10-118K)", propellant_mass_kg=6000.0, dry_mass_kg=950.0, isp_vac_s=319.2, thrust_vac_n=43.4e3),
            StageSpec("PAM-D (Star 48B Solid)", propellant_mass_kg=2010.0, dry_mass_kg=130.0, isp_vac_s=286.0, thrust_vac_n=67.1e3),
        ]
    ),
}


def get_launcher(vehicle_key: str) -> LaunchVehicle:
    """Retrieves a LaunchVehicle instance from the catalog by key."""
    key = vehicle_key.lower().strip()
    if key not in LAUNCH_VEHICLE_CATALOG:
        available = ", ".join(LAUNCH_VEHICLE_CATALOG.keys())
        raise KeyError(f"Launcher '{vehicle_key}' not found. Available launchers: {available}")
    return LAUNCH_VEHICLE_CATALOG[key]


def list_available_launchers() -> List[Dict[str, str]]:
    """Returns a list of all available launchers with their status and operator."""
    return [
        {"key": key, "name": lv.name, "operator": lv.operator, "status": lv.status}
        for key, lv in LAUNCH_VEHICLE_CATALOG.items()
    ]