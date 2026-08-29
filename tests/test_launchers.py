"""
Unit tests for core/launchers.py.
Validates launch vehicle catalog lookup, C3 capacity curve bounds,
and multi-stage Tsiolkovsky Delta-V equations.
"""
import unittest
from core.launchers import get_launcher, list_available_launchers, StageSpec


class TestLaunchers(unittest.TestCase):

    def test_catalog_retrieval(self):
        """Ensure all primary catalog launchers are retrievable."""
        f9 = get_launcher("falcon_9_reusable")
        self.assertEqual(f9.operator, "SpaceX")
        self.assertEqual(f9.status, "active")

        saturn_v = get_launcher("saturn_v")
        self.assertEqual(saturn_v.status, "legacy")
        self.assertGreater(saturn_v.tli_capacity_kg, 40000.0)

    def test_c3_payload_capacity_decay(self):
        """As C3 energy increases, delivered payload mass must monotonically decrease."""
        f9_exp = get_launcher("falcon_9_expendable")
        mass_c3_0 = f9_exp.payload_for_c3(0.0)
        mass_c3_15 = f9_exp.payload_for_c3(15.0)
        mass_c3_30 = f9_exp.payload_for_c3(30.0)

        self.assertGreater(mass_c3_0, mass_c3_15)
        self.assertGreater(mass_c3_15, mass_c3_30)

        # C3 exceeding max capability must return 0.0 kg
        self.assertEqual(f9_exp.payload_for_c3(200.0), 0.0)

    def test_stage_tsiolkovsky_delta_v(self):
        """Check Tsiolkovsky calculation for a simple stage."""
        # 10,000 kg propellant, 1,000 kg dry, 300s Isp, 500 kg payload
        # Initial mass = 11,500 kg, Final mass = 1,500 kg
        stage = StageSpec("Test Upper Stage", propellant_mass_kg=10000.0, dry_mass_kg=1000.0, isp_vac_s=300.0, thrust_vac_n=50e3)
        dv = stage.stage_delta_v(payload_mass_kg=500.0)

        # DeltaV = 9.80665 * 300 * ln(11500 / 1500) = 2941.995 * 2.03688 = ~5992.5 m/s
        self.assertAlmostEqual(dv, 5992.5, delta=5.0)

    def test_multi_stage_stack_delta_v(self):
        """Verify sequential multi-stage delta-V calculation."""
        electron = get_launcher("electron")
        dv_total = electron.total_vehicle_delta_v(payload_mass_kg=150.0)
        # Total orbital injection Delta-V should be in the 9,000 to 11,000 m/s range
        self.assertGreater(dv_total, 8500.0)
        self.assertLess(dv_total, 12000.0)

    def test_isro_fleet_specifications(self):
        """Verify parameters and C3 calculations for the ISRO vehicle fleet."""
        # 1. LVM3 (Heavy lunar/GTO launcher)
        lvm3 = get_launcher("isro_lvm3")
        self.assertEqual(lvm3.operator, "ISRO")
        self.assertEqual(lvm3.gto_capacity_kg, 4300.0)
        self.assertGreater(lvm3.payload_for_c3(10.0), 1500.0)

        # 2. PSLV-XL (MOM / Aditya-L1 interplanetary launcher)
        pslv = get_launcher("isro_pslv_xl")
        self.assertEqual(len(pslv.stages), 4)
        self.assertGreater(pslv.payload_for_c3(0.0), 1200.0)

        # 3. GSLV Mk II (CUS cryogenic launcher)
        gslv = get_launcher("isro_gslv_mk2")
        self.assertEqual(gslv.stages[2].name, "GS3 (Cryogenic CUS / CE-7.5)")
        self.assertAlmostEqual(gslv.stages[2].isp_vac_s, 454.0, places=1)

        # 4. SSLV (Small-sat micro launcher)
        sslv = get_launcher("isro_sslv")
        self.assertEqual(sslv.leo_capacity_kg, 500.0)
        self.assertEqual(sslv.payload_for_c3(10.0), 0.0)  # Exceeds SSLV max C3

if __name__ == '__main__':
    unittest.main()