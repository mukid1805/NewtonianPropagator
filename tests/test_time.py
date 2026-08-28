"""
tests/test_time.py - Validation of J2000 time conversions
"""
import unittest
from datetime import datetime
from core.time import datetime_to_jd, jd_to_datetime, date_to_mjd2000, JD_J2000


class TestTimeConversions(unittest.TestCase):

    def test_j2000_exact_epoch(self):
        """J2000.0 (2000-01-01 12:00:00) must yield JD = 2451545.0 and MJD2000 = 0.0."""
        dt_j2000 = datetime(2000, 1, 1, 12, 0, 0)
        jd = datetime_to_jd(dt_j2000)
        self.assertAlmostEqual(jd, JD_J2000, places=6)
        self.assertAlmostEqual(date_to_mjd2000(dt_j2000), 0.0, places=6)

    def test_bidirectional_roundtrip(self):
        """Converting datetime -> JD -> datetime must preserve exact timestamp."""
        test_dt = datetime(2026, 8, 28, 16, 21, 26)
        jd = datetime_to_jd(test_dt)
        dt_recovered = jd_to_datetime(jd)
        self.assertEqual(test_dt, dt_recovered)


if __name__ == '__main__':
    unittest.main()