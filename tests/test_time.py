"""
tests/test_time.py
------------------
Unit test suite verifying astronomical epoch and time conversions.
"""
import unittest
from datetime import datetime

from core.time import datetime_to_jd, jd_to_datetime, JD_J2000


class TestTimeConversions(unittest.TestCase):

    def test_known_epoch_j2000(self):
        """Validates that J2000 epoch corresponds to JD 2451545.0."""
        dt_j2000 = datetime(2000, 1, 1, 12, 0, 0)
        jd = datetime_to_jd(dt_j2000)
        self.assertAlmostEqual(jd, JD_J2000, places=6)

    def test_bidirectional_roundtrip(self):
        """Converting datetime -> JD -> datetime preserves timestamp within 1 millisecond."""
        test_dt = datetime(2026, 8, 28, 16, 21, 26)
        jd = datetime_to_jd(test_dt)
        dt_recovered = jd_to_datetime(jd)

        # Compare total difference in seconds (must be < 1e-3 s)
        time_diff = abs((test_dt - dt_recovered).total_seconds())
        self.assertLess(time_diff, 1e-3)


if __name__ == '__main__':
    unittest.main()