"""
core/time.py - Time & Epoch Conversion Utilities for Astrodynamics
Supports conversions between UTC Gregorian datetime, Julian Date (JD),
Modified Julian Date (MJD), and J2000.0 epoch offsets.
"""
from datetime import datetime
from typing import Union, Tuple


# Standard Astronomical Epoch Constants
JD_J2000 = 2451545.0          # Julian Date of epoch J2000.0 (2000-01-01 12:00:00 TT)
MJD_OFFSET = 2400000.5        # Offset between JD and standard MJD (JD - 2400000.5)
MJD_J2000 = 51544.5           # Modified Julian Date of J2000.0
DAYS_PER_JULIAN_CENTURY = 36525.0
SECS_PER_DAY = 86400.0


def datetime_to_jd(
    dt: datetime
) -> float:
    """
    Converts a Python UTC datetime object to Julian Date (JD).

    Parameters
    ----------
    dt : datetime
        UTC Calendar datetime.

    Returns
    -------
    float
        Julian Date (days).
    """
    year = dt.year
    month = dt.month
    day = dt.day

    # Fractional day from hours, minutes, seconds, microseconds
    frac_day = (dt.hour + dt.minute / 60.0 + (dt.second + dt.microsecond * 1e-6) / 3600.0) / 24.0

    if month <= 2:
        year -= 1
        month += 12

    # Fliegel-Van Flandern algorithm
    a = year // 100
    b = 2 - a + (a // 4)

    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5 + frac_day
    return float(jd)


def jd_to_datetime(
    jd: float
) -> datetime:
    """
    Converts Julian Date (JD) back to a Python UTC datetime object.

    Parameters
    ----------
    jd : float
        Julian Date (days).

    Returns
    -------
    datetime
        UTC Calendar datetime.
    """
    jd_adjusted = jd + 0.5
    z = int(jd_adjusted)
    f = jd_adjusted - z

    if z < 2299161:
        a = z
    else:
        alpha = int((z - 1867216.25) / 36524.25)
        a = z + 1 + alpha - (alpha // 4)

    b = a + 1524
    c = int((b - 122.1) / 365.25)
    d = int(365.25 * c)
    e = int((b - d) / 30.6001)

    day_float = b - d - int(30.6001 * e) + f
    day = int(day_float)
    frac_day = day_float - day

    if e < 14:
        month = e - 1
    else:
        month = e - 13

    if month > 2:
        year = c - 4716
    else:
        year = c - 4715

    # Extract time components
    total_seconds = frac_day * SECS_PER_DAY
    hours = int(total_seconds // 3600)
    total_seconds %= 3600
    minutes = int(total_seconds // 60)
    seconds_float = total_seconds % 60
    seconds = int(seconds_float)
    microseconds = int(round((seconds_float - seconds) * 1e6))

    if microseconds >= 1_000_000:
        microseconds -= 1_000_000
        seconds += 1

    return datetime(year, month, day, hours, minutes, seconds, microseconds)


def date_to_mjd2000(
    date_val: Union[datetime, str, float]
) -> float:
    """
    Converts input date representation to days elapsed since J2000.0 (MJD2000).

    Parameters
    ----------
    date_val : datetime, str, or float
        - If datetime: interpreted as UTC datetime.
        - If str: parsed as ISO 8601 string (e.g., '2026-08-28 12:00:00').
        - If float: treated as already Julian Date (JD).

    Returns
    -------
    float
        Days since J2000.0 epoch (2000-01-01 12:00:00 TT).
    """
    if isinstance(date_val, str):
        dt = datetime.fromisoformat(date_val)
        jd = datetime_to_jd(dt)
    elif isinstance(date_val, datetime):
        jd = datetime_to_jd(date_val)
    elif isinstance(date_val, (int, float)):
        jd = float(date_val)
    else:
        raise TypeError(f"Unsupported date format: {type(date_val)}")

    return jd - JD_J2000


def mjd2000_to_datetime(
    mjd2000: float
) -> datetime:
    """Converts days since J2000.0 (MJD2000) to Python UTC datetime."""
    jd = mjd2000 + JD_J2000
    return jd_to_datetime(jd)


def mjd2000_to_julian_centuries(
    mjd2000: float
) -> float:
    """Converts days since J2000.0 to Julian centuries (T)."""
    return mjd2000 / DAYS_PER_JULIAN_CENTURY