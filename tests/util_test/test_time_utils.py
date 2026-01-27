import pytest
import numpy as np
from src.utils.time_utils import (
    calc_jd, jd_to_datetime, day_to_mdtime, calc_gmst,
    calc_last, calc_lst, calc_doy, is_leap_year, convert_time
)

def test_calc_jd():
    # J2000 Epoch: 2000 Jan 1 12:00:00 TT -> JD 2451545.0
    jd, jdfrac = calc_jd(2000, 1, 1, 12, 0, 0)
    assert jd + jdfrac == 2451545.0
    
def test_jd_datetime_roundtrip():
    y, m, d, h, mn, s = 2024, 1, 18, 10, 30, 45.0
    jd, jdfrac = calc_jd(y, m, d, h, mn, s)
    y2, m2, d2, h2, mn2, s2 = jd_to_datetime(jd, jdfrac)
    
    assert y == y2
    assert m == m2
    assert d == d2
    assert h == h2
    assert mn == mn2
    assert np.isclose(s, s2, atol=1e-6)

def test_calc_gmst():
    # J2000 GMST verify
    jd = 2451545.0
    gmst_rad = calc_gmst(jd)
    gmst_deg = np.degrees(gmst_rad)
    
    # Expected approx 280.4606 deg
    assert np.isclose(gmst_deg, 280.4606, atol=1e-2)
    
    # Range check
    assert 0 <= gmst_rad < 2*np.pi

def test_is_leap_year():
    assert is_leap_year(2000)
    assert is_leap_year(2024)
    assert not is_leap_year(2100)
    assert not is_leap_year(2023)

def test_calc_doy():
    assert calc_doy(2024, 1, 1) == 1
    assert calc_doy(2024, 2, 29) == 60
    assert calc_doy(2023, 2, 28) == 59
    assert calc_doy(2023, 3, 1) == 60

def test_calc_lst():
    # LST = GMST + lon
    gmst = 1.0
    lon = 0.5
    lst = calc_lst(gmst, lon)
    assert np.isclose(lst, 1.5)

def test_convert_time():
    # Input: 2024 Jan 1, 12:00:00, timezone 0, output?, dut1=0, dat=37
    # Should run without error
    res = convert_time(2024, 1, 1, 12, 0, 0, 0, 0, 0, 37)
    assert isinstance(res, tuple)
    assert len(res) == 15
