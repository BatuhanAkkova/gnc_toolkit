import pytest
import numpy as np
from src.utils.frame_conversion import eci2ecef, ecef2eci, eci2lvlh_dcm, eci2llh, elements2perifocal_dcm

def test_eci_ecef_roundtrip():
    """Test ECI to ECEF and back."""
    reci = np.array([7000, 0, 0])
    veci = np.array([0, 7.5, 0])
    jdut1 = 2451545.0 # Some Julian Date
    
    recef, vecef = eci2ecef(reci, veci, jdut1)
    reci_back, veci_back = ecef2eci(recef, vecef, jdut1)
    
    np.testing.assert_allclose(reci, reci_back, atol=1e-10)
    np.testing.assert_allclose(veci, veci_back, atol=1e-10)

def test_eci2lvlh_dcm():
    """Test ECI to LVLH direction cosine matrix."""
    r = np.array([7000, 0, 0])
    v = np.array([0, 7.5, 0])
    
    # Orbit normal h = r x v = (0, 0, 7000*7.5) = +Z direction
    # y_lvlh = -h/norm = (0, 0, -1)
    # z_lvlh = -r/norm = (-1, 0, 0)
    # x_lvlh = y x z = (0, 0, -1) x (-1, 0, 0) = (0, 1, 0)
    
    expected_dcm = np.array([
        [0, 1, 0],
        [0, 0, -1],
        [-1, 0, 0]
    ])
    
    dcm = eci2lvlh_dcm(r, v)
    np.testing.assert_allclose(dcm, expected_dcm, atol=1e-10)

def test_eci2llh():
    r_test = [6678137.0, 0, 0]
    jdut1 = 2451545.0
    lat, lon, h = eci2llh(r_test, jdut1)
    np.testing.assert_allclose(h, 300000, atol=1e-10)

def test_elements2perifocal_dcm():
    """Test elements to perifocal direction cosine matrix."""
    # Case: Identity (all zero)
    dcm = elements2perifocal_dcm(0, 0, 0)
    np.testing.assert_allclose(dcm, np.eye(3), atol=1e-10)
    
    # Case: 90 deg inclination (x-rotation)
    dcm_inc = elements2perifocal_dcm(0, np.pi/2, 0)
    np.testing.assert_allclose(np.dot(dcm_inc, dcm_inc.T), np.eye(3), atol=1e-10)
    
    # Explicit check for 90 deg inclination (Rotation about X axis of perifocal? No, ECI X if node is there)
    # R = [1, 0, 0; 0, 0, -1; 0, 1, 0]
    expected_inc = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    np.testing.assert_allclose(dcm_inc, expected_inc, atol=1e-10)

    # Case: RAAN 90 (z-rotation)
    # R = Rz(90) = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    dcm_raan = elements2perifocal_dcm(np.pi/2, 0, 0)
    expected_raan = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    np.testing.assert_allclose(dcm_raan, expected_raan, atol=1e-10)
