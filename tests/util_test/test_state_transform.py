import pytest
import numpy as np
from src.utils.state_to_elements import eci2kepler, kepler2eci

def test_state_transform_roundtrip():
    # Define a state in ECI (Meters, m/s)
    # 7000 km orbit, circular-ish
    r = np.array([7000000.0, 0, 0])
    mu = 398600.4415e9
    v_mag = np.sqrt(mu / np.linalg.norm(r))
    v = np.array([0, v_mag, 0])
    
    a, ecc, incl, raan, argp, nu, M, E, p, arglat, truelon, lonper = eci2kepler(r, v)
    
    assert np.isclose(a, 7000000.0, rtol=1e-6)
    assert np.isclose(ecc, 0, rtol=1e-6)
    assert np.isclose(incl, 0, rtol=1e-6)
    assert np.isclose(raan, 0, rtol=1e-6)
    assert np.isclose(argp, 0, rtol=1e-6)
    assert np.isclose(nu, 0, rtol=1e-6)
    assert np.isclose(M, 0, rtol=1e-6)
    assert np.isclose(E, 0, rtol=1e-6)
    assert np.isclose(p, 7000000.0, rtol=1e-6)
    assert np.isclose(arglat, 0, rtol=1e-6)
    assert np.isclose(truelon, 0, rtol=1e-6)
    assert np.isclose(lonper, 0, rtol=1e-6)
    
    # Round trip
    r_back, v_back = kepler2eci(a, ecc, incl, raan, argp, nu)
    
    np.testing.assert_allclose(r_back, r, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(v_back, v, rtol=1e-6, atol=1e-6)
