import pytest
import numpy as np
from src.environment.mag_field import tilted_dipole_field, igrf_field
from datetime import datetime

def test_tilted_dipole_field():
    # Test at some random position
    r_ecef = np.array([7000000.0, 0, 0])
    B = tilted_dipole_field(r_ecef)
    
    # Check shape and non-zero
    assert B.shape == (3,)
    assert np.linalg.norm(B) > 0.0
    
    # Check magnitude roughly (at 7000km, should be in microTesla range)
    # B approx: 3.12e-5 * (6371/7000)^3 = 2e-5 T
    assert np.linalg.norm(B) < 1e-4
    assert np.linalg.norm(B) > 1e-6

def test_igrf_field_import():    
    try:
        import ppigrf
    except (ImportError, FileNotFoundError, OSError):
        pytest.skip("ppigrf not installed or broken")
        
    lat = 0.0
    lon = 0.0
    alt = 600.0 # km
    date = datetime(2024, 1, 1)
    
    res = igrf_field(lon, lat, alt, date)
    assert len(res) == 3