import unittest
import numpy as np
from gnc_toolkit.guidance.maneuvers import (
    hohmann_transfer,
    bi_elliptic_transfer,
    phasing_maneuver,
    plane_change,
    combined_plane_change
)
from gnc_toolkit.guidance.rendezvous import solve_lambert, cw_equations, cw_targeting

class TestManeuvers(unittest.TestCase):
    def test_hohmann_transfer(self):
        # LEO (r=7000) to GEO (r=42164)
        r1 = 7000
        r2 = 42164
        mu = 398600.4418
        
        dv1, dv2, t_trans = hohmann_transfer(r1, r2, mu)
        
        # Expected values (approx)
        # v1_circ = 7.546
        # v_trans_p = 10.15
        # dv1 = 2.4 km/s
        self.assertAlmostEqual(dv1, 2.4, delta=0.3) # Rough check
        self.assertGreater(t_trans, 0)
        
    def test_bi_elliptic_transfer(self):
        r1 = 7000
        r2 = 105000 # r2/r1 = 15
        rb = 200000
        mu = 398600.4418
        
        dv1, dv2, dv3, t = bi_elliptic_transfer(r1, r2, rb, mu)
        total_dv = dv1 + dv2 + dv3
        
        # Check against Hohmann
        dh1, dh2, th = hohmann_transfer(r1, r2, mu)
        total_hohmann = dh1 + dh2
        
        # Bi-elliptic should be slightly efficient
        self.assertLess(total_dv, total_hohmann)
        
    def test_plane_change(self):
        v = 7.5
        di = np.radians(10)
        dv = plane_change(v, di)
        expected = 2 * v * np.sin(np.radians(5))
        self.assertAlmostEqual(dv, expected)
        
    def test_phasing(self):
        a = 7000
        # Wait 100s
        dv, t_phasing = phasing_maneuver(a, 100)
        
        self.assertGreater(dv, 0)
        self.assertAlmostEqual(t_phasing - 2*np.pi*np.sqrt(a**3/398600.4418), 100, delta=1.0)

class TestRendezvous(unittest.TestCase):
    def test_solve_lambert(self):
        mu = 398600.4418
        # Earth to Mars simplified (Hohmann-like transfer)
        r1 = np.array([7000.0, 0.0, 0.0])
        r2 = np.array([0.0, 10000.0, 0.0]) # 90 deg separation, diff radius
        
        # known example from Vallado
        # r1 = [15945, 0, 0], r2 = [12214, 10249, 0], dt = 76 min
        
        dt = 2000.0 # seconds
        
        v1, v2 = solve_lambert(r1, r2, dt, mu=mu, tm=1)
        
        # Check velocities are finite
        self.assertFalse(np.isnan(v1).any())
        self.assertFalse(np.isnan(v2).any())
        
    def test_cw_equations(self):
        # Station keeping (zero relative velocity, zero position)
        r0 = np.array([0., 0., 0.])
        v0 = np.array([0., 0., 0.])
        n = 0.001
        t = 100.0
        
        rt, vt = cw_equations(r0, v0, n, t)
        np.testing.assert_array_almost_equal(rt, r0)
        np.testing.assert_array_almost_equal(vt, v0)
        
        # Drift
        r0 = np.array([0., 0., 0.])
        v0 = np.array([0., -0.1, 0.]) # Drift backwards
        rt, vt = cw_equations(r0, v0, n, t)
        
        self.assertNotEqual(rt[1], 0)
        
    def test_cw_targeting(self):
        n = 0.0011 # LEO approx
        t_transfer = 1000.0
        
        r0 = np.array([10.0, 0.0, 0.0]) # 10 km below/above
        r_target = np.array([0.0, 0.0, 0.0]) # To origin
        
        v0_req = cw_targeting(r0, r_target, t_transfer, n)
        
        # Propagate to check
        rt, vt = cw_equations(r0, v0_req, n, t_transfer)
        
        np.testing.assert_allclose(rt, r_target, atol=1e-5)

if __name__ == '__main__':
    unittest.main()
