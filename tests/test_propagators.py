import pytest
import numpy as np
from gnc_toolkit.propagators.kepler import KeplerPropagator
from gnc_toolkit.propagators.cowell import CowellPropagator

def test_kepler_circular_period():
    """
    Test Kepler propagator with a circular orbit.
    Propagating by one period should return to the initial state.
    """
    mu = 398600.4418e9
    r_mag = 7000e3 # 7000 km
    v_circ = np.sqrt(mu / r_mag)
    
    r0 = np.array([r_mag, 0, 0])
    v0 = np.array([0, v_circ, 0])
    
    period = 2 * np.pi * np.sqrt(r_mag**3 / mu)
    
    propagator = KeplerPropagator(mu=mu)
    r_f, v_f = propagator.propagate(r0, v0, period)

    np.testing.assert_allclose(r_f, r0, atol=1e-5)
    np.testing.assert_allclose(v_f, v0, atol=1e-5)

def test_cowell_two_body_consistency():
    """
    Test that Cowell propagator (with no perturbations) matches Kepler propagator
    for a short duration.
    """
    mu = 398600.4418e9
    r0 = np.array([7000e3, 0, 0])
    v0 = np.array([0, 7.5e3, 0])
    
    dt = 100.0 # Propagate for 100 seconds
    
    # Kepler
    kepler_prop = KeplerPropagator(mu=mu)
    r_k, v_k = kepler_prop.propagate(r0, v0, dt)
    
    # Cowell (default integrator RK4, no perturbations)
    cowell_prop = CowellPropagator(mu=mu)
    r_c, v_c = cowell_prop.propagate(r0, v0, dt, dt_step=1.0) # Small step for accuracy
    
    # Tolerances might need adjustment depending on integrator order and step size
    np.testing.assert_allclose(r_c, r_k, rtol=1e-5)
    np.testing.assert_allclose(v_c, v_k, rtol=1e-5)

def test_cowell_with_perturbation():
    """
    Test Cowell propagator with a dummy perturbation.
    Check if it diverges from the unperturbed path.
    """
    mu = 398600.4418e9
    r0 = np.array([7000e3, 0, 0])
    v0 = np.array([0, 7.5e3, 0])
    dt = 100.0
    
    def constant_thrust(t, r, v):
        # Constant acceleration in velocity direction
        return 0.01 * v / np.linalg.norm(v)

    cowell_prop = CowellPropagator(mu=mu)
    
    # Unperturbed
    r_unp, _ = cowell_prop.propagate(r0, v0, dt, dt_step=1.0)
    
    # Perturbed
    r_pert, _ = cowell_prop.propagate(r0, v0, dt, perturbation_acc_fn=constant_thrust, dt_step=1.0)
    
    # Should be different
    assert not np.allclose(r_unp, r_pert)
