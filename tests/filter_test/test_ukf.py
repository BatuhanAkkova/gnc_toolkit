import numpy as np
import pytest
from src.kalman_filters.ukf import UKF

def test_ukf_nonlinear_tracking():
    """Test UKF with non-linear model (vector state version)."""
    dt = 0.1
    ukf = UKF(dim_x=1, dim_z=1)
    
    # Process: x_k+1 = x_k + vel * dt
    def f_func(x, dt): return x + 1.0 * dt
    
    # Measurement: z = x^2
    def h_func(x): return x**2
    
    ukf.P *= 1.0
    ukf.R *= 0.01
    ukf.Q *= 0.001
    ukf.x = np.array([1.0])
    
    true_x = 1.0
    
    for _ in range(20):
        true_x += 1.0 * dt
        z = true_x**2 + np.random.normal(0, 0.1)
        
        ukf.predict(dt, f_func)
        ukf.update(np.array([z]), h_func)
        
    error = np.abs(true_x - ukf.x[0])
    assert error < 0.5
