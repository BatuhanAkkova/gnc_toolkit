import numpy as np
import pytest
from src.kalman_filters.ekf import EKF

def test_ekf_nonlinear_tracking():
    """Test EKF with a non-linear measurement model (measuring squared state)."""
    dt = 0.1
    ekf = EKF(dim_x=1, dim_z=1)
    
    # Process Model: x_k+1 = x_k + vel * dt
    def f_func(x, dt, u, **kwargs): return x + 1.0 * dt
    def f_jac(x, dt, u, **kwargs): return np.array([[1.0]])
    
    # Measurement Model: z = x^2
    def h_func(x, **kwargs): return x**2
    def h_jac(x, **kwargs): return np.array([[2*x[0]]])
    
    ekf.P *= 1.0
    ekf.R *= 0.01
    ekf.Q *= 0.001
    ekf.x = np.array([1.0])
    
    true_x = 1.0
    
    for _ in range(20):
        true_x += 1.0 * dt
        z = true_x**2 + np.random.normal(0, 0.1)
        
        ekf.predict(f_func, f_jac, dt)
        ekf.update(np.array([z]), h_func, h_jac)
        
    error = np.abs(true_x - ekf.x[0])
    assert error < 0.5
