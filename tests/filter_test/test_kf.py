import numpy as np
import pytest
from src.kalman_filters.kf import KF

def test_kf_initialization():
    dim_x = 2
    dim_z = 1
    kf = KF(dim_x, dim_z)
    assert kf.x.shape == (dim_x,)
    assert kf.P.shape == (dim_x, dim_x)
    assert kf.F.shape == (dim_x, dim_x)
    assert kf.H.shape == (dim_z, dim_x)
    assert kf.R.shape == (dim_z, dim_z)
    assert kf.Q.shape == (dim_x, dim_x)

def test_kf_constant_velocity():
    """Test KF with a simple 1D constant velocity model."""
    dt = 0.1
    kf = KF(dim_x=2, dim_z=1)
    
    # State: [position, velocity]
    kf.F = np.array([[1, dt], [0, 1]])
    kf.H = np.array([[1, 0]])
    kf.P *= 10
    kf.R *= 0.1
    kf.Q = np.array([[0.001, 0], [0, 0.001]])

    true_x = np.array([0., 1.0]) # Pos=0, Vel=1 m/s
    
    # Run filter for 20 steps
    for _ in range(20):
        # Simulate Truth
        true_x = np.dot(kf.F, true_x)
        
        # Simulate Measurement
        z = np.dot(kf.H, true_x) + np.random.normal(0, np.sqrt(kf.R[0,0]))
        
        # Filter Cycle
        kf.predict()
        kf.update(z)
        
    error_pos = np.abs(true_x[0] - kf.x[0])
    error_vel = np.abs(true_x[1] - kf.x[1])
    
    # Relaxed thresholds for stochastic test
    assert error_pos < 1.0
    assert error_vel < 0.5
