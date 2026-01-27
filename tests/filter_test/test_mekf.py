import numpy as np
import pytest
from src.kalman_filters.mekf import MEKF
from src.utils.quat_utils import quat_mult, quat_normalize, axis_angle_to_quat, quat_rot, quat_conj

def test_mekf_attitude_tracking():
    """Test MEKF for attitude estimation."""
    dt = 0.1
    omega_body = np.array([0.1, 0.05, 0.2]) # Constant body rate
    z_ref_inertial = np.array([1.0, 0.0, 0.0])
    
    q0 = np.array([0, 0, 0, 1.0])
    mekf = MEKF(q_init=q0)
    
    mekf.Q = np.eye(6) * 0.0001
    mekf.R = np.eye(3) * 0.01
    
    q_true = q0.copy()
    
    for _ in range(50):
        # Truth Update
        dq_true = axis_angle_to_quat(omega_body * dt)
        q_true = quat_normalize(quat_mult(q_true, dq_true))
        
        # Measurement simulation
        z_body_true = quat_rot(quat_conj(q_true), z_ref_inertial)
        z_body_meas = z_body_true + np.random.normal(0, 0.01, 3)
        z_body_meas /= np.linalg.norm(z_body_meas)
        
        # Filter Step
        mekf.predict(omega_body, dt)
        mekf.update(z_body_meas, z_ref_inertial)
        
    error = 1.0 - np.abs(np.dot(q_true, mekf.q))
    assert error < 1e-3
