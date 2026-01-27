import pytest
import numpy as np
from src.kalman_filters.ukf import UKF_Attitude
from src.utils.quat_utils import quat_normalize, axis_angle_to_quat, quat_mult, quat_conj, quat_rot

class MockSunSensor:
    def get_measurement(self, q_true):
        s_eci = np.array([1.0, 0.0, 0.0])
        q_conj = quat_conj(q_true)
        s_body = quat_rot(q_conj, s_eci)
        return s_body, s_eci

def test_ukf_attitude_initialization():
    ukf = UKF_Attitude()
    assert ukf.x.shape == (7,) 
    assert ukf.P.shape == (6, 6) 
    
    assert np.isclose(np.linalg.norm(ukf.x[:4]), 1.0)

def test_ukf_attitude_prediction():
    ukf = UKF_Attitude()
    dt = 0.1
    
    ukf.x = np.array([0., 0., 0., 1., 0.1, 0.1, 0.1]) 
    
    def fx(x, dt, omega_meas):
        q = x[:4]
        bias = x[4:]
        
        omega = omega_meas - bias
        
        omega_norm = np.linalg.norm(omega)
        if omega_norm > 1e-10:
            axis = omega / omega_norm
            angle = omega_norm * dt
            dq = axis_angle_to_quat(axis * angle)
            q_new = quat_mult(q, dq)
            q_new = quat_normalize(q_new)
        else:
            q_new = q
            
        return np.concatenate([q_new, bias])
    
    omega_meas = np.array([0.1, 0.1, 0.1]) 
    
    ukf.predict(dt, fx, omega_meas=omega_meas)
    
    assert np.allclose(ukf.x[:4], np.array([0, 0, 0, 1]), atol=1e-5)
    
    assert np.allclose(ukf.x[4:], np.array([0.1, 0.1, 0.1]), atol=1e-5)

def test_ukf_attitude_update():
    ukf = UKF_Attitude()
    ukf.P *= 0.1
    ukf.R *= 0.001
    
    angle = np.pi/2
    q_true = np.array([0, 0, np.sin(angle/2), np.cos(angle/2)])
    
    sensor = MockSunSensor()
    z_meas, z_ref = sensor.get_measurement(q_true)
    
    def hx(x, z_ref):
        q = x[:4]
        q_conj = quat_conj(q)
        z_pred = quat_rot(q_conj, z_ref)
        return z_pred
        
    ukf.update(z_meas, hx, z_ref=z_ref)
    
    assert ukf.x.shape == (7,)
    assert np.isclose(np.linalg.norm(ukf.x[:4]), 1.0)
    pass
