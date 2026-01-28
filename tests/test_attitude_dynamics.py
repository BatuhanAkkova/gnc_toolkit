import numpy as np
import pytest
from gnc_toolkit.attitude_dynamics.rigid_body import euler_equations

def test_euler_equations_zero_values():
    """Test with zero torque and angular velocity."""
    J = np.eye(3)
    omega = np.zeros(3)
    torque = np.zeros(3)
    
    omega_dot = euler_equations(J, omega, torque)
    
    np.testing.assert_array_equal(omega_dot, np.zeros(3))

def test_euler_equations_principal_axis_rotation_stable():
    """
    Test rotation about a principal axis (stable).
    If omega is aligned with a principal axis and no torque, omega_dot should be zero.
    """
    J = np.diag([10, 20, 30])
    omega = np.array([1.0, 0.0, 0.0]) # Rotation about x-axis (principal axis)
    torque = np.zeros(3)
    
    omega_dot = euler_equations(J, omega, torque)
    
    np.testing.assert_allclose(omega_dot, np.zeros(3), atol=1e-12)

def test_euler_equations_external_torque():
    """Test with a simple external torque acting on a symmetric body."""
    J = np.eye(3) * 10
    omega = np.zeros(3)
    torque = np.array([10.0, 0.0, 0.0])
    
    # Expected omega_dot = J_inv * torque = [1, 0, 0]
    expected_omega_dot = np.array([1.0, 0.0, 0.0])
    
    omega_dot = euler_equations(J, omega, torque)
    
    np.testing.assert_allclose(omega_dot, expected_omega_dot, atol=1e-12)

def test_euler_equations_general_case():
    """
    Test a general case with analytical verification.
    J = diag([1, 2, 3])
    omega = [1, 1, 1]
    torque = [0, 0, 0]
    
    Equations:
    J1*w1_dot + (J3-J2)*w2*w3 = T1 -> 1*w1_dot + (3-2)*1*1 = 0 -> w1_dot = -1
    J2*w2_dot + (J1-J3)*w3*w1 = T2 -> 2*w2_dot + (1-3)*1*1 = 0 -> 2*w2_dot - 2 = 0 -> w2_dot = 1
    J3*w3_dot + (J2-J1)*w1*w2 = T3 -> 3*w3_dot + (2-1)*1*1 = 0 -> 3*w3_dot + 1 = 0 -> w3_dot = -1/3
    """
    J = np.diag([1.0, 2.0, 3.0])
    omega = np.array([1.0, 1.0, 1.0])
    torque = np.zeros(3)
    
    expected_omega_dot = np.array([-1.0, 1.0, -1.0/3.0])
    
    omega_dot = euler_equations(J, omega, torque)
    
    np.testing.assert_allclose(omega_dot, expected_omega_dot, atol=1e-12)

def test_euler_equations_invalid_shapes():
    """Test that ValueError is raised for incorrect shapes."""
    J = np.eye(3)
    omega = np.zeros(3)
    torque = np.zeros(3)
    
    with pytest.raises(ValueError):
        euler_equations(np.eye(2), omega, torque) # Wrong J shape
        
    with pytest.raises(ValueError):
        euler_equations(J, np.zeros(2), torque) # Wrong omega shape
        
    with pytest.raises(ValueError):
        euler_equations(J, omega, np.zeros(4)) # Wrong torque shape
