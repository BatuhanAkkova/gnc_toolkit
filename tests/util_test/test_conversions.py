import pytest
import numpy as np
from src.utils.state_conversion import (
    quat_to_dcm, quat_to_euler, dcm_to_quat, dcm_to_euler,
    euler_to_quat, euler_to_dcm, rot_x, rot_y, rot_z
)

def test_rotation_matrices():
    """Test rotation matrices rot_x, rot_y, rot_z."""
    angle = np.pi / 2
    
    Rx = rot_x(angle)
    expected_Rx = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    np.testing.assert_allclose(Rx, expected_Rx, atol=1e-10)
    
    Ry = rot_y(angle)
    expected_Ry = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    np.testing.assert_allclose(Ry, expected_Ry, atol=1e-10)
    
    Rz = rot_z(angle)
    expected_Rz = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    np.testing.assert_allclose(Rz, expected_Rz, atol=1e-10)

def test_quat_dcm_roundtrip():
    """Test quaternion to direction cosine matrix and back."""
    # Identity
    q_id = np.array([0, 0, 0, 1])
    dcm_id = quat_to_dcm(q_id)
    np.testing.assert_allclose(dcm_id, np.eye(3), atol=1e-10)
    q_out = dcm_to_quat(dcm_id)
    # Quaternion double cover check
    if np.dot(q_id, q_out) < 0:
        q_out = -q_out
    np.testing.assert_allclose(q_out, q_id, atol=1e-10)

    # Random rotation
    angle = np.pi / 3
    axis = np.array([1, 1, 1])
    axis = axis / np.linalg.norm(axis)
    q_rot = np.array([
        axis[0] * np.sin(angle/2),
        axis[1] * np.sin(angle/2),
        axis[2] * np.sin(angle/2),
        np.cos(angle/2)
    ])
    
    dcm = quat_to_dcm(q_rot)
    q_res = dcm_to_quat(dcm)
    
    if np.dot(q_rot, q_res) < 0:
        q_res = -q_res
    np.testing.assert_allclose(q_res, q_rot, atol=1e-10)

def test_euler_dcm_roundtrip():
    """Test Euler angles to direction cosine matrix and back."""
    # Sequence 3-2-1 (scipy style z-y-x)
    angles = np.array([0.1, 0.2, 0.3])
    seq = "321"
    
    dcm = euler_to_dcm(angles, seq)
    angles_out = dcm_to_euler(dcm, seq)
    
    np.testing.assert_allclose(angles_out, angles, atol=1e-10)

def test_euler_quat_roundtrip():
    """Test Euler angles to quaternion and back."""
    angles = np.array([0.5, -0.2, 0.1])
    seq = "123"
    
    q = euler_to_quat(angles, seq)
    dcm_from_q = quat_to_dcm(q)
    dcm_from_e = euler_to_dcm(angles, seq)
    
    np.testing.assert_allclose(dcm_from_q, dcm_from_e, atol=1e-10)
    
    angles_back = quat_to_euler(q, seq)
    np.testing.assert_allclose(angles_back, angles, atol=1e-10)

def test_invalid_sequence():
    """Test invalid Euler angle sequences."""
    with pytest.raises(ValueError):
        euler_to_dcm([0,0,0], "12")
    with pytest.raises(ValueError):
        euler_to_quat([0,0,0], "1234")
