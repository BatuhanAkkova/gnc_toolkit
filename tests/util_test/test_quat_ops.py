import pytest
import numpy as np
from src.utils.quat_utils import (
    quat_normalize, quat_conj, quat_norm, quat_inv,
    quat_mult, quat_rot, quat_to_rmat, axis_angle_to_quat
)

def test_quat_norm_normalize():
    q = np.array([3, 0, 4, 0]) # Norm is 5
    assert np.isclose(quat_norm(q), 5.0)
    
    q_norm = quat_normalize(q)
    assert np.isclose(np.linalg.norm(q_norm), 1.0)
    assert np.allclose(q_norm, np.array([0.6, 0, 0.8, 0]))

    with pytest.raises(ValueError):
        quat_normalize(np.zeros(4))

def test_quat_ops_basic():
    q = np.array([0, 0, 0, 1]) # Identity
    
    # Conjugate
    q_conj = quat_conj(q)
    np.testing.assert_array_equal(q_conj, np.array([0, 0, 0, 1]))
    
    # Inverse
    q_inv = quat_inv(q)
    np.testing.assert_array_equal(q_inv, np.array([0, 0, 0, 1]))
    
    q2 = np.array([1, 0, 0, 0])
    q_mult = quat_mult(q2, q2)
    np.testing.assert_array_equal(q_mult, np.array([0, 0, 0, -1]))

def test_quat_rot():
    # Rotate vector [1, 0, 0] by 90 deg around z-axis
    # q = [0, 0, sin(45), cos(45)] = [0, 0, 0.7071, 0.7071]
    angle = np.pi/2
    q = np.array([0, 0, np.sin(angle/2), np.cos(angle/2)])
    
    v = np.array([1, 0, 0])
    v_rot = quat_rot(q, v)
    
    # Expect [0, 1, 0]
    np.testing.assert_allclose(v_rot, np.array([0, 1, 0]), atol=1e-10)

def test_axis_angle_to_quat():
    axis = np.array([0, 0, 1], dtype=float)
    angle = np.pi/2
    axis_angle = axis * angle
    
    q = axis_angle_to_quat(axis_angle)
    expected = np.array([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])
    
    np.testing.assert_allclose(q, expected, atol=1e-10)
    
    # Zero rotation
    q_zero = axis_angle_to_quat(np.zeros(3))
    np.testing.assert_array_equal(q_zero, np.array([0, 0, 0, 1]))

def test_quat_to_rmat():
    # Identity
    q = np.array([0, 0, 0, 1])
    rmat = quat_to_rmat(q)
    np.testing.assert_array_equal(rmat, np.eye(3))
    
    # 90 deg Z rotation
    # q = [0, 0, 1/sqrt(2), 1/sqrt(2)]
    val = 1.0/np.sqrt(2)
    q_z = np.array([0, 0, val, val])
    rmat_z = quat_to_rmat(q_z)
    
    expected = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    np.testing.assert_allclose(rmat_z, expected, atol=1e-10)
