import numpy as np


# joint constraints for UR10e
q_max = np.array([np.pi]*6)
q_min = np.array([-np.pi]*6)

q_dot_max = np.array([np.pi/3, np.pi/3, np.pi/3, np.pi/2, np.pi/2, np.pi/2])
q_torque_max = np.array([90., 90., 90., 90., 12., 12., 12.])

def skew(x):
    """
    Returns the skew-symmetric matrix of x
    param x: 3x1 vector
    """
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

def get_quaternion_error(curr_quat, des_quat):
    """
    Calculates the difference between the current quaternion and the desired quaternion.
    See Siciliano textbook page 140 Eq 3.91

    param curr_quat: current quaternion
    param des_quat: desired quaternion
    return: difference between current quaternion and desired quaternion
    """
    return curr_quat[0] * des_quat[1:] - des_quat[0] * curr_quat[1:] - skew(des_quat[1:]) @ curr_quat[1:]

def rotation_distance(p: np.array, q: np.array):
    """
    Calculates the rotation angular between two quaternions
    param p: quaternion
    param q: quaternion
    theta: rotation angle between p and q (rad)
    """
    assert p.shape == q.shape, "p and q should be quaternion"
    theta = 2 * np.arccos(abs(p @ q))
    return theta


def rot_to_quat(theta, axis):
    """
    Converts rotation angle along an axis to quaternion
    param theta: rotation angle (rad)
    param axis: rotation axis
    return: quaternion
    """
    quant = np.zeros(4)
    quant[0] = np.sin(theta / 2.)
    quant[1:] = np.cos(theta / 2.) * axis
    return quant