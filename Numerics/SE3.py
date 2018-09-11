import numpy as np
import math


def extract_rotation(se3 : np.ndarray):
    return se3[0:3, 0:3]


def extract_translation(se3 : np.ndarray):
    return se3[0:3, 3:4]


def extract_r_t(se3: np.ndarray):
    # returns 3x4 sub matrix
    return se3[0:3, 0:4]


def invert(se3: np.ndarray):
    rotation = extract_rotation(se3)
    rotation_transpose = np.transpose(rotation)

    translation = extract_translation(se3)
    translation_inverse = np.multiply(-1, np.matmul(rotation_transpose, translation))

    m = np.concatenate((rotation_transpose, translation_inverse), axis=1)
    se3_inv = append_homogeneous_along_y(m)

    return se3_inv


def append_homogeneous_along_y(m):
    return np.concatenate((m, np.array([[0, 0, 0, 1]])), axis=0)


#https://www.learnopencv.com/rotation-matrix-to-euler-angles/
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    #assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])
