import numpy as np


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
