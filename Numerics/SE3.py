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
#
# """A Data Object which encodes the extrinsic state of the camera.
#
# Attributes:
#   se3: the matrix which encodes rotation and translation
#   se3_inv: the inverse matrix
# """
#
#
# class SE3:
#     def __init__(self, se3 : np.ndarray, invert_matrices=False):
#         self.se3 = se3
#         self.se3_inv = SE3.inverse(se3)
#
#         if(invert_matrices):
#             self.se3 = self.se3_inv
#             self.se3_inv = se3
#
#
#     @staticmethod
#     def extract_rotation(se3):
#         return se3[0:3,0:3]
#
#     @staticmethod
#     def extract_translation(se3):
#         return se3[0:3,3:4]
#
#     @staticmethod
#     def extract_r_t(se3):
#         # returns 3x4 sub matrix
#         return se3[0:3,0:4]
#
#     @staticmethod
#     def inverse(se3):
#         rotation = SE3.extract_rotation(se3)
#         rotation_transpose = np.transpose(rotation)
#
#         translation = SE3.extract_translation(se3)
#         translation_inverse = np.multiply(-1,np.matmul(rotation_transpose,translation))
#
#         m = np.concatenate((rotation_transpose, translation_inverse), axis=1)
#         se3_inv = SE3.append_homogeneous_along_y(m)
#
#         return se3_inv
#
#     @staticmethod
#     def append_homogeneous_along_y(m):
#         return np.concatenate((m, np.array([[0, 0, 0, 1]])), axis=0)
