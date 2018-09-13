import numpy as np
import math
from Numerics.Utils import matrix_data_type


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

def makeS03(roll_rad,pitch_rad,yaw_rad):
    R_x = np.identity(3)
    R_y = np.identity(3)
    R_z = np.identity(3)

    R_x[1,1] = math.cos(roll_rad)
    R_x[1,2] = -math.sin(roll_rad)
    R_x[2,1] = math.sin(roll_rad)
    R_x[2,2] = math.cos(roll_rad)

    R_y[0,0] = math.cos(pitch_rad)
    R_y[0,2] = math.sin(pitch_rad)
    R_y[2,0] = -math.sin(pitch_rad)
    R_y[2,2] = math.cos(pitch_rad)

    R_z[0,0] = math.cos(yaw_rad)
    R_z[0,1] = -math.sin(yaw_rad)
    R_z[1,0] = math.sin(yaw_rad)
    R_z[1,1] = math.cos(yaw_rad)

    S03 = np.matmul(R_z,np.matmul(R_y,R_x))

    return S03

#https: // en.wikipedia.org / wiki / Conversion_between_quaternions_and_Euler_angles
def Quaternion_toEulerianRadians(x_raw, y_raw, z_raw, w_raw):

    #n =  math.sqrt(x_raw*x_raw+y_raw*y_raw+z_raw*z_raw+w_raw*w_raw)
    x = x_raw
    y = y_raw
    z = z_raw
    w = w_raw

    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = 1 if t2 > 1 else t2
    t2 = -1 if t2 < -1 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = math.atan2(t3, t4)

    return X, Y, Z

'''Returns 4x4 SE3 Matrix'''
'''Source will become new origin of coordiante system'''
def pose_pose_composition_inverse(SE3_source, SE3_target):

    R_source = SE3_source[0:3,0:3]
    R_target = SE3_target[0:3,0:3]

    x_source = SE3_source[0,3]
    y_source = SE3_source[1,3]
    z_source = SE3_source[2,3]

    x_target = SE3_target[0,3]
    y_target = SE3_target[1,3]
    z_target = SE3_target[2,3]

    R_source_inv = np.transpose(R_source)
    #R_target_inv = np.transpose(R_target)

    translation_source = np.array([[x_source],[y_source],[z_source]],matrix_data_type)
    translation_target = np.array([[x_target],[y_target],[z_target]],matrix_data_type)

    translation_source_target = translation_target - translation_source
    translation_source_target_prime = np.matmul(R_source_inv,translation_source_target)

    R_origin_target = np.matmul(R_source_inv,R_target)

    se3_source_target = np.identity(4,matrix_data_type)
    se3_source_target[0:3,3:4] = translation_source_target_prime
    se3_source_target[0:3,0:3] = R_origin_target[0:3,0:3]

    return se3_source_target

def quaternion_to_s03(qx, qy, qz, qw):
    mag = math.sqrt(qx*qx+qy*qy+qz*qz+qw*qw)

    qx /= mag
    qy /= mag
    qz /= mag
    qw /= mag

    qw_sq = qw*qw
    qx_sq = qx*qx
    qy_sq = qy*qy
    qz_sq = qz*qz

    so3 = np.identity(3,matrix_data_type)

    so3[0,0] = qw_sq + qx_sq - qy_sq - qz_sq
    so3[0,1] = 2*(qx*qy - qw*qz)
    so3[0,2] = 2*(qx*qz + qw*qy)

    so3[1,0] = 2*(qx*qy + qw*qz)
    so3[1,1] = qw_sq - qx_sq + qy_sq - qz_sq
    so3[1,2] = 2*(qz*qy - qw*qx)

    so3[2,0] = 2*(qx*qz - qw*qy)
    so3[2,1] = 2*(qz*qy + qw*qx)
    so3[2,2] = qw_sq - qx_sq - qy_sq + qz_sq

    return so3



