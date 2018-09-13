import numpy as np
from Numerics import SE3

'''list format is: tx, ty, tz, qx, qy, qz, qw'''
def generate_se3_from_groundtruth(groundtruth_list):
    tx = float(groundtruth_list[0])
    ty = float(groundtruth_list[1])
    tz = float(groundtruth_list[2])

    qx = float(groundtruth_list[3])
    qy = float(groundtruth_list[4])
    qz = float(groundtruth_list[5])
    qw = float(groundtruth_list[6])

    se3 = np.identity(4)

    roll, pitch, yaw = SE3.Quaternion_toEulerianRadians(qx,qy,qz,qw)
    SO3 = SE3.makeS03(roll, pitch, yaw) #  seems to be more precise
    #SO3 = SE3.quaternion_to_s03(qx,qy,qz,qw)

    se3[0:3,0:3] = SO3[0:3,0:3]
    se3[0,3] = tx
    se3[1,3] = ty
    se3[2,3] = tz

    return se3




