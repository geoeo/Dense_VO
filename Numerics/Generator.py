import numpy as np
import Numerics.Utils as Utils
from math import pi, cos, sin


def generate_3d_plane(a,b,d,pointCount,sigma):
    c = -1 # fixed
    plane_normal = np.array([a,b,c]).astype(np.float32)
    plane_normal /= np.linalg.norm(plane_normal)

    noise_x = np.random.normal(0, sigma, pointCount)
    noise_y = np.random.normal(0, sigma, pointCount)
    noise_z = np.random.normal(0, sigma, pointCount)

    xs = np.random.uniform(-10,10,pointCount)
    ys = np.random.uniform(-10,10,pointCount)
    # divided by c = -1 implicitly
    zs = plane_normal[0]*xs + plane_normal[1]*ys + np.repeat(d,pointCount)

    plane_normal_x_pertrubed = list(map(lambda x: x*plane_normal[0],noise_x))
    plane_normal_y_pertrubed = list(map(lambda x: x*plane_normal[1],noise_y))
    plane_normal_z_pertrubed = list(map(lambda x: x*plane_normal[2],noise_z))

    return (xs + plane_normal_x_pertrubed,ys + plane_normal_y_pertrubed,zs + plane_normal_z_pertrubed)


def generate_random_se3():

    t = np.reshape(np.random.uniform(-10,10,3),(3,1))

    yaw = np.random.normal(pi / 2, 16) #around z
    pitch = np.random.normal(0, 16)# around y
    roll = np.random.normal(pi / 2, 16) # around x

    R_z = np.array([[cos(yaw), -sin(yaw), 0],
           [sin(yaw), cos(yaw), 0],
           [0, 0, 1]])

    R_y = np.array([[cos(pitch), 0, sin(pitch)],
           [0, 1, 0],
           [-sin(pitch), 0, cos(pitch)]])

    R_x = np.array([[1, 0, 0],
           [0, cos(roll), -sin(roll)],
           [0, sin(roll), cos(roll)]])

    R = np.matmul(np.matmul(R_z,R_y),R_x)

    SE3 = np.append(np.append(R,t,axis=1),Utils.homogenous_for_SE3(),axis=0)

    return SE3