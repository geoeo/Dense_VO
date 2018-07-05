import numpy as np
import Numerics.Utils as Utils
from Numerics.Utils import matrix_data_type

def generator_x():
    return np.array([[0,0,0,1],
                     [0,0,0,0],
                     [0,0,0,0],
                     [0,0,0,0]], dtype=matrix_data_type)


def generator_y():
    return np.array([[0,0,0,0],
                     [0,0,0,1],
                     [0,0,0,0],
                     [0,0,0,0]], dtype=matrix_data_type)


def generator_z():
    return np.array([[0,0,0,0],
                     [0,0,0,0],
                     [0,0,0,1],
                     [0,0,0,0]], dtype=matrix_data_type)


# Rotation Around X
def generator_roll():
    return np.array([[0,0,0,0],
                     [0,0,-1,0],
                     [0,1,0,0],
                     [0,0,0,0]], dtype=matrix_data_type)


# Rotation Around Y
def generator_pitch():
    return np.array([[0,0,1,0],
                     [0,0,0,0],
                     [-1,0,0,0],
                     [0,0,0,0]], dtype=matrix_data_type)


# Rotation Around Z
def generator_yaw():
    return np.array([[0,-1,0,0],
                     [1,0,0,0],
                     [0,0,0,0],
                     [0,0,0,0]], dtype=matrix_data_type)

def adjoint_se3(R,t):
    t_x = Utils.skew_symmetric(t[0],t[1],t[2])
    top = np.append(R,np.matmul(t_x,R),axis=1)
    bottom = np.append(np.identity(3),R,axis=1)
    return np.append(top,bottom,axis=0)