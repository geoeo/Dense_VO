import numpy as np
import math
import Numerics.Utils as Utils
from Numerics.Utils import matrix_data_type

I_3 = np.identity(3, dtype=matrix_data_type)
zero_trans = np.zeros((3,1),dtype=matrix_data_type)

def generator_x():
    return np.array([[0,0,0,1],
                     [0,0,0,0],
                     [0,0,0,0],
                     [0,0,0,0]], dtype=matrix_data_type)

def generator_x_3_4():
    return np.array([[0,0,0,1],
                     [0,0,0,0],
                     [0,0,0,0]], dtype=matrix_data_type)

def generator_x_3_4_neg():
    return np.array([[0,0,0,-1],
                     [0,0,0,0],
                     [0,0,0,0]], dtype=matrix_data_type)


def generator_y():
    return np.array([[0,0,0,0],
                     [0,0,0,1],
                     [0,0,0,0],
                     [0,0,0,0]], dtype=matrix_data_type)

def generator_y_3_4():
    return np.array([[0,0,0,0],
                     [0,0,0,1],
                     [0,0,0,0]], dtype=matrix_data_type)

def generator_y_3_4_neg():
    return np.array([[0,0,0,0],
                     [0,0,0,-1],
                     [0,0,0,0]], dtype=matrix_data_type)



def generator_z():
    return np.array([[0,0,0,0],
                     [0,0,0,0],
                     [0,0,0,1],
                     [0,0,0,0]], dtype=matrix_data_type)

def generator_z_3_4():
    return np.array([[0,0,0,0],
                     [0,0,0,0],
                     [0,0,0,1]], dtype=matrix_data_type)#

def generator_z_3_4_neg():
    return np.array([[0,0,0,0],
                     [0,0,0,0],
                     [0,0,0,-1]], dtype=matrix_data_type)


# Rotation Around X
def generator_roll():
    return np.array([[0,0,0,0],
                     [0,0,-1,0],
                     [0,1,0,0],
                     [0,0,0,0]], dtype=matrix_data_type)

def generator_roll_3_4():
    return np.array([[0,0,0,0],
                     [0,0,-1,0],
                     [0,1,0,0]], dtype=matrix_data_type)

def generator_roll_3_4_neg():
    return np.array([[0,0,0,0],
                     [0,0,1,0],
                     [0,-1,0,0]], dtype=matrix_data_type)


# Rotation Around Y
def generator_pitch():
    return np.array([[0,0,1,0],
                     [0,0,0,0],
                     [-1,0,0,0],
                     [0,0,0,0]], dtype=matrix_data_type)

def generator_pitch_3_4():
    return np.array([[0,0,1,0],
                     [0,0,0,0],
                     [-1,0,0,0]], dtype=matrix_data_type)

def generator_pitch_3_4_neg():
    return np.array([[0,0,-1,0],
                     [0,0,0,0],
                     [1,0,0,0]], dtype=matrix_data_type)


# Rotation Around Z
def generator_yaw():
    return np.array([[0,-1,0,0],
                     [1,0,0,0],
                     [0,0,0,0],
                     [0,0,0,0]], dtype=matrix_data_type)

def generator_yaw_3_4():
    return np.array([[0,-1,0,0],
                     [1,0,0,0],
                     [0,0,0,0]], dtype=matrix_data_type)

def adjoint_se3(R,t):
    t_x = Utils.skew_symmetric(t[0], t[1], t[2])
    top = np.append(R,np.matmul(t_x,R),axis=1)
    bottom = np.append(np.identity(3),R,axis=1)
    return np.append(top,bottom,axis=0)

def exp(w, twist_size):
    w_angle = w[3:twist_size]
    w_angle_transpose = np.transpose(w_angle)
    w_x = Utils.skew_symmetric(w[3], w[4], w[5])
    w_x_squared = np.matmul(w_x, w_x)

    # closed form solution for exponential map
    theta_sqred = np.matmul(w_angle_transpose, w_angle)[0][0]
    theta = math.sqrt(theta_sqred)

    A = 0
    B = 0
    C = 0

    # TODO use Taylor Expansion when theta_sqred is small
    if not theta == 0:
        A = math.sin(theta) / theta

    if not theta_sqred == 0:
        B = (1 - math.cos(theta)) / theta_sqred
        C = (1 - A) / theta_sqred

    u = np.array([w[0], w[1], w[2]]).reshape((3, 1))

    R_new = I_3 + np.multiply(A, w_x) + np.multiply(B, w_x_squared)
    V = I_3 + np.multiply(B, w_x) + np.multiply(C, w_x_squared)

    t_new = np.matmul(V, u)

    return R_new, t_new

def ln(R, t,twist_size):
    w = np.zeros((twist_size,1),dtype=matrix_data_type)

    trace = np.trace(R)
    theta = math.acos((trace-1.0)/2.0)
    theta_sqred = math.pow(theta,2.0)

    R_transpose = np.transpose(R)
    ln_R = I_3

    # TODO use Taylor Expansion when theta is small
    if not theta == 0:
        ln_R = (theta/(2*math.sin(theta)))*(R-R_transpose)

    w[3] = ln_R[2,1]
    w[4] = ln_R[0,2]
    w[5] = ln_R[1,0]

    w_x = Utils.skew_symmetric(w[3], w[4], w[5])
    w_x_squared = np.matmul(w_x, w_x)

    A = 0
    B = 0
    coeff = 0

    # TODO use Taylor Expansion when theta_sqred is small
    if not theta == 0:
        A = math.sin(theta) / theta

    if not theta_sqred == 0:
        B = (1 - math.cos(theta)) / theta_sqred

    if not (theta == 0 or theta_sqred == 0):
        coeff = (1.0/(theta_sqred))*(1.0 - (A/(2.0*B)))

    V_inv = I_3 + np.multiply(0.5,w_x) + np.multiply(coeff,w_x_squared)

    u = np.matmul(V_inv,t)

    w[0] = u[0]
    w[1] = u[1]
    w[2] = u[2]

    return w


