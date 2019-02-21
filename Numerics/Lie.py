import numpy as np
import math
from Numerics.Utils import matrix_data_type
from Numerics import Utils, SE3


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

def generator_yaw_3_4_neg():
    return np.array([[0,1,0,0],
                     [-1,0,0,0],
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

def lie_ackermann_correction(gradient_step, motion_cov_inv, ackermann_twist, vo_twist, twist_size):
    # ack_prior = np.multiply(Gradient_step_manager.current_alpha,ackermann_pose_prior)
    ack_prior = ackermann_twist

    #ack_prior = np.matmul(motion_cov_inv, ack_prior) # 1
    ack_prior = np.multiply(gradient_step, ack_prior) # 2

    R_w, t_w = exp(vo_twist, twist_size)
    R_ack, t_ack = exp(ack_prior, twist_size)

    SE_3_w = np.append(np.append(R_w, t_w, axis=1), Utils.homogenous_for_SE3(), axis=0)
    SE_3_ack = np.append(np.append(R_ack, t_ack, axis=1), Utils.homogenous_for_SE3(), axis=0)

    SE3_w_ack = SE3.pose_pose_composition_inverse(SE_3_w, SE_3_ack)

    w_inc = ln(SE3.extract_rotation(SE3_w_ack), SE3.extract_translation(SE3_w_ack), twist_size)

    #w_inc = np.multiply(gradient_step, np.matmul(motion_cov_inv, w_inc)) # 1
    w_inc = np.matmul(motion_cov_inv, w_inc) # 2
    #w_inc = np.multiply(gradient_step, w_inc)

    return w_inc


