import numpy as np
import Numerics.SE3 as SE3
import Camera.Intrinsic as Intrinsic
from Numerics.Utils import matrix_data_type
import math


def get_jacobians_lie(generator_x,generator_y,generator_z,generator_yaw,generator_pitch,generator_roll,Y_est,N,stacked_obs_size):
    # Variant 1
    # using generators of so3 to compute derivative with respect to parameters
    G_1_y = np.matmul(generator_x, Y_est)
    G_1_y_stacked = np.reshape(G_1_y, (stacked_obs_size, 1), order='F')

    G_2_y = np.matmul(generator_y, Y_est)
    G_2_y_stacked = np.reshape(G_2_y, (stacked_obs_size, 1), order='F')

    G_3_y = np.matmul(generator_z, Y_est)
    G_3_y_stacked = np.reshape(G_3_y, (stacked_obs_size, 1), order='F')

    G_4_y = np.matmul(generator_roll, Y_est)
    G_4_y_stacked = np.reshape(G_4_y, (stacked_obs_size, 1), order='F')

    G_5_y = np.matmul(generator_pitch, Y_est)
    G_5_y_stacked = np.reshape(G_5_y, (stacked_obs_size, 1), order='F')

    G_6_y = np.matmul(generator_yaw, Y_est)
    G_6_y_stacked = np.reshape(G_6_y, (stacked_obs_size, 1), order='F')

    G_translation = np.append(np.append(G_1_y_stacked, G_2_y_stacked, axis=1), G_3_y_stacked, axis=1)
    G_rot = np.append(np.append(G_4_y_stacked, G_5_y_stacked, axis=1), G_6_y_stacked, axis=1)
    G = np.append(G_translation, G_rot, axis=1)

    Js = np.multiply(2.0, np.vsplit(G, N))
    return Js

# Variant #2 - A lot slower! 1000ms Probably due to memory alloc in loop
# for i in range(0,N,1):
#     Y_est_i = Y_est[:,i]
#     y_x = np.multiply(-1,Utils.skew_symmetric(Y_est_i[0],Y_est_i[1],Y_est_i[2]))
#     J = np.multiply(2,np.append(np.append(I_3,y_x,axis=1),jacbobi_padding,axis=0))
#     J_t = np.transpose(J)
#     diff_n = np.reshape(diff[:,i],(position_vector_size,1))
#     J_v += np.matmul(J_t,diff_n)
#     normal_matrix += np.matmul(J_t,J)

#def get_jacobian_rigid_body(se3):
#    translation = SE3.extract_translation(se3)
#    x = translation[0]
#    y = translation[1]
#    z = translation[2]
#    jacobian_rigid = np.array([[x, 0, 0,y,0,0,z,0,0,1,0,0],
#                               [0, x, 0,0,y,0,0,z,0,0,1,0],
#                               [0, 0, x,0,0,y,0,0,z,0,0,1]], dtype=matrix_data_type)
#    return jacobian_rigid

def get_jacobian_camera_model(intrinsics : Intrinsic.Intrinsic,X):
    #translation = SE3.extract_translation(se3)
    #x = translation[0]
    #y = translation[1]
    #z = translation[2]
    (vector_size,N) = X.shape
    jacobian_camera = np.zeros((2,3,N),matrix_data_type)
    for i in range(0,N):
        x = X[0,i]
        y = X[1,i]
        z = X[2,i]
        f_x = intrinsics.extract_fx()
        f_y = intrinsics.extract_fy()
        z_sqrd = math.pow(z,2)

        v11 = f_x/z
        v22 = f_y/z
        v13 = (-f_x*x)/z_sqrd
        v23 = (-f_y*y)/z_sqrd

        jacobian_camera[:,i] = np.array([[v11, 0, v13],
                                         [0, v22, v23]], dtype=matrix_data_type)
    return jacobian_camera

def get_jacobian_image(image_g_x,image_g_y,x,y):
    jacobian_image = np.array([0,0],dtype=matrix_data_type)
    jacobian_image[0] = image_g_x[y,x]
    jacobian_image[1] = image_g_y[y,x]
    return jacobian_image

