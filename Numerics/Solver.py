import numpy as np
from scipy import linalg
import Numerics.Lie as Lie
import Numerics.Utils as Utils
import math

def solve_SE3(X,Y,max_its,eps):
    #init
    # array for twist values x, y, z, roll, pitch, yaw
    t_est = np.array([0, 0, 0],dtype=np.float64).reshape((3,1))
    R_est = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]],dtype=np.float64)
    (position_vector_size,N) = X.shape
    twist_size = 6
    stacked_obs_size = position_vector_size*N
    v_mean = -1

    SE_3_est = np.append(np.append(R_est,t_est,axis=1),Utils.homogenous_for_SE3(),axis=0)

    for it in range(0,max_its,1):
        # accumulators
        J_v = np.zeros((twist_size,1))
        normal_matrix = np.zeros((twist_size, twist_size))

        Y_est = np.matmul(SE_3_est,X)
        diff = np.multiply(0.25,Y - Y_est)
        v = np.sum(np.square(diff),axis=0)
        v_mean = np.mean(v)

        if v_mean < eps:
            print('done')
            break

        #using generators of so3 to compute derivative with respect to parameters
        G_1_y = np.matmul(Lie.generator_x(),Y_est)
        G_1_y_stacked = np.reshape(G_1_y,(stacked_obs_size,1),order='F')

        G_2_y = np.matmul(Lie.generator_y(),Y_est)
        G_2_y_stacked = np.reshape(G_2_y,(stacked_obs_size,1),order='F')

        G_3_y = np.matmul(Lie.generator_z(), Y_est)
        G_3_y_stacked = np.reshape(G_3_y,(stacked_obs_size,1),order='F')

        G_4_y = np.matmul(Lie.generator_roll(),Y_est)
        G_4_y_stacked = np.reshape(G_4_y,(stacked_obs_size,1),order='F')

        G_5_y = np.matmul(Lie.generator_pitch(), Y_est)
        G_5_y_stacked = np.reshape(G_5_y,(stacked_obs_size,1),order='F')

        G_6_y = np.matmul(Lie.generator_yaw(), Y_est)
        G_6_y_stacked = np.reshape(G_6_y,(stacked_obs_size,1),order='F')

        G_translation = np.append(np.append(G_1_y_stacked,G_2_y_stacked,axis=1),G_3_y_stacked,axis=1)
        G_rot = np.append(np.append(G_4_y_stacked,G_5_y_stacked,axis=1),G_6_y_stacked,axis=1)

        G = np.append(G_translation,G_rot,axis=1)
        Gs = np.hsplit(np.transpose(G),N)

        for i in range(0,N,1):
            G_i = np.multiply(2.0,Gs[i]) # This is J_transpose
            G_i_t = np.transpose(G_i)
            diff_n = np.reshape(diff[:,i],(position_vector_size,1))
            J_v += np.matmul(G_i,diff_n)
            normal_matrix += np.matmul(G_i,G_i_t)

        try:
            pseudo_inv = linalg.inv(normal_matrix)
            #(Q,R) = linalg.qr(normal_matrix_2)
        except:
            print('Cant invert')
            return SE_3_est
        w = np.matmul(pseudo_inv,J_v)
        w_transpose = np.transpose(w)
        w_x = np.array([[0, -w[5], w[4]],
                        [w[5], 0, -w[3]],
                        [-w[4], w[3], 0]],dtype=np.float64)

        #closed form solution for exponential map
        theta = math.sqrt(np.matmul(w_transpose,w))
        theta_sqred = math.pow(theta,2)
        try:
            A = math.sin(theta) / theta
            B = (1 - math.cos(theta)) / (theta_sqred)
        except:
            print('bad theta')
            return SE_3_est

        #translation parameters are just read out
        delta_trans = np.array([w[0], w[1], w[2]])

        t_est = t_est + delta_trans

        R_new = np.identity(3) + np.multiply(A,w_x) + np.multiply(B,np.matmul(w_x,w_x))
        R_est = np.matmul(R_new,R_est)

        SE_3_est = np.append(np.append(R_est,t_est,axis=1),Utils.homogenous_for_SE3(),axis=0)

    print('mean error:',v_mean)
    return SE_3_est