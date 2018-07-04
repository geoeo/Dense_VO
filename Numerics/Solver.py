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
    stacked_obs_size = position_vector_size*N

    SE_3_est = np.append(np.append(R_est,t_est,axis=1),Utils.homogenous_for_SE3(),axis=0)

    for it in range(0,max_its,1):
        # accumulators
        w = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64).reshape((6,1))
        w_sub = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)
        J = np.zeros((1,6))
        J_v = np.zeros((6,1))
        J_v_2 = np.zeros((6,1))
        normal_matrix_2 = np.zeros((6,6))
        normal_matrix = np.zeros((1, 1))
        inv_J = 0

        Y_est = np.matmul(SE_3_est,X)
        diff = np.multiply(0.5,Y - Y_est)
        diff_stacked = np.reshape(diff,(stacked_obs_size,1),order='F')
        v = np.sum(np.square(diff),axis=0)
        v_transpose = np.reshape(v,(N,1))
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
            G_i = Gs[i]
            G_i_t = np.transpose(G_i)
            diff_n = np.reshape(diff[:,i],(4,1))
            v_i = v[i]
            #J_prime = np.matmul(G_i,diff_n).reshape((6,1))
            #J_t_prime = np.reshape(J_prime,(1,6))
            #J_v += np.multiply(J_prime,v_i)
            J_v += np.matmul(G_i,diff_n)
            #normal_matrix += np.matmul(J_t_prime,J_prime)
            normal_matrix_2 += np.matmul(G_i,G_i_t)



        try:
            #pseudo_inv = linalg.inv(normal_matrix)
            pseudo_inv = linalg.inv(normal_matrix_2)
            #(Q,R) = linalg.qr(normal_matrix_2)
        except:
            print('Cant invert')
            return SE_3_est
        #R_inv = linalg.solve_triangular(R, np.identity(6))
        #Q_transpose = np.transpose(Q)
        #pseudo_inv_2 = np.matmul(R_inv,Q_transpose) #TODO: Use scipy for invertion of upper triangluar
        #w = np.multiply(pseudo_inv,J_v)
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

        #TODO: Use summantion of twist, only compute SE3 at the end
        R_new = np.identity(3) + np.multiply(A,w_x) + np.multiply(B,np.matmul(w_x,w_x))
        R_est = np.matmul(R_new,R_est)

        SE_3_est = np.append(np.append(R_est,t_est,axis=1),Utils.homogenous_for_SE3(),axis=0)

    return SE_3_est