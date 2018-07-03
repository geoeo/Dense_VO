import numpy as np
import Numerics.Lie as Lie
import Numerics.Utils as Utils
import math

def solve_SE3(X,Y,max_its,eps):
    #init
    # array for twist values x, y, z, roll, pitch, yaw
    t_est = np.array([0, 0, 0],dtype=np.float32).reshape((3,1))
    R_est = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]],dtype=np.float32)
    (position_vector_size,N) = X.shape
    stacked_obs_size = position_vector_size*N

    SE_3_est = np.append(np.append(R_est,t_est,axis=1),Utils.homogenous_for_SE3(),axis=0)

    for it in range(0,max_its,1):
        # accumulators
        w = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
        w_sub = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
        J = np.empty((1,6))
        J_v = np.empty((6,1))
        normal_matrix = np.empty((6,6))
        inv_J = 0


        Y_est = np.matmul(SE_3_est,X)
        diff = Y - Y_est
        diff_stacked = np.reshape(diff,(stacked_obs_size,1),order='F')
        v = np.sum(np.square(diff),axis=0)
        v_transpose = np.reshape(v,(N,1))
        v_mean = np.mean(v)

        if v_mean < eps:
            print('done')
            break

        #for n in range(0,N,1):
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

        #G_translation = np.append(np.append(G_1_y,G_2_y,axis=0),G_3_y,axis=0)
        #G_rot = np.append(np.append(G_4_y,G_5_y,axis=0),G_6_y,axis=0)

        G = np.append(G_translation,G_rot,axis=1)
        Gs = np.hsplit(np.transpose(G),N)


        for i in range(0,N,1):
            G_i = Gs[i]
            Y_est_i = Y_est[:,i]
            v_i = v[i]
            J_prime = np.matmul(G_i,Y_est_i).reshape(1,6)
            J_t_prime = np.reshape(J_prime,(6,1))
            J += J_prime
            J_v += np.multiply(J_t_prime,v_i)
            normal_matrix += np.matmul(J_t_prime,J_prime)

        t = 2
        #J_transpose = np.reshape(J,(6,1))
        #normal_matrix_2 = np.matmul(J_transpose,J)
        pseudo_inv = np.linalg.inv(normal_matrix)
        #pseudo_inv_2 = np.linalg.inv(normal_matrix_2)
        #(Q,R) = np.linalg.qr(normal_matrix)
        w = np.matmul(pseudo_inv,J_v)
        #J_v = np.matmul(J_transpose,v_transpose)
        t= 1
