import numpy as np
from scipy import linalg
import Numerics.Lie as Lie
import Numerics.Utils as Utils
from Numerics.Utils import matrix_data_type
import math

def solve_SE3(X,Y,max_its,eps):
    #init
    # array for twist values x, y, z, roll, pitch, yaw
    t_est = np.array([0, 0, 0],dtype=matrix_data_type).reshape((3,1))
    R_est = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]],dtype=matrix_data_type)
    I_3 = np.identity(3,dtype=matrix_data_type)
    (position_vector_size,N) = X.shape
    twist_size = 6
    stacked_obs_size = position_vector_size*N
    jacbobi_padding = Utils.padding_for_generator_jacobi()
    homogeneous_se3_padding = Utils.homogenous_for_SE3()
    v_mean = -1
    it = -1

    SE_3_est = np.append(np.append(R_est,t_est,axis=1),Utils.homogenous_for_SE3(),axis=0)

    generator_x = Lie.generator_x()
    generator_y = Lie.generator_y()
    generator_z = Lie.generator_z()
    generator_roll = Lie.generator_roll()
    generator_pitch = Lie.generator_pitch()
    generator_yaw = Lie.generator_yaw()

    for it in range(0,max_its,1):
        # accumulators
        J_v = np.zeros((twist_size,1))
        normal_matrix = np.zeros((twist_size, twist_size))

        Y_est = np.matmul(SE_3_est,X)
        #  (0.25f(x))_t * (0.25f(x)) = 0.5 f(x)_t f(x)
        diff = np.multiply(0.25,Y - Y_est)
        v = np.sum(np.square(diff),axis=0)
        v_mean = np.mean(v)

        if v_mean < eps:
            print('done')
            break

        # TODO: Profile Variants

        # Variant 1
        # using generators of so3 to compute derivative with respect to parameters
        G_1_y = np.matmul(generator_x,Y_est)
        G_1_y_stacked = np.reshape(G_1_y,(stacked_obs_size,1),order='F')

        G_2_y = np.matmul(generator_y,Y_est)
        G_2_y_stacked = np.reshape(G_2_y,(stacked_obs_size,1),order='F')

        G_3_y = np.matmul(generator_z, Y_est)
        G_3_y_stacked = np.reshape(G_3_y,(stacked_obs_size,1),order='F')

        G_4_y = np.matmul(generator_roll,Y_est)
        G_4_y_stacked = np.reshape(G_4_y,(stacked_obs_size,1),order='F')

        G_5_y = np.matmul(generator_pitch, Y_est)
        G_5_y_stacked = np.reshape(G_5_y,(stacked_obs_size,1),order='F')

        G_6_y = np.matmul(generator_yaw, Y_est)
        G_6_y_stacked = np.reshape(G_6_y,(stacked_obs_size,1),order='F')

        G_translation = np.append(np.append(G_1_y_stacked,G_2_y_stacked,axis=1),G_3_y_stacked,axis=1)
        G_rot = np.append(np.append(G_4_y_stacked,G_5_y_stacked,axis=1),G_6_y_stacked,axis=1)
        G = np.append(G_translation,G_rot,axis=1)

        Gs = np.hsplit(np.transpose(G),N)

        for i in range(0,N,1):
               G_i = Gs[i]
               J_t = np.multiply(2.0,G_i)
               J = np.transpose(J_t)
               diff_n = np.reshape(diff[:,i],(position_vector_size,1))
               J_v += np.matmul(J_t,diff_n)
               normal_matrix += np.matmul(J_t,J)

        ##########################################################

        # Variant #2 - A lot slower! 1000ms Probably due to memory alloc in loop
        #for i in range(0,N,1):
        #     Y_est_i = Y_est[:,i]
        #     y_x = np.multiply(-1,Utils.skew_symmetric(Y_est_i[0],Y_est_i[1],Y_est_i[2]))
        #     J = np.multiply(2,np.append(np.append(I_3,y_x,axis=1),jacbobi_padding,axis=0))
        #     J_t = np.transpose(J)
        #     diff_n = np.reshape(diff[:,i],(position_vector_size,1))
        #     J_v += np.matmul(J_t,diff_n)
        #     normal_matrix += np.matmul(J_t,J)

        # TODO: Investigate faster inversion with QR
        try:
            pseudo_inv = linalg.inv(normal_matrix)
            #(Q,R) = linalg.qr(normal_matrix_2)
        except:
            print('Cant invert')
            return SE_3_est
        w = np.matmul(pseudo_inv,J_v)
        w_transpose = np.transpose(w)
        w_x = Utils.skew_symmetric(w[3],w[4],w[5])
        w_x_squared = np.matmul(w_x,w_x)

        # closed form solution for exponential map
        theta = math.sqrt(np.matmul(w_transpose,w))
        theta_sqred = math.pow(theta,2)
        # TODO use Taylor Expansion when theta_sqred is small
        try:
            A = math.sin(theta) / theta
            B = (1 - math.cos(theta)) / theta_sqred
            C = (1 - A) / theta_sqred
        except:
            print('bad theta')
            return SE_3_est

        u = np.array([w[0], w[1], w[2]]).reshape((3,1))

        R_new = I_3 + np.multiply(A,w_x) + np.multiply(B,w_x_squared)
        V = I_3 + np.multiply(B,w_x) + np.multiply(C,w_x_squared)

        # TODO: Investigate with adding u works as well
        t_est += + np.matmul(V,u)
        #t_est += u
        R_est = np.matmul(R_new,R_est)

        SE_3_est = np.append(np.append(R_est,t_est,axis=1),homogeneous_se3_padding,axis=0)

    print('mean error:',v_mean, 'iteation: ', it)
    return SE_3_est