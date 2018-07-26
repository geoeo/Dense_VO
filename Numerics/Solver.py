import numpy as np
from scipy import linalg
import Numerics.Lie as Lie
import Numerics.Utils as Utils
import Numerics.JacobianGenerator as JacobianGenerator
from Numerics.Utils import matrix_data_type
import math


def solve_SE3(X, Y, max_its, eps):
    # init
    # array for twist values x, y, z, roll, pitch, yaw
    t_est = np.array([0, 0, 0], dtype=matrix_data_type).reshape((3, 1))
    R_est = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]], dtype=matrix_data_type)
    I_3 = np.identity(3, dtype=matrix_data_type)
    (position_vector_size, N) = X.shape
    twist_size = 6
    stacked_obs_size = position_vector_size * N
    jacbobi_padding = Utils.padding_for_generator_jacobi()
    homogeneous_se3_padding = Utils.homogenous_for_SE3()
    v_mean = -1
    it = -1

    SE_3_est = np.append(np.append(R_est, t_est, axis=1), Utils.homogenous_for_SE3(), axis=0)

    generator_x = Lie.generator_x()
    generator_y = Lie.generator_y()
    generator_z = Lie.generator_z()
    generator_roll = Lie.generator_roll()
    generator_pitch = Lie.generator_pitch()
    generator_yaw = Lie.generator_yaw()

    for it in range(0, max_its, 1):
        # accumulators
        J_v = np.zeros((twist_size, 1))
        normal_matrix = np.zeros((twist_size, twist_size))

        Y_est = np.matmul(SE_3_est, X)
        #  (0.25f(x))_t * (0.25f(x)) = 0.5 f(x)_t f(x)
        diff = np.multiply(0.25, Y - Y_est)
        v = np.sum(np.square(diff), axis=0)
        v_mean = np.mean(v)

        if v_mean < eps:
            print('done')
            break

        Js = JacobianGenerator.get_jacobians_lie(generator_x, generator_y, generator_z, generator_yaw, generator_pitch,
                                                 generator_roll, Y_est, N, stacked_obs_size)

        for i in range(0, N, 1):
            J = Js[i]
            J_t = np.transpose(J)
            diff_n = np.reshape(diff[:, i], (position_vector_size, 1))
            J_v += np.matmul(J_t, diff_n)
            normal_matrix += np.matmul(J_t, J)

        ##########################################################

        # TODO: Investigate faster inversion with QR
        try:
            pseudo_inv = linalg.inv(normal_matrix)
            # (Q,R) = linalg.qr(normal_matrix_2)
        except:
            print('Cant invert')
            return SE_3_est
        w = np.matmul(pseudo_inv, J_v)
        w_transpose = np.transpose(w)
        w_x = Utils.skew_symmetric(w[3], w[4], w[5])
        w_x_squared = np.matmul(w_x, w_x)

        # closed form solution for exponential map
        theta = math.sqrt(np.matmul(w_transpose, w))
        theta_sqred = math.pow(theta, 2)
        # TODO use Taylor Expansion when theta_sqred is small
        try:
            A = math.sin(theta) / theta
            B = (1 - math.cos(theta)) / theta_sqred
            C = (1 - A) / theta_sqred
        except:
            print('bad theta')
            return SE_3_est

        u = np.array([w[0], w[1], w[2]]).reshape((3, 1))

        R_new = I_3 + np.multiply(A, w_x) + np.multiply(B, w_x_squared)
        V = I_3 + np.multiply(B, w_x) + np.multiply(C, w_x_squared)

        t_est += + np.matmul(V, u)
        R_est = np.matmul(R_new, R_est)

        SE_3_est = np.append(np.append(R_est, t_est, axis=1), homogeneous_se3_padding, axis=0)

    print('mean error:', v_mean, 'iteration: ', it)
    return SE_3_est
