import numpy as np
import cv2
from scipy import linalg
import Numerics.Lie as Lie
import Numerics.Utils as Utils
import Numerics.JacobianGenerator as JacobianGenerator
from Numerics.Utils import matrix_data_type
import math
import Numerics.ImageProcessing as ImageProcessing


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
    homogeneous_se3_padding = Utils.homogenous_for_SE3()
    L_mean = -1
    it = -1
    # Step Factor
    alpha = 0.125

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
        v = Y_est - Y

        L = np.sum(np.square(v), axis=0)
        L_mean = np.mean(L)

        if L_mean < eps:
            print('done')
            break

        Js = JacobianGenerator.get_jacobians_lie(generator_x, generator_y, generator_z, generator_yaw, generator_pitch,
                                                 generator_roll, Y_est, N, stacked_obs_size, coefficient=2.0)

        for i in range(0, N, 1):
            J = Js[i]
            J_t = np.transpose(J)
            error_vector = np.reshape(v[:, i], (position_vector_size, 1))
            J_v += np.matmul(J_t, error_vector)
            normal_matrix += np.matmul(-J_t, J)

        ##########################################################

        # TODO: Investigate faster inversion with QR
        try:
            pseudo_inv = linalg.inv(normal_matrix)
            # (Q,R) = linalg.qr(normal_matrix_2)
        except:
            print('Cant invert')
            return SE_3_est
        w = np.matmul(pseudo_inv, J_v)
        # Apply Step Factor
        w = alpha*w

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

    print('mean error:', L_mean, 'iteration: ', it)
    return SE_3_est


# TODO: TEST!
# TODO: implement the SE3 estimation with the objective function in image space
def solve_photometric(frame_reference, frame_target, max_its, eps, debug = False):
    # init
    # array for twist values x, y, z, roll, pitch, yaw
    t_est = np.array([0, 0, 0], dtype=matrix_data_type).reshape((3, 1))
    R_est = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]], dtype=matrix_data_type)
    I_3 = np.identity(3, dtype=matrix_data_type)

    (height,width) = frame_target.pixel_image.shape
    N = height*width
    position_vector_size = 3
    twist_size = 6
    stacked_obs_size = position_vector_size * N
    homogeneous_se3_padding = Utils.homogenous_for_SE3()
    # Step Factor
    #alpha = 0.125
    alpha = 1
    index_array = np.zeros((1,2),matrix_data_type)
    valid_image_range = 10

    SE_3_est = np.append(np.append(R_est, t_est, axis=1), Utils.homogenous_for_SE3(), axis=0)

    generator_x = Lie.generator_x_3_4()
    generator_y = Lie.generator_y_3_4()
    generator_z = Lie.generator_z_3_4()
    generator_roll = Lie.generator_roll_3_4()
    generator_pitch = Lie.generator_pitch_3_4()
    generator_yaw = Lie.generator_yaw_3_4()

    X = np.ones((4, N),Utils.matrix_data_type)
    valid_measurements = np.full(N,False)

    #TODO: Optimize
    # Precompute back projection of pixels
    for y in range(0, height, 1):
        for x in range(0, width, 1):
            flat_index = Utils.matrix_to_flat_index_rows(y,x,height)
            depth = frame_reference.pixel_depth[y, x]
            X[0:3,flat_index] = frame_reference.camera.back_project_pixel(x, y, depth)[:,0]

    if debug:
        # render/save image of projected, back projected points
        projected_back_projected = frame_reference.camera.apply_perspective_pipeline(X)
        # scale ndc if applicable
        #projected_back_projected[0,:] = projected_back_projected[0,:]*width
        #projected_back_projected[1,:] = projected_back_projected[1,:]*height
        debug_buffer = np.zeros((height,width), dtype=np.float64)
        for i in range(0,N,1):
            u = projected_back_projected[0,i]
            v = projected_back_projected[1,i]

            if not np.isnan(u) and not np.isnan(v):
                debug_buffer[int(v),int(u)] = 1.0
        cv2.imwrite("debug_buffer.png", ImageProcessing.normalize_to_image_space(debug_buffer))


    # Precompute the Jacobian of SE3 around the identity
    J_lie = JacobianGenerator.get_jacobians_lie(generator_x, generator_y, generator_z, generator_yaw,
                                                     generator_pitch,
                                                     generator_roll, X, N, stacked_obs_size,coefficient=1.0)

    # Precompute the Jacobian of the projection function
    J_pi = JacobianGenerator.get_jacobian_camera_model(frame_reference.camera.intrinsic, X, valid_measurements)
    # count the number of true
    number_of_valid_measurements = np.sum(valid_measurements)

    # vectorize image
    image_key_flat = np.reshape(frame_reference.pixel_image, (N, 1), order='F')
    image_warped = np.full((height, width),-1, dtype=matrix_data_type)

    for it in range(0, max_its, 1):
        # accumulators
        J_v = np.zeros((twist_size, 1))
        normal_matrix = np.zeros((twist_size, twist_size))

        # Warp with the current SE3 estimate
        Y_est = np.matmul(SE_3_est, X)
        L = 0
        L_mean = -10000

        target_index_projections = frame_target.camera.apply_perspective_pipeline(Y_est)

        # Compute residual
        #TODO: Optimize this
        for y in range(0, height, 1):
            for x in range(0, width, 1):
                flat_index = Utils.matrix_to_flat_index_rows(y, x, height)
                x_index = target_index_projections[0,flat_index]
                y_index = target_index_projections[1,flat_index]
                # TODO: check if projected uv are valid image addresses
                if not valid_measurements[flat_index]:
                    continue
                x_target = math.floor(x_index)
                y_target = math.floor(y_index)
                image_warped[y,x] = frame_target.pixel_image[y_target,x_target]

        image_warped_flat = np.reshape(image_warped, (N, 1), order='F')
        #v = image_warped_flat - image_key_flat
        v = image_key_flat - image_warped_flat

        for y in range(0,height,1):
            for x in range(0,width,1):
                flat_index = Utils.matrix_to_flat_index_rows(y, x, height)
                if valid_measurements[flat_index]:
                    L += v[flat_index,0]
        L_mean = L / number_of_valid_measurements
        #L = np.sum(np.square(v), axis=0)
        #L_mean = np.mean(L)

        if np.abs(L_mean) < eps:
            print('done')
            break

        #TODO: Optimize this
        for y in range(0,height,1):
            for x in range(0,width,1):
                flat_index = Utils.matrix_to_flat_index_rows(y, x, height)
                if not valid_measurements[flat_index]:
                    continue
                J_image = JacobianGenerator.get_jacobian_image(frame_target.grad_x, frame_target.grad_y, x, y)
                J_pi_element = J_pi[flat_index]
                J_lie_element = J_lie[flat_index]

                J_pi_lie = np.matmul(J_pi_element,J_lie_element)
                J_full = np.matmul(J_image,J_pi_lie)
                J_t = np.transpose(J_full)
                error_vector = v[flat_index][0]
                J_v += np.multiply(error_vector,J_t)
                normal_matrix += np.matmul(-J_t, J_full)

        # TODO: Investigate faster inversion with QR
        try:
            pseudo_inv = linalg.inv(normal_matrix)
            # (Q,R) = linalg.qr(normal_matrix_2)
        except:
            print('Cant invert')
            return SE_3_est
        w = np.matmul(pseudo_inv, J_v)
        # Apply Step Factor
        w = alpha*w

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
        print('Runtime: mean error:', L_mean)

    print('mean error:', L_mean, 'iteration: ', it)

    return SE_3_est
