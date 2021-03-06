import numpy as np
import cv2
import math
from scipy import linalg
from Numerics import Lie, Utils, ImageProcessing, JacobianGenerator
from Numerics.Utils import matrix_data_type
from Numerics import SE3
from VisualOdometry import GradientStepManager
from VisualOdometry import GaussNewtonRoutines
from Visualization import Plot3D
import time
from sys import  maxsize

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
            J_v += np.matmul(-J_t, error_vector)
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
        print('Runtime mean error:', L_mean)

    print('mean error:', L_mean, 'iteration: ', it)
    return SE_3_est


def solve_photometric(frame_reference,
                      frame_target,
                      threadLock,
                      pose_estimate_list,
                      max_its,
                      eps,
                      alpha_step,
                      gradient_monitoring_window_start,
                      image_range_offset_start,
                      max_depth,
                      twist_prior = None,
                      motion_cov_inv_in = None,
                      use_ndc = False,
                      use_robust = False,
                      track_pose_estimates = False,
                      use_motion_prior = False,
                      ackermann_pose_prior = None,
                      use_ackermann = False,
                      debug = False):

    if track_pose_estimates and (threadLock == None or pose_estimate_list == None):
        raise RuntimeError('Visualization Flag is set, but no list and lock are supplied')

    # init
    # array for twist values x, y, z, roll, pitch, yaw
    t_est = np.array([0, 0, 0], dtype=matrix_data_type).reshape((3, 1))
    #R_est = np.array([[0.0, -1.0, 0],
    #                  [1.0, 0.0, 0],
    #                  [0, 0, 1]], dtype=matrix_data_type)
    R_est = np.identity(3, dtype=matrix_data_type)
    I_3 = np.identity(3, dtype=matrix_data_type)
    I_4 = np.identity(4,dtype=matrix_data_type)
    I_6 = np.identity(6,dtype=matrix_data_type)
    zero_cov = np.zeros((6,6),dtype=matrix_data_type)
    #SE3_best = np.identity(4,dtype=matrix_data_type)
    (height,width) = frame_target.pixel_image.shape
    N = height*width
    position_vector_size = 3
    twist_size = 6
    stacked_obs_size = position_vector_size * N
    homogeneous_se3_padding = Utils.homogenous_for_SE3()
    variance = -1
    v_mean = maxsize
    image_range_offset = image_range_offset_start
    degrees_of_freedom = 5.0 # empirically derived: see paper
    normal_matrix_ret = np.identity(6, dtype=Utils.matrix_data_type)
    motion_cov_inv = motion_cov_inv_in
    #motion_cov_inv = np.linalg.inv(motion_cov_inv_in)
    w = np.zeros((twist_size,1),dtype=Utils.matrix_data_type)
    w_empty = np.zeros((twist_size,1),dtype=Utils.matrix_data_type)
    w_prev = np.zeros((twist_size,1),dtype=Utils.matrix_data_type)
    w_acc = np.zeros((twist_size,1),dtype=Utils.matrix_data_type)
    v_id = np.zeros((N, 1), dtype=matrix_data_type, order='F')
    pseudo_inv = np.identity(twist_size,dtype=matrix_data_type)
    not_better = False
    valid_pixel_ratio = 1.0
    motion_cov_inv_norm = Utils.norm_covariance_row(motion_cov_inv_in)


    fx = frame_reference.camera.intrinsic.extract_fx()
    fy = frame_reference.camera.intrinsic.extract_fy()

    depth_factor = np.sign(fx)
    #depth_factor = -np.sign(fx)

    Gradient_step_manager = GradientStepManager.GradientStepManager(alpha_start = alpha_step,
                                                                    alpha_min = -0.7,
                                                                    alpha_step = -0.01 ,
                                                                    alpha_change_rate = 0,
                                                                    gradient_monitoring_window_start = gradient_monitoring_window_start,
                                                                    gradient_monitoring_window_size = 0)



    SE_3_est = np.append(np.append(R_est, t_est, axis=1), Utils.homogenous_for_SE3(), axis=0)
    SE_3_prev = np.append(np.append(R_est, t_est, axis=1), Utils.homogenous_for_SE3(), axis=0)
    #SE_3_est_orig = np.append(np.append(R_est, t_est, axis=1), Utils.homogenous_for_SE3(), axis=0)
    #SE_3_est_last_valid = np.append(np.append(R_est, t_est, axis=1), Utils.homogenous_for_SE3(), axis=0)

    generator_x = Lie.generator_x_3_4()
    #generator_x = Lie.generator_x_3_4_neg()
    generator_y = Lie.generator_y_3_4()
    #generator_y = Lie.generator_y_3_4_neg()
    #generator_z = Lie.generator_z_3_4()
    generator_z = Lie.generator_z_3_4_neg()

    # Depth factor of -1.0 leads to inverted roll and pitch when displaying
    # Why?: Generator defines the direction of increase (My thoughts)
    generator_roll = Lie.generator_roll_3_4()
    #generator_roll = Lie.generator_roll_3_4_neg()
    #generator_pitch = Lie.generator_pitch_3_4()
    generator_pitch = Lie.generator_pitch_3_4_neg()
    generator_yaw = Lie.generator_yaw_3_4()
    #generator_yaw = Lie.generator_yaw_3_4_neg()

    X_back_projection = depth_factor*np.ones((4, N), Utils.matrix_data_type)
    X_back_projection[3,:] = 1.0
    #X_back_projection_last_valid = np.ones((4, N), Utils.matrix_data_type)
    valid_measurements_reference = np.full(N,False)
    valid_measurements_target = np.full(N, False)
    #valid_measurements_last = np.full(N,False)
    #valid_measurements_target = np.full(N,False)
    valid_measurements = valid_measurements_reference
    number_of_valid_measurements = N
    #v = np.zeros((N, 1), dtype=matrix_data_type, order='F')

    # Precompute back projection of pixels
    GaussNewtonRoutines.back_project_image(width,
                                       height,
                                       image_range_offset,
                                       frame_reference.camera,
                                       frame_reference.pixel_depth,
                                       frame_target.pixel_depth,
                                       X_back_projection,
                                       valid_measurements,
                                       valid_measurements_target,
                                       use_ndc,
                                       depth_factor,
                                       max_depth)

    count = np.sum(valid_measurements)
    count_target = np.sum(valid_measurements_target)

    z_rot = SE3.makeS03(0,0,math.pi)
    se3_rot = np.identity(4, dtype=matrix_data_type)
    se3_rot[0:3,0:3] = z_rot
    #X_back_projection = np.matmul(se3_rot,X_back_projection)

    if debug:
        Plot3D.save_projection_of_back_projected(height,width,frame_reference,X_back_projection)

    # Precompute the Jacobian of SE3 around the identity
    J_lie = JacobianGenerator.get_jacobians_lie(generator_x, generator_y, generator_z, generator_yaw,
                                                generator_pitch,
                                                generator_roll, X_back_projection, N, stacked_obs_size, coefficient=1.0)

    # Precompute the Jacobian of the projection function
    J_pi = JacobianGenerator.get_jacobian_camera_model(frame_reference.camera.intrinsic, X_back_projection)

    # count the number of true
    #valid_measurements_total = np.logical_and(valid_measurements_reference,valid_measurements_target)

    #number_of_valid_reference = np.sum(valid_measurements_reference)
    #number_of_valid_total = np.sum(valid_measurements_total)
    #number_of_valid_measurements = number_of_valid_reference

    #target_index_projections_id = frame_target.camera.apply_perspective_pipeline(I_4)

    target_index_projections = frame_target.camera.apply_perspective_pipeline(X_back_projection, use_ndc, width, height)

    GaussNewtonRoutines.compute_residual(width,
                                         height,
                                         target_index_projections,
                                         valid_measurements,
                                         valid_measurements_target,
                                         frame_target.pixel_image,
                                         frame_reference.pixel_image,
                                         frame_target.pixel_depth,
                                         frame_reference.pixel_depth,
                                         v_id,
                                         image_range_offset)

    v = np.copy(v_id)
    W = np.ones((1, N), dtype=matrix_data_type, order='F')

    for it in range(0, max_its, 1):
        start = time.time()
        # accumulators
        #TODO: investigate preallocate and clear in a for loop
        g = np.zeros((twist_size, 1))
        normal_matrix = np.identity(twist_size, dtype=matrix_data_type)


        # TODO investigate performance impact
        if track_pose_estimates:
            threadLock.acquire()
            pose_estimate_list.append(SE_3_est)
            threadLock.release()

        #v_diff = math.fabs(Gradient_step_manager.last_error_mean_abs - v_mean)
        #v_diff = Gradient_step_manager.last_error_mean_abs - v_mean

        #Gradient_step_manager.track_gradient(v_mean,it)

        # TODO investigate absolute error threshold aswel?
        #if ((v_diff <= eps)) and Gradient_step_manager.check_iteration(it) :
        #    print('done, mean error:', v_mean, 'diff: ', v_diff, 'pixel ratio:', valid_pixel_ratio)
        #    break

        # no if statement means solver 2
        #if v_mean <= Gradient_step_manager.last_error_mean_abs:
        not_better = False
        prior_empty = False
        if twist_prior[0] == 0 and twist_prior[1] == 0 and twist_prior[2] == 0 and twist_prior[3] == 0 and \
                twist_prior[4] == 0 and twist_prior[5] == 0:
            prior_empty = True

        if use_motion_prior:
            converged = GaussNewtonRoutines.gauss_newton_step_motion_prior(width,
                                              height,
                                              valid_measurements,
                                              valid_measurements_target,
                                              W,
                                              J_pi,
                                              J_lie,
                                              frame_target.grad_x,
                                              frame_target.grad_y,
                                              v,
                                              g,
                                              normal_matrix,
                                              motion_cov_inv,
                                              twist_prior,
                                              w,
                                              image_range_offset)
        else:
            converged = GaussNewtonRoutines.gauss_newton_step(width,
                                              height,
                                              valid_measurements,
                                              valid_measurements_target,
                                              W,
                                              J_pi,
                                              J_lie,
                                              frame_target.grad_x,
                                              frame_target.grad_y,
                                              v,
                                              g,
                                              normal_matrix,
                                              image_range_offset)
        normal_matrix_ret = normal_matrix

        #try:
        #    pseudo_inv = linalg.inv(normal_matrix)
        #except:
        #    print('Cant invert')
        #    return SE_3_est

        #w_new = np.matmul(pseudo_inv, g)

        try:
            w_new = linalg.solve(normal_matrix, g)
        except:
            print('Cant solve')
            return SE_3_est

        # initial step with empty motion prior seems to be quite large
        #if use_motion_prior and prior_empty:
        #    w_new = np.multiply(Gradient_step_manager.current_alpha/2.0, w_new)
        #else:
        w_new = np.multiply(Gradient_step_manager.current_alpha, w_new)

        # For using ackermann motion
        if use_ackermann:
            # V1
            # inc = ackermann_pose_prior - w
            # w_new += np.matmul(motion_cov_inv,inc)
            # w_new += inc

            # V2
            #factor = 0.1*Gradient_step_manager.current_alpha
            factor = 0.1
            #factor = math.pow(Gradient_step_manager.current_alpha,it)
            # ack_prior = np.multiply(Gradient_step_manager.current_alpha,ackermann_pose_prior)
            ack_prior = ackermann_pose_prior

            w_new += Lie.lie_ackermann_correction(factor, motion_cov_inv, ack_prior, w, twist_size)


        #else:
        #    not_better = True
        #    w_new = w_empty





        R_cur, t_cur = Lie.exp(w,twist_size)
        R_new, t_new = Lie.exp(w_new,twist_size)

        # C_new . C_cur
        #t_est = np.add(np.matmul(R_new, t_cur), t_new)
        #R_est = np.matmul(R_new, R_cur)

        # C_Cur . C_new
        t_est = np.add(np.matmul(R_cur, t_new), t_cur)
        R_est = np.matmul(R_cur,R_new)

        w = Lie.ln(R_est, t_est, twist_size)

        #SE_3_current = np.append(np.append(R_cur, t_cur, axis=1), homogeneous_se3_padding, axis=0)
        SE_3_est = np.append(np.append(R_est, t_est, axis=1), homogeneous_se3_padding, axis=0)

        #debug_list  = [i for i, x in enumerate(valid_measurements) if x]

        Y_est = np.matmul(SE_3_est, X_back_projection)

        target_index_projections = frame_target.camera.apply_perspective_pipeline(Y_est, use_ndc, width, height)
        #target_index_projections[2,:] -= depth_factor*1

        v = GaussNewtonRoutines.compute_residual(width,
                                                 height,
                                                 target_index_projections,
                                                 valid_measurements,
                                                 valid_measurements_target,
                                                 frame_target.pixel_image,
                                                 frame_reference.pixel_image,
                                                 frame_target.pixel_depth,
                                                 frame_reference.pixel_depth,
                                                 v,
                                                 image_range_offset)



        number_of_valid_measurements = np.sum(valid_measurements)
        valid_pixel_ratio = number_of_valid_measurements / N

        if number_of_valid_measurements <= 0 and Gradient_step_manager.check_iteration(it):
            print('pixel ratio break')
            print('done, mean error:', v_mean, 'diff: ', v_diff, 'pixel ratio:', valid_pixel_ratio)
            #SE_3_est = SE3_best
            break

        if use_robust:
            variance = GaussNewtonRoutines.compute_t_dist_variance(v, degrees_of_freedom, N, valid_measurements,
                                                                   valid_measurements_target,number_of_valid_measurements,
                                                                   variance_min=1000,
                                                                   eps=0.001)
            if variance > 0.0:
                # clear old weights
                for i in range(0,N):
                    W[0,i] = 1
                GaussNewtonRoutines.generate_weight_matrix(W, v, variance, degrees_of_freedom, N)


        v_weighted = np.copy(v)
        GaussNewtonRoutines.multiply_v_by_diagonal_matrix(W,v_weighted,N,valid_measurements)

        v_sum = np.matmul(np.transpose(v),v_weighted)[0][0]

        end = time.time()

        #if v_mean < Gradient_step_manager.last_error_mean_abs:
        #    SE3_best = np.copy(SE_3_est)
        #if not not_better: # solver 6
        Gradient_step_manager.save_previous_mean_error(v_mean)

        if number_of_valid_measurements > 0:
            v_mean = v_sum / number_of_valid_measurements
        else:
            v_mean = maxsize

        v_diff = math.fabs(Gradient_step_manager.last_error_mean_abs - v_mean)
        print('mean error:', v_mean, 'error diff: ',v_diff, 'iteration: ', it,'valid pixel ratio: ', valid_pixel_ratio, 'runtime: ', end-start, 'variance: ', variance)
        #v_diff = Gradient_step_manager.last_error_mean_abs - v_mean

        #Gradient_step_manager.track_gradient(v_mean,it)

        # TODO investigate absolute error threshold aswel?
        if ((v_diff <= eps)) and Gradient_step_manager.check_iteration(it) :
            print('done, mean error:', v_mean, 'diff: ', v_diff, 'pixel ratio:', valid_pixel_ratio)
            break


    motion_cov_inv = normal_matrix_ret

    return SE_3_est, w, motion_cov_inv
