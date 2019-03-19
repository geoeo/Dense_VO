import math
from Numerics.Utils import matrix_to_flat_index_rows
from Numerics.JacobianGenerator import get_jacobian_image
import numpy as np
import time


def back_project_image(width, height, image_range_offset, reference_camera, reference_depth_image, target_depth_image, X_back_projection,
                       valid_measurements,valid_measurements_target, use_ndc, depth_direction, max_depth ):
    start = time.time()
    for y in range(0, height, 1):
        for x in range(0, width, 1):
            flat_index = matrix_to_flat_index_rows(y, x, height)
            depth = reference_depth_image[y, x]
            depth_target = target_depth_image[y, x]
            valid_measurements[flat_index] = True
            valid_measurements_target[flat_index] = True
            # For opencl maybe do this in a simple kernel before
            # Sets invalid depth measurements to False such that they do not impact the gauss newton step
            if depth == 0:
                # this value directly influences the pose estimation!
                # TODO write about this
                depth = depth_direction*(1.0+max_depth)
                #depth = depth_direction*(max_depth)
                #depth = depth_direction*1
                valid_measurements[flat_index] = False
                #continue
            if depth_target == 0:
                valid_measurements_target[flat_index] = False
            depth_ref = depth

            #depending on the direction of the focal length, the depth sign has to be adjusted
            # Since our virtual image plane is on the same side as our depth values
            # we push all depth values out to guarantee that they are always infront of the image plane
            # Better depth results without pushing it out (?)
            if valid_measurements[flat_index]:
                depth_ref = depth_direction*(1.0 + depth)
                #depth_ref = depth_direction*(depth)

            # back projection from ndc seems to give better convergence
            x_back = x
            y_back = y
            if use_ndc:
                x_back /= width
                y_back /= height

            #X_back_projection[0:3, flat_index] = reference_camera.back_project_pixel(x, y, depth_ref)[:, 0]
            X_back_projection[0:3, flat_index] = reference_camera.back_project_pixel(x_back, y_back, depth_ref)[:, 0]
    end = time.time()
    #print('Runtime for Back Project Image:', end - start)


def compute_residual(width, height, target_index_projections, valid_measurements, valid_measurements_target, target_image, reference_image,
                     target_depth, reference_depth, v,
                     image_range_offset):
    v_sum = 0
    start = time.time()
    for y in range(image_range_offset, height - image_range_offset, 1):
        for x in range(image_range_offset, width - image_range_offset, 1):
            flat_index = matrix_to_flat_index_rows(y, x, height)
            v[flat_index][0] = 0
            #if true then invalid depth measurements are being considered
            #if not valid_measurements[flat_index]:
            #    continue
            x_index = target_index_projections[0, flat_index]
            y_index = target_index_projections[1, flat_index]

            if not (image_range_offset < y_index < height - image_range_offset) or not (image_range_offset < x_index < width - image_range_offset):
                # res no flag - these have to be disabled for R300 since the depth image is very sparse!
                valid_measurements[flat_index] = False
                continue
            # A newer SE3 estimate might re-validate a sample / pixel
            # TODO: investigate this flag in thesis
            # Might set invalid depth measurements to True such that they can contribute to the residual
            # res no flag - these have to be disabled for R300 since the depth image is very sparse!
            valid_measurements[flat_index] = True
            x_target = math.floor(x_index)
            y_target = math.floor(y_index)
            flat_index_target = matrix_to_flat_index_rows(y_target, x_target, height)
            #if not valid_measurements_target[flat_index_target]:
            #    continue
            #error = target_image[y_target, x_target] - reference_image[y, x] #1
            error = reference_image[y, x] - target_image[y_target, x_target] #2
            v[flat_index][0] = error

    end = time.time()
    # print('Runtime for Compute Residual:', end-start)
    # return v_sum
    return v

#TODO make own gauss_newton step

def gauss_newton_step_motion_prior(width, height, valid_measurements, valid_measurements_target, W, J_pi, J_lie, target_image_grad_x, target_image_grad_y, v,
                      g, normal_matrix_return, motion_cov_inv, twist_prior, twist_prev, image_range_offset):
    convergence = False
    start = time.time()
    twist_delta = np.subtract(twist_prior, twist_prev)
    motion_prior = np.matmul(motion_cov_inv, twist_delta)

    for y in range(image_range_offset, height - image_range_offset, 1):
        for x in range(image_range_offset, width - image_range_offset, 1):
            flat_index = matrix_to_flat_index_rows(y, x, height)
            if not valid_measurements[flat_index] or not valid_measurements_target[flat_index]:
                continue
            J_image = get_jacobian_image(target_image_grad_x, target_image_grad_y, x, y)
            J_pi_element = J_pi[flat_index]
            J_lie_element = J_lie[flat_index]

            J_pi_lie = np.matmul(J_pi_element, J_lie_element)
            J_full = np.matmul(J_image, J_pi_lie)
            J_t = np.transpose(J_full)
            w_i = W[0,flat_index]
            error_sample = v[flat_index][0]

            g += np.multiply(w_i,np.multiply(-J_t, error_sample))
            normal_matrix_return += np.multiply(w_i,np.matmul(J_t, J_full))

            # TODO: Can optimize this into one mult and 1 add per line
            #g += motion_prior
            #normal_matrix_return += motion_cov_inv
    # different stopping criterion using max norm
    #if math.fabs(np.amax(g)< 0.001):
        #convergence = True
    g += motion_prior
    normal_matrix_return += motion_cov_inv
    end = time.time()

    #print('Runtime Gauss Newton Step:', end-start)
    return convergence


def gauss_newton_step(width, height, valid_measurements, valid_measurements_target, W, J_pi, J_lie, target_image_grad_x, target_image_grad_y, v,
                      g, normal_matrix_return, image_range_offset):
    convergence = False
    start = time.time()
    for y in range(image_range_offset, height - image_range_offset, 1):
        for x in range(image_range_offset, width - image_range_offset, 1):
            flat_index = matrix_to_flat_index_rows(y, x, height)
            if not valid_measurements[flat_index] or not valid_measurements_target[flat_index]:
                continue
            J_image = get_jacobian_image(target_image_grad_x, target_image_grad_y, x, y)
            J_pi_element = J_pi[flat_index]
            J_lie_element = J_lie[flat_index]

            J_pi_lie = np.matmul(J_pi_element, J_lie_element)
            J_full = np.matmul(J_image, J_pi_lie)
            J_t = np.transpose(J_full)
            w_i = W[0,flat_index]
            error_sample = v[flat_index][0]

            g += np.multiply(w_i,np.multiply(-J_t, error_sample))
            normal_matrix_return += np.multiply(w_i,np.matmul(J_t, J_full))

    # different stopping criterion using max norm
    #if math.fabs(np.amax(g)< 0.001):
        #convergence = True
    end = time.time()
    #print('Runtime Gauss Newton Step:', end-start)
    return convergence


def compute_t_dist_variance(v, degrees_of_freedom, N, valid_measurements, valid_measurements_target, number_of_valid_measurements, variance_min, eps):
    variance = variance_min
    variance_prev = variance
    max_it = 100
    for i in range(0,max_it):
        variance = compute_t_dist_variance_round(v, degrees_of_freedom, N, valid_measurements, valid_measurements_target, number_of_valid_measurements,variance_prev)
        if math.fabs(variance_prev - variance) < eps or variance == 0.0:
            break
        variance_prev = variance
        if i == max_it-1:
            print('max variance iteration')
    return variance


def compute_t_dist_variance_round(v, degrees_of_freedom, N, valid_measurements,valid_measurements_target, number_of_valid_measurements, variance_prev):
    numerator = degrees_of_freedom + 1.0
    variance = variance_prev
    for i in range(0,N):
        if not valid_measurements[i] or not valid_measurements_target[i]:
            continue
        if variance == 0.0:
            break
        r = v[i][0]
        r_sq = r*r
        denominator = degrees_of_freedom + (r_sq/variance_prev)
        variance += (numerator/denominator)*r_sq
    variance /= number_of_valid_measurements
    return variance

def generate_weight_matrix(W, v, variance, degrees_of_freedom, N):

    numerator = degrees_of_freedom + 1.0
    for i in range(0,N):
        v_i = v[i][0]
        t = v_i/variance
        t_sq = t*t
        frac = (t_sq/variance)
        W[0,i] = numerator / (degrees_of_freedom + frac)

def multiply_v_by_diagonal_matrix(W,v,N,valid_measurements):
    for i in range(0,N):
        if valid_measurements[i]:
            v[i] *= W[0,i]



