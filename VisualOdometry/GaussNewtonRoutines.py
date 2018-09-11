import math
from Numerics.Utils import matrix_to_flat_index_rows
from Numerics.JacobianGenerator import get_jacobian_image
import numpy as np
import time


def back_project_image(width, height, reference_camera, reference_depth_image, X_back_projection,
                       image_range_offset):
    start = time.time()
    for y in range(image_range_offset, height - image_range_offset, 1):
        for x in range(image_range_offset, width - image_range_offset, 1):
            flat_index = matrix_to_flat_index_rows(y, x, height)
            depth_ref = reference_depth_image[y, x]
            if depth_ref == 0:
                depth_ref = 1000
            # back projection from ndc seems to give better convergence
            #X_back_projection[0:3, flat_index] = reference_camera.back_project_pixel(x, y, depth_ref)[:, 0]
            X_back_projection[0:3, flat_index] = reference_camera.back_project_pixel(x/width, y/height, depth_ref)[:, 0]
    end = time.time()
    #print('Runtime for Back Project Image:', end - start)


def compute_residual(width, height, target_index_projections, valid_measurements, target_image, reference_image, v, image_range_offset):
    v_sum = 0
    start = time.time()
    for y in range(image_range_offset, height-image_range_offset, 1):
        for x in range(image_range_offset, width-image_range_offset, 1):
            flat_index = matrix_to_flat_index_rows(y, x, height)
            x_index = target_index_projections[0, flat_index]
            y_index = target_index_projections[1, flat_index]
            v[flat_index][0] = 0
            if not 0 < y_index < height or not 0 < x_index < width:
                valid_measurements[flat_index] = False
                continue
            # A newer SE3 estimate might re-validate a sample / pixel
            valid_measurements[flat_index] = True
            x_target = math.floor(x_index)
            y_target = math.floor(y_index)
            error = target_image[y_target, x_target] - reference_image[y, x]
            error_sq = error*error
            v[flat_index][0] = error_sq
            v_sum += error_sq

    #v_sum = v_sum*v_sum
    # If the estimate is so bad that all measurements are invalid
    if v_sum == 0:
        v_sum = -1000

    end = time.time()
    #print('Runtime for Compute Residual:', end-start)
    return v_sum


def gauss_newton_step(width, height, valid_measurements, J_pi, J_lie, target_image_grad_x, target_image_grad_y, v,
                      J_v_return, normal_matrix_return, image_range_offset):
    start = time.time()
    for y in range(image_range_offset, height - image_range_offset, 1):
        for x in range(image_range_offset, width - image_range_offset, 1):
            flat_index = matrix_to_flat_index_rows(y, x, height)
            if not valid_measurements[flat_index]:
                continue
            J_image = get_jacobian_image(target_image_grad_x, target_image_grad_y, x, y)
            J_pi_element = J_pi[flat_index]
            J_lie_element = J_lie[flat_index]

            J_pi_lie = np.matmul(J_pi_element, J_lie_element)
            J_full = np.matmul(J_image, J_pi_lie)
            J_t = np.transpose(J_full)
            error_vector = v[flat_index][0]
            J_v_return += np.multiply(error_vector, -J_t)
            normal_matrix_return += np.matmul(J_t, J_full)
    end = time.time()
    #print('Runtime Gauss Newton Step:', end-start)