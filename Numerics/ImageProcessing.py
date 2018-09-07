import math
import numpy as np
from Numerics.Utils import matrix_to_flat_index_rows
from Numerics.JacobianGenerator import get_jacobian_image

def z_standardise(pixels):
    mean = np.mean(pixels)
    std_dev = np.std(pixels)

    return (pixels - mean) / std_dev


def normalize_to_image_space(pixels):
    pixels_min = np.min(pixels)
    pixels_max = np.max(pixels)

    # https://en.wikipedia.org/wiki/Normalization_(image_processing)
    mapped_float = (pixels - pixels_min) * (255.0 / (pixels_max - pixels_min))
    return mapped_float.astype(np.uint8)


def is_row_valid(r,matrix_height):
    return 0 <= r < matrix_height


def is_column_valid(c,matrix_width):
    return 0 <= c < matrix_width


def back_project_image(width, height, reference_camera, reference_depth_image, target_depth_image, X_back_projection,
                       valid_measurements_reference, valid_measurements_target):
    for y in range(0, height, 1):
        for x in range(0, width, 1):
            flat_index = matrix_to_flat_index_rows(y, x, height)
            depth_ref = reference_depth_image[y, x]
            depth_target = target_depth_image[y, x]
            X_back_projection[0:3, flat_index] = reference_camera.back_project_pixel(x, y, depth_ref)[:, 0]
            if depth_ref != 0:
                valid_measurements_reference[flat_index] = True
            if depth_target != 0:
                valid_measurements_target[flat_index] = True


def compute_residual(width, height, target_index_projections, valid_measurements, target_image, reference_image,v):
    v_sum = 0
    for y in range(0, height, 1):
        for x in range(0, width, 1):
            flat_index = matrix_to_flat_index_rows(y, x, height)
            x_index = target_index_projections[0, flat_index]
            y_index = target_index_projections[1, flat_index]
            v[flat_index][0] = 0
            if not valid_measurements[flat_index]:
                continue

            if not 0 < y_index < height or not 0 < x_index < width:
                valid_measurements[flat_index] = False
                continue
            # A newer SE3 estimate might re-validate a sample / pixel
            valid_measurements[flat_index] = True
            x_target = math.floor(x_index)
            y_target = math.floor(y_index)
            error = target_image[y_target, x_target] - reference_image[y, x]
            v[flat_index][0] = error
            v_sum += error

    return v_sum


def gauss_newton_step(width, height, valid_measurements, J_pi, J_lie, target_image_grad_x, target_image_grad_y, v,
                      J_v_return, normal_matrix_return):
    for y in range(0, height, 1):
        for x in range(0, width, 1):
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
            J_v_return += np.multiply(error_vector, J_t)
            normal_matrix_return += np.matmul(J_t, J_full)
