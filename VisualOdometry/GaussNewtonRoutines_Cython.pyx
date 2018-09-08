import math
import numpy as np
import time

def hello_world():
    return "Hello World!"

#matrix_data_type = np.float32
matrix_data_type = np.cfloat

def matrix_to_flat_index_rows(y,x,rows):
    return rows*x+y

def get_jacobian_image(image_g_x,image_g_y,x,y):
    jacobian_image = np.zeros((1,2),dtype=matrix_data_type)
    jacobian_image[0,0] = image_g_x[y,x]
    jacobian_image[0,1] = image_g_y[y,x]
    return jacobian_image

def back_project_image(int width, int height, reference_camera, float[:, :] reference_depth_image, float[:, :] target_depth_image, X_back_projection,
                       int image_range_offset):


    # Py_ssize_t is the proper C type for Python array indices.
    cdef Py_ssize_t x, y

    start = time.time()
    for y in range(image_range_offset, height - image_range_offset, 1):
        for x in range(image_range_offset, width - image_range_offset, 1):
            flat_index = matrix_to_flat_index_rows(y, x, height)
            depth_ref = reference_depth_image[y, x]
            #depth_target = target_depth_image[y, x]
            if depth_ref == 0:
                depth_ref = 1000
            X = reference_camera.back_project_pixel(x, y, depth_ref)[:, 0]
            X_back_projection[0:3, flat_index] = X
    end = time.time()
    #print('Runtime for Back Project Image:', end-start)


def compute_residual(int width, int height, double[:,:] target_index_projections, valid_measurements, double[:,:] target_image, double[:,:] reference_image, float[:,:] v, int image_range_offset):
    cdef int v_sum = 0

    # Py_ssize_t is the proper C type for Python array indices.
    cdef Py_ssize_t x, y

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


# Doesnt work
def gauss_newton_step(width, height, valid_measurements, J_pi, J_lie, target_image_grad_x,target_image_grad_y, v,
                     J_v_return, normal_matrix_return, int image_range_offset):

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
            res = np.multiply(error_vector, -J_t)
            J_v_return = np.add(J_v_return,res)
            res_2 = np.matmul(J_t, J_full)
            normal_matrix_return = np.add(normal_matrix_return,res_2)
    end = time.time()
    #print('Runtime Gauss Newton Step:', end-start)
