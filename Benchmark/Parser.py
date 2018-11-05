import cv2
import numpy as np
from Numerics import SE3
from Benchmark import associate
from Numerics import ImageProcessing, Utils

'''list format is: tx, ty, tz, qx, qy, qz, qw'''
def generate_se3_from_groundtruth(groundtruth_list):
    tx = float(groundtruth_list[0])
    ty = float(groundtruth_list[1])
    tz = float(groundtruth_list[2])

    qx = float(groundtruth_list[3])
    qy = float(groundtruth_list[4])
    qz = float(groundtruth_list[5])
    qw = float(groundtruth_list[6])

    #qx *= -1
    #qy *= -1
    #qz *= -1
    #qw *= -1

    se3 = np.identity(4)

    roll, pitch, yaw = SE3.Quaternion_toEulerianRadians(qx, qy, qz, qw)
    #roll*=-1
    #pitch*=-1
    #yaw*=-1
    SO3 = SE3.makeS03(roll, pitch, yaw) #  seems to be more precise
    #SO3 = SE3.quaternion_to_s03(qx,qy,qz,qw)

    se3[0:3,0:3] = SO3[0:3,0:3]
    se3[0,3] = tx
    se3[1,3] = ty
    se3[2,3] = tz

    return se3

def generate_ground_truth_se3(ground_truth_file_path,image_groundtruth_dict, reference_id, target_id):
    groundtruth_ts_ref = image_groundtruth_dict[reference_id]
    groundtruth_data_ref = associate.return_groundtruth(ground_truth_file_path, groundtruth_ts_ref)
    SE3_ref = generate_se3_from_groundtruth(groundtruth_data_ref)

    groundtruth_ts_target = image_groundtruth_dict[target_id]
    groundtruth_data_target = associate.return_groundtruth(ground_truth_file_path, groundtruth_ts_target)
    SE3_target = generate_se3_from_groundtruth(groundtruth_data_target)

    SE3_ref_target = SE3.pose_pose_composition_inverse(SE3_ref, SE3_target)

    return SE3_ref_target

def generate_image_depth_pair(dataset_root, rgb_file_path, depth_file_path, match_text, image_id):
    rgb_ref_file_path, depth_ref_file_path = associate.return_rgb_depth_from_rgb_selection(rgb_file_path, depth_file_path,
                                                                                           match_text, dataset_root,
                                                                                           image_id)
    im_greyscale_reference = cv2.imread(rgb_ref_file_path, cv2.IMREAD_GRAYSCALE).astype(Utils.image_data_type)
    im_greyscale_reference = ImageProcessing.z_standardise(im_greyscale_reference)
    im_depth_reference = cv2.imread(depth_ref_file_path, cv2.IMREAD_ANYDEPTH).astype(Utils.depth_data_type_float)
    return im_greyscale_reference, im_depth_reference

# No match.txt means image and depth have same ts
def generate_image_depth_pair(dataset_root, rgb_folder, depth_folder, image_id, ext):

    image_id_str = f'{image_id:.9f}'
    rgb_ref_file_path = dataset_root + rgb_folder + image_id_str + ext
    depth_ref_file_path = dataset_root + depth_folder + image_id_str + ext

    im_greyscale_reference = cv2.imread(rgb_ref_file_path, cv2.IMREAD_GRAYSCALE).astype(Utils.image_data_type)
    im_greyscale_reference = ImageProcessing.z_standardise(im_greyscale_reference)
    im_depth_reference = cv2.imread(depth_ref_file_path, cv2.IMREAD_ANYDEPTH).astype(Utils.depth_data_type_float)
    return im_greyscale_reference, im_depth_reference




