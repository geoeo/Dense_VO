import numpy as np
from Numerics import Utils, SE3
from Camera import Intrinsic, Camera
from VisualOdometry import Frame, SolverThreadManager
from Benchmark import Parser, associate, ListGenerator, FileIO
from Visualization import Visualizer, PostProcessGroundTruth
from MotionModels import Ackermann,SteeringCommand


bench_path = '/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Bench/'
dataset = 'rgbd_dataset_freiburg2_desk/'
#dataset = 'rgbd_dataset_freiburg1_desk/'
#dataset = 'rgbd_dataset_freiburg1_desk2/'
#dataset = 'rgbd_dataset_freiburg1_xyz/'

start_count = 0
count = -1
post_process_gt = PostProcessGroundTruth.PostProcessTUM_F2() # f1_d2
#post_process_gt = PostProcessGroundTruth.PostProcessTUM_F1()

output_dir = 'output/'
rgb_folder = 'rgb/'
depth_folder = 'depth/'
ext = '.png'

data_file_2 = None
data_file_3 = None
data_file_4 = None


#d2f1

#data_file = '1305031536.739722013_30_0.0005_3.5_0_False_True_False_False_301_1_other_solver_1_z_neg_no_res_flag'
#data_file = '1305031536.739722013_30_0.0005_30.0_0_False_True_False_False_120_1_other_solver_1_z_neg_no_res_flag_scharr'
##
#data_file = '1311868250.648756981_30_5e-05_2.5_0_False_True_False_False_301_1_solver_2_valid_other_res_2_z_stand_with_duplicates_kernel_1_pitch_neg_new_jacobian_no_update_using_invalid_res_no_flag'
#data_file = '1311868250.648756981_30_5e-05_25.0_0_False_True_False_False_301_1_solver_2_valid_other_res_2_z_stand_with_duplicates_scharr_pitch_neg_new_jacobian_no_update_using_invalid_res_no_flag'
#data_file = '1311868250.648756981_30_5e-05_25.0_0_False_True_False_False_301_1_solver_2_valid_other_res_2_z_stand_with_duplicates_scharr_pitch_neg_new_jacobian_no_update_using_invalid_res_no_flag'
#data_file = '1311868250.648756981_30_5e-05_10.0_0_False_True_False_False_301_1_solver_2_valid_other_res_2_z_stand_with_duplicates_scharr_pitch_neg_new_jacobian_no_update_using_invalid_res_no_flag_new_W'

#data_file = '1311868164.363181114_30_5e-05_2.5_0_False_True_False_False_301_1_solver_2_valid_other_res_2_z_stand_with_duplicates_kernel_1_pitch_neg_new_jacobian_no_update_using_invalid_res_no_flag'
#data_file = '1311868164.363181114__30_5e-05_10.0_0_False_True_False_False_120_1_solver_2_valid_other_res_2_z_stand_with_duplicates_kernel_3_pitch_neg_new_jacobian_no_update_using_invalid_res_no_flag'
#data_file = '1311868164.363181114_30_5e-05_25.0_0_False_True_False_False_301_1_solver_2_valid_other_res_2_z_stand_with_duplicates_scharr_pitch_neg_new_jacobian_no_update_using_invalid_res_no_flag'
#data_file = '1311868164.363181114_30_5e-05_2.5_0_False_True_False_False_120_1_solver_2_valid_other_res_2_z_stand_with_duplicates_kernel_3_pitch_neg_new_jacobian_no_update_using_invalid_res_no_flag'
#data_file = '1311868164.363181114_30_5e-05_10.0_0_False_True_False_False_120_1_solver_2_valid_other_res_2_z_stand_with_duplicates_kernel_3_pitch_neg_new_jacobian_no_update_using_invalid_res_no_flag'
#data_file = '1311868164.363181114_30_5e-05_10.0_0_False_True_False_False_301_1_solver_2_valid_other_res_2_z_stand_with_duplicates_scharr_pitch_neg_new_jacobian_no_update_using_invalid_res_no_flag_new_W'
#data_file = '1311868164.363181114_30_5e-05_1.0_0_False_True_False_False_180_1_solver_2_valid_other_res_2_z_stand_with_duplicates_kernel_1_pitch_neg_new_jacobian_no_update_using_invalid_res_no_flag_new_W'

#data_file = '1311868164.363181114_30_5e-07_10_0_False_True_False_False_10_1_scharr'
#data_file = '1311868164.363181114_30_5e-07_10_0_False_True_False_False_301_1_scharr'
#data_file = '1311868164.363181114_30_5e-07_10_0_False_True_False_False_300_1_scharr_res_ee'
#data_file = '1311868164.363181114_30_5e-07_1_0_False_True_False_False_300_1_sobel_1_res_ee'
#data_file = '1311868164.363181114_30_5e-07_1_0_False_True_False_False_300_1_sobel_1'
#data_file = '1311868164.363181114_30_5e-07_1_0_False_True_True_False_300_1_sobel_1'
#data_file = '1311868174.699578047_30_5e-07_10_0_False_True_False_False_180_1_scharr'
#data_file = '1311868174.699578047_30_5e-07_1_0_False_True_False_False_180_1_sobel_1'
#data_file = '1311868174.699578047_30_5e-07_1_0_False_True_False_False_180_1_sobel_1_eps_001'
#data_file = '1311868174.699578047_30_5e-07_1_0_False_True_False_False_180_1_sobel_1_eps_1'
#data_file = '1311868174.699578047_30_5e-07_1_0_False_True_True_False_180_1_sobel_1'


data_file = '1311868250.648756981_30_5e-07_1_0_False_True_False_False_300_1_sobel_1'
#data_file = '1311868250.648756981_30_5e-07_10_0_False_True_False_False_300_1_scharr'
#data_file = '1311868250.648756981_30_5e-07_1_0_False_True_True_False_300_1_sobel_1_eps_001'

#data_file = '1311868164.363181114_30_5e-05_0.1_0_True_True_False_False_10_1_solver_2_valid_other_res_2_z_stand_with_duplicates_scharr_pitch_neg_new_jacobian_no_update_using_invalid_res_no_flag_new_W'

#data_file = '1311868164.363181114_30_5e-05_10_0_True_True_True_False_80_1_scharr_new_W_only_steering'
#data_file = '1311868164.363181114_30_5e-05_10_0_True_True_True_False_20_1_scharr_new_W_only_steering'
#data_file = '1311868164.363181114_30_5e-05_1_0_True_True_True_False_10_1_scharr_new_W_motion_1.0_only_steering'
#data_file = '1311868164.363181114_30_5e-05_10_0_True_True_True_False_300_1_scharr_new_W_motion_1.0_only_steering'


#data_file = '1311868174.699578047_30_5e-05_2.5_0_False_True_False_False_180_1_solver_2_valid_other_res_2_z_stand_with_duplicates_kernel_1_pitch_neg_new_jacobian_no_update_using_invalid_res_no_flag'
#data_file = '1311868174.699578047_30_5e-05_45.0_0_False_True_False_False_180_1_solver_2_valid_other_res_2_z_stand_with_duplicates_scharr_pitch_neg_new_jacobian_no_update_using_invalid_res_no_flag'
#data_file = '1311868174.699578047_30_5e-05_25.0_0_False_True_False_False_180_1_solver_2_valid_other_res_2_z_stand_with_duplicates_scharr_pitch_neg_new_jacobian_no_update_using_invalid_res_no_flag'
#data_file = '1311868174.699578047_30_0.005_25.0_0_False_True_False_False_10_1_solver_2_valid_other_res_2_z_stand_with_duplicates_scharr_pitch_neg_new_jacobian_no_update_using_invalid_res_no_flag_new_W'
#data_file = '1311868174.699578047_30_0.005_25.0_0_False_True_False_False_180_1_solver_2_valid_other_res_2_z_stand_with_duplicates_scharr_pitch_neg_new_jacobian_no_update_using_invalid_res_no_flag_new_W'
#data_file = '1311868174.699578047_30_5e-05_10.0_0_False_True_False_False_180_1_solver_2_valid_other_res_2_z_stand_with_duplicates_scharr_pitch_neg_new_jacobian_no_update_using_invalid_res_no_flag_new_W'
#data_file = '1311868174.699578047_30_5e-05_1.0_0_False_True_False_False_180_1_solver_2_valid_other_res_2_z_stand_with_duplicates_kernel_1_pitch_neg_new_jacobian_no_update_using_invalid_res_no_flag_new_W'

# f1d2

#data_file = '1305031536.739722013_30_0.0005_25.0_0_False_True_False_False_120_1_other_solver_1_z_neg_no_res_flag_scharr'
#data_file = '1305031536.739722013_30_5e-05_10.0_0_False_True_False_False_120_1_other_solver_1_z_neg_no_res_flag_scharr_new_W'
data_ext = '.txt'

print(data_file)

plot_vo = True

dataset_root = bench_path + dataset
output_dir_path = dataset_root + output_dir
rgb_text = dataset_root +'rgb.txt'
depth_text = dataset_root +'depth.txt'
#match_text = dataset_root+'matches.txt'
match_text = dataset_root+'matches_with_duplicates.txt'
groundtruth_text = dataset_root+'groundtruth.txt'

data_file_path = output_dir_path+data_file+data_ext
if data_file_2:
    data_file_path_2 = output_dir_path+data_file_2+data_ext
if data_file_3:
    data_file_path_3 = output_dir_path+data_file_3+data_ext
if data_file_4:
    data_file_path_4 = output_dir_path+data_file_4+data_ext

match_dict = associate.read_file_list(match_text)
groundtruth_dict = associate.read_file_list(groundtruth_text)

rgb_folder_full = dataset_root+rgb_folder
depth_folder_full = dataset_root+depth_folder

rgb_files = ListGenerator.get_files_from_directory(rgb_folder_full, delimiter='.')
depth_files = ListGenerator.get_files_from_directory(depth_folder_full, delimiter='.')

rgb_file_total = len(rgb_files)
depth_file_total = len(depth_files)

image_groundtruth_dict = dict(associate.match(rgb_text, groundtruth_text,max_difference=0.2,with_duplicates=True))
#image_groundtruth_dict = dict(associate.match(rgb_text, groundtruth_text)) #f1_d2


parameters = data_file.split('_')
if data_file_2:
    parameters_2 = data_file_2.split('_')
if data_file_3:
    parameters_3 = data_file_3.split('_')
if data_file_4:
    parameters_4 = data_file_4.split('_')


start_idx = float(parameters[0])
max_count = int(parameters[9])
offset = int(parameters[10])

max_its = int(parameters[1])
eps = float(parameters[2])
alpha_step = float(parameters[3])
image_range_offset_start = bool(parameters[5])
use_robust = bool(parameters[6])
use_motion_prior = bool(parameters[7])
use_ackermann = bool(parameters[8])

if data_file_2:
    start_idx_2 = float(parameters_2[0])
    max_count_2 = int(parameters_2[9])
    offset_2 = int(parameters_2[10])

if data_file_3:
    start_idx_3 = float(parameters_3[0])
    max_count_3 = int(parameters_3[9])
    offset_3 = int(parameters_3[10])

if data_file_4:
    start_idx_4 = float(parameters_4[0])
    max_count_4 = int(parameters_4[9])
    offset_4 = int(parameters_4[10])


ground_truth_acc = np.identity(4,Utils.matrix_data_type)
#ground_truth_acc[0:3,0:3] = so3_prior
se3_estimate_acc = np.identity(4,Utils.matrix_data_type)
pose_estimate_list = []
pose_estimate_list_2 = []
pose_estimate_list_3 = []
pose_estimate_list_4 = []
ground_truth_list = []
ref_image_list = []
target_image_list = []
encoder_list = []
vo_twist_list = []
pose_estimate_list_loaded, encoder_list_loaded = FileIO.load_vo_from_file(data_file_path)
if data_file_2:
    pose_estimate_list_loaded_2, encoder_list_loaded_2 = FileIO.load_vo_from_file(data_file_path_2)
if data_file_3:
    pose_estimate_list_loaded_3, encoder_list_loaded_3 = FileIO.load_vo_from_file(data_file_path_3)
if data_file_4:
    pose_estimate_list_loaded_4, encoder_list_loaded_4 = FileIO.load_vo_from_file(data_file_path_4)



start = ListGenerator.get_index_of_id(start_idx,rgb_files)
ref_id_list, target_id_list, ref_files_failed_to_load = ListGenerator.generate_files_to_load_match(
    rgb_files,
    start=start,
    max_count=max_count,
    offset=offset,
    ground_truth_dict=image_groundtruth_dict,
    match_dict=match_dict,
    reverse=False)

if data_file_2:
    start_2 = ListGenerator.get_index_of_id(start_idx_2, rgb_files)
    ref_id_list_2, target_id_list_2, ref_files_failed_to_load_2 = ListGenerator.generate_files_to_load_match(
        rgb_files,
        start=start_2,
        max_count=max_count_2,
        offset=offset_2,
        ground_truth_dict=image_groundtruth_dict,
        match_dict=match_dict,
        reverse=False)

if data_file_3:
    start_3 = ListGenerator.get_index_of_id(start_idx_3, rgb_files)
    ref_id_list_3, target_id_list_3, ref_files_failed_to_load_3 = ListGenerator.generate_files_to_load_match(
        rgb_files,
        start=start_3,
        max_count=max_count_3,
        offset=offset_3,
        ground_truth_dict=image_groundtruth_dict,
        match_dict=match_dict,
        reverse=False)

if data_file_4:
    start_4 = ListGenerator.get_index_of_id(start_idx_4, rgb_files)
    ref_id_list_4, target_id_list_4, ref_files_failed_to_load_4 = ListGenerator.generate_files_to_load_match(
        rgb_files,
        start=start_4,
        max_count=max_count_4,
        offset=offset_4,
        ground_truth_dict=image_groundtruth_dict,
        match_dict=match_dict,
        reverse=False)

dt_list = ListGenerator.generate_time_step_list(
    rgb_files,
    start=start,
    max_count=max_count,
    offset=offset)

ref_list_len = len(ref_id_list)

if count == -1:
    count = ref_list_len
#pose_estimate_list_loaded_len = len(pose_estimate_list_loaded)

for i in range(start_count, count):

    ref_id = ref_id_list[i]
    target_id = target_id_list[i]

    SE3_ref_target = Parser.generate_ground_truth_se3(groundtruth_dict,image_groundtruth_dict,ref_id,target_id,post_process_object=None)
    im_greyscale_reference, im_depth_reference = Parser.generate_image_depth_pair_match(dataset_root,rgb_text,depth_text,match_text,ref_id)
    im_greyscale_target, im_depth_target = Parser.generate_image_depth_pair_match(dataset_root,rgb_text,depth_text,match_text, ref_id)

    post_process_gt.post_process_in_mem(SE3_ref_target)

    ground_truth_acc = np.matmul(ground_truth_acc,SE3_ref_target)
    ground_truth_list.append(ground_truth_acc)

    ref_image_list.append((im_greyscale_reference, im_depth_reference))
    target_image_list.append((im_greyscale_target, im_depth_target))


    SE3_est = pose_estimate_list_loaded[i]
    #SE3.post_process_pose_for_display_in_mem(se3_estimate_acc)
    se3_estimate_acc = np.matmul(se3_estimate_acc, SE3_est)
    pose_estimate_list.append(se3_estimate_acc)


    if data_file_2:
        SE3_est_2 = pose_estimate_list_loaded_2[i]
        se3_estimate_acc_2 = np.matmul(se3_estimate_acc_2, SE3_est_2)
        pose_estimate_list_2.append(se3_estimate_acc_2)

    if data_file_3:
        SE3_est_3 = pose_estimate_list_loaded_3[i]
        se3_estimate_acc_3 = np.matmul(se3_estimate_acc_3, SE3_est_3)
        pose_estimate_list_3.append(se3_estimate_acc_3)

    if data_file_4:
        SE3_est_4 = pose_estimate_list_loaded_4[i]
        se3_estimate_acc_4 = np.matmul(se3_estimate_acc_4, SE3_est_4)
        pose_estimate_list_4.append(se3_estimate_acc_4)


delta = 30
if (count - 1) - start_count >= delta:

    print(SE3.rmse_avg_raw(ground_truth_list,pose_estimate_list, delta))

visualizer = Visualizer.Visualizer(ground_truth_list,plot_steering=False, plot_trajectory=False)
visualizer.visualize_ground_truth(clear=True,draw=False)
if plot_vo:
    visualizer.visualize_poses(pose_estimate_list, draw= False)
    if data_file_2:
        visualizer.visualize_poses(pose_estimate_list_2, draw= False, style='-ro')
    if data_file_3:
        visualizer.visualize_poses(pose_estimate_list_3, draw= False, style='-bx')
    if data_file_3:
        visualizer.visualize_poses(pose_estimate_list_4, draw= False, style='-bo')
print('visualizing..')
visualizer.show()