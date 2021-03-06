import numpy as np
from Numerics import Utils, SE3
from Benchmark import Parser, associate, ListGenerator, FileIO
from Visualization import Visualizer, PostProcessGroundTruth


bench_path = '/Volumes/Sandisk/Diplomarbeit/Results/TUW/'
dataset = 'dataset_3/'
#dataset = 'dataset_4/'
#dataset = 'dataset_5/'



start_count = 0
count = -1
post_process_gt = PostProcessGroundTruth.PostProcessTUW_R300_DS3()
#post_process_gt = PostProcessGroundTruth.PostProcessTUW_R300_DS4()
#post_process_gt = PostProcessGroundTruth.PostProcessTUW_R300_DS5()
plot_vo = True
plot_steering = False

output_dir = 'output/'
rgb_folder = 'color/'
depth_folder = 'depth_large_norm/'
ext = '.png'

# Dataset1
res_d1_1 = '299202.723105334_30_5e-13_1.0_0_False_True_False_True_300_1_False_True_False_rgb_depth_large_norm_depth_large_norm_sobel_1_eps_1_ack_01_only_steering'
res_d1_2 = '299202.723105334_30_5e-13_10.0_0_False_True_False_True_300_1_False_True_False_rgb_depth_large_norm_depth_large_norm_scharr_eps_001_ack_01' # filter
res_d1_3 = '299202.723105334_30_5e-15_10.0_0_False_True_False_False_300_1_False_False_False_rgb_depth_large_norm_depth_large_norm_scharr_eps_1' # filter
res_d1_4 = '299202.723105334_30_5e-13_1.0_0_False_True_False_True_300_1_False_True_False_rgb_depth_large_norm_depth_large_norm_sobel_1_eps_1_ack_01' # Ackerman
res_d1_5 = '299202.723105334_30_5e-11_10.0_0_False_True_False_True_300_1_False_True_False_rgb_depth_large_norm_depth_large_norm_scharr_eps_1_ack_05' # Ackerman
res_d1_6 = '299202.723105334_30_5e-15_10.0_0_False_True_True_False_300_1_False_True_False_rgb_depth_large_norm_depth_large_norm_scharr_eps_1' # Motion Prior

# Dataset 2
res_d2_1 = '299337.011086615_30_5e-10_10.0_0_False_True_False_True_300_1_False_True_False_rgb_depth_large_norm_depth_large_norm_scharr_1.0_only_steering'
res_d2_2 = '299337.011086615_30_5e-13_1.0_0_False_True_False_False_300_1_False_False_False_rgb_depth_large_norm_depth_large_norm_sobel_1_eps_1' # filter
res_d2_3 = '299337.011086615_30_5e-20_10.0_0_False_True_False_False_300_1_False_False_False_rgb_depth_large_norm_depth_large_norm_scharr_1.0' # filter
res_d2_4 = '299337.011086615_30_5e-13_10.0_0_False_True_False_True_300_1_False_True_False_rgb_depth_large_norm_depth_large_norm_scharr_eps_001_ack_01' # Ackerman
res_d2_5 = '299337.011086615_30_5e-10_10.0_0_False_True_False_True_300_1_False_True_False_rgb_depth_large_norm_depth_large_norm_scharr_0.5' # Ackerman
res_d2_6 = '299337.011086615_30_5e-13_10.0_0_False_True_True_False_300_1_False_True_False_rgb_depth_large_norm_depth_large_norm_scharr_eps_001_ack_01' # Motion Prior

# Dataset 3
res_d3_1 = '299489.237554490_30_5e-11_10.0_0_False_True_False_False_300_1_False_False_False_rgb_depth_large_norm_depth_large_norm_scharr_eps_1_ack_1.1_only_steering'
res_d3_2 = '299489.237554490_30_5e-11_10.0_0_False_True_False_False_300_1_False_False_False_rgb_depth_large_norm_depth_large_norm_scharr_eps_1' # filter
res_d3_3 = '299489.237554490_30_5e-13_1.0_0_False_True_False_False_300_1_False_False_False_rgb_depth_large_norm_depth_large_norm_sobel_1_eps_1' # filter
res_d3_4 = '299489.237554490_30_5e-13_10.0_0_False_True_False_True_300_1_False_True_False_rgb_depth_large_norm_depth_large_norm_scharr_eps_001_ack_01'
res_d3_5 = '299489.237554490_30_5e-13_10.0_0_False_True_False_True_300_1_False_True_False_rgb_depth_large_norm_depth_large_norm_scharr_eps_1_ack_05'
res_d3_6 = '299489.237554490_30_5e-15_10.0_0_False_True_True_False_300_1_False_True_False_rgb_depth_large_norm_depth_large_norm_scharr_eps_1' # Motion Prior


label_1 = 'ackermann'
label_2 = 'scharr 10'
label_3 = 'ack 0.1'
label_4 = 'ack 0.5'
label_steering = 'control in'

data_file = res_d1_1
data_file_2 = ''
data_file_2 = res_d1_3
data_file_3 = ''
data_file_3 = res_d1_4
data_file_4 = ''
data_file_4 = res_d1_5

data_ext = '.txt'

print("1: " + data_file)
if data_file_2:
    print("2: " + data_file_2)
if data_file_3:
    print("3: " + data_file_3)
if data_file_4:
    print("4: " + data_file_4)

dataset_root = bench_path + dataset
output_dir_path = dataset_root + output_dir
rgb_text = dataset_root +'rgb.txt'
depth_text = dataset_root +'depth_large_norm.txt'
#match_text = dataset_root+'matches.txt'
match_text = dataset_root+'matches_with_duplicates_norm.txt'
groundtruth_text = dataset_root+'groundtruth.txt'
encoder_text = dataset_root+'encoder.txt'
rgb_encoder_text = dataset_root+'encoder_rgb.txt'

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

image_groundtruth_dict = dict(associate.match(rgb_text, groundtruth_text,with_duplicates=True,max_difference=0.3))
rgb_encoder_dict = associate.read_file_list(rgb_encoder_text)
encoder_dict = associate.read_file_list(encoder_text)


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
se3_estimate_acc_2 = np.identity(4,Utils.matrix_data_type)
se3_estimate_acc_3 = np.identity(4,Utils.matrix_data_type)
se3_estimate_acc_4 = np.identity(4,Utils.matrix_data_type)
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

    SE3_ref_target = Parser.generate_ground_truth_se3(groundtruth_dict,image_groundtruth_dict,ref_id,target_id,post_process_object=post_process_gt)
    SE3_ref_target_clean = Parser.generate_ground_truth_se3(groundtruth_dict,image_groundtruth_dict,ref_id,target_id,post_process_object=None)
    im_greyscale_reference, im_depth_reference = Parser.generate_image_depth_pair_match(dataset_root,rgb_text,depth_text,match_text,ref_id)
    im_greyscale_target, im_depth_target = Parser.generate_image_depth_pair_match(dataset_root,rgb_text,depth_text,match_text, ref_id)

    #post_process_gt.post_process_in_mem(SE3_ref_target)

    ground_truth_acc = np.matmul(ground_truth_acc, SE3_ref_target)
    ground_truth_acc[0,3] = -SE3_ref_target_clean[0,3] # ds3
    #ground_truth_acc[1,3] = SE3_ref_target_clean[1,3]
    ground_truth_list.append(ground_truth_acc)

    ref_image_list.append((im_greyscale_reference, im_depth_reference))
    target_image_list.append((im_greyscale_target, im_depth_target))


    SE3_est = pose_estimate_list_loaded[i]
    #SE3.post_process_pose_for_display_in_mem(se3_estimate_acc)
    se3_estimate_acc = np.matmul(se3_estimate_acc, SE3_est)
    pose_estimate_list.append(se3_estimate_acc)

    encoder_ts = float(rgb_encoder_dict[ref_id][0])
    encoder_values = encoder_dict[encoder_ts]
    encoder_values_float = [float(encoder_values[0]),float(encoder_values[1])]
    encoder_list.append(encoder_values_float)


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
    print("1: " + str(SE3.rmse_avg_raw(ground_truth_list,pose_estimate_list, delta)))
    if data_file_2:
        print("2: " + str(SE3.rmse_avg_raw(ground_truth_list, pose_estimate_list_2, delta)))
    if data_file_3:
        print("3: " + str(SE3.rmse_avg_raw(ground_truth_list, pose_estimate_list_3, delta)))
    if data_file_4:
        print("4: " + str(SE3.rmse_avg_raw(ground_truth_list, pose_estimate_list_4, delta)))

handles = []

visualizer = Visualizer.Visualizer(ground_truth_list,plot_steering=plot_steering, plot_trajectory=False, plot_rmse=False)
visualizer.visualize_ground_truth(clear=True,draw=False)


if plot_vo:
    patch_0 = Visualizer.make_patch(color='green', label='gt')
    handles.append(patch_0)
    patch_1 = Visualizer.make_patch(color='red', label=label_1)
    handles.append(patch_1)
    visualizer.visualize_poses(pose_estimate_list, draw= False, style='-rx')
    if data_file_2:
        patch_2 = Visualizer.make_patch(color='blue', label=label_2)
        handles.append(patch_2)
        visualizer.visualize_poses(pose_estimate_list_2, draw= False, style='-bx')
    if data_file_3:
        patch_3 = Visualizer.make_patch(color='magenta', label=label_3)
        handles.append(patch_3)
        visualizer.visualize_poses(pose_estimate_list_3, draw= False, style='-mx')
    if data_file_4:
        patch_4 = Visualizer.make_patch(color='cyan', label=label_4)
        handles.append(patch_4)
        visualizer.visualize_poses(pose_estimate_list_4, draw= False, style='-cx')
print('visualizing..')
if plot_steering:
    patch_5 = Visualizer.make_patch(color='yellow', label=label_steering)
    handles.append(patch_5)
    visualizer.visualize_steering(encoder_list,clear=False,draw=False)

anchor = (1.25, 1.02, 0., .102)
visualizer.legend(handles, anchor)
visualizer.show()