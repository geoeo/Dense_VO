import numpy as np
from Numerics import Utils, SE3
from Camera import Intrinsic, Camera
from VisualOdometry import Frame, SolverThreadManager
from Benchmark import Parser, associate, ListGenerator, FileIO
from Visualization import Visualizer, PostProcessGroundTruth
from MotionModels import Ackermann,SteeringCommand


bench_path = '/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/rccar_15_11_18/'
dataset = 'marc_4_full/'
output_dir = 'output/'
rgb_folder = 'color/'
depth_folder = 'depth_large/'
ext = '.png'
### ds 4
#data_file = '299337.011086615_30_5e-11_0.25_0_False_True_False_False_120_1_False_False_False_rgb_depth_large_norm_depth_large_norm_z_neg_using_invalid_no_divide_steering_neg'
data_file = '299337.011086615_30_5e-11_0.25_0_False_True_False_True_120_1_False_True_False_rgb_depth_large_norm_depth_large_norm_z_neg_using_invalid_no_divide_steering_neg_steering_0.5'
#data_file = '299337.011086615_500_5e-11_0.25_0_False_True_False_True_120_1_False_True_False_rgb_depth_large_norm_depth_large_norm_z_neg_using_invalid_no_divide_steering_neg_save'
### ds 5
#data_file = '299475.190163022_50_5e-09_0.15_0_False_True_False_True_120_1_False_True_False_rgb_depth_large_norm_depth_large_norm_z_neg_using_invalid_no_divide_steering_neg_ack_corr_4'

data_ext = '.txt'

post_process_gt = None

#post_process_gt = PostProcessGroundTruth.PostProcessTUW_R300()
#post_process_gt = PostProcessGroundTruth.PostProcessTUW_R300_DS2()
post_process_gt = PostProcessGroundTruth.PostProcessTUW_R300_DS4()
#post_process_gt = PostProcessGroundTruth.PostProcessTUW_R300_DS5()


count = -1
start_count = 0
plot_vo = True

print(data_file)

dataset_root = bench_path + dataset
output_dir_path = dataset_root + output_dir

rgb_text = dataset_root +'rgb.txt'
depth_text = dataset_root +'depth_large_norm.txt'
match_text = dataset_root+'matches_with_duplicates_norm.txt'
rgb_encoder_text = dataset_root+'encoder_rgb.txt'

groundtruth_text = dataset_root+'groundtruth.txt'
#groundtruth_text = dataset_root+'groundtruth_opti.txt'
encoder_text = dataset_root+'encoder.txt'
data_file_path = output_dir_path+data_file+data_ext

match_dict = associate.read_file_list(match_text)
groundtruth_dict = associate.read_file_list(groundtruth_text)
rgb_encoder_dict = associate.read_file_list(rgb_encoder_text)
encoder_dict = associate.read_file_list(encoder_text)

rgb_folder_full = dataset_root+rgb_folder
depth_folder_full = dataset_root+depth_folder

rgb_files = ListGenerator.get_files_from_directory(rgb_folder_full, delimiter='.')
depth_files = ListGenerator.get_files_from_directory(depth_folder_full, delimiter='.')

rgb_file_total = len(rgb_files)
depth_file_total = len(depth_files)

image_groundtruth_dict = dict(associate.match(rgb_text, groundtruth_text,with_duplicates=True,max_difference=0.3))

plot_steering = True

parameters = data_file.split('_')

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


ground_truth_acc = np.identity(4,Utils.matrix_data_type)
#ground_truth_acc[0:3,0:3] = so3_prior
se3_estimate_acc = np.identity(4,Utils.matrix_data_type)
pose_estimate_list = []
ground_truth_list = []
ref_image_list = []
target_image_list = []
encoder_list = []
vo_twist_list = []
pose_estimate_list_loaded, encoder_list_loaded = FileIO.load_vo_from_file(data_file_path)


start = ListGenerator.get_index_of_id(start_idx,rgb_files)

ref_id_list, target_id_list, ref_files_failed_to_load = ListGenerator.generate_files_to_load_match(
    rgb_files,
    start=start,
    max_count=max_count,
    offset=offset,
    ground_truth_dict=image_groundtruth_dict,
    match_dict=match_dict,
    reverse=False)

dt_list = ListGenerator.generate_time_step_list(
    rgb_files,
    start=start,
    max_count=max_count,
    offset=offset)

if count == -1:
    count = len(ref_id_list)


for i in range(start_count, count):

    ref_id = ref_id_list[i]
    target_id = target_id_list[i]

    SE3_ref_target = Parser.generate_ground_truth_se3(groundtruth_dict,image_groundtruth_dict,ref_id,target_id,post_process_object=post_process_gt)
    im_greyscale_reference, im_depth_reference = Parser.generate_image_depth_pair_match(dataset_root,rgb_text,depth_text,match_text,ref_id)
    im_greyscale_target, im_depth_target = Parser.generate_image_depth_pair_match(dataset_root,rgb_text,depth_text,match_text, ref_id)

    ground_truth_acc = np.matmul(ground_truth_acc,SE3_ref_target)
    ground_truth_list.append(ground_truth_acc)

    ref_image_list.append((im_greyscale_reference, im_depth_reference))
    target_image_list.append((im_greyscale_target, im_depth_target))

    encoder_ts = float(rgb_encoder_dict[ref_id][0])
    encoder_values = encoder_dict[encoder_ts]
    encoder_values_float = [float(encoder_values[0]),float(encoder_values[1])]

    encoder_list.append(encoder_values_float)

    SE3_est = pose_estimate_list_loaded[i]

    se3_estimate_acc = np.matmul(se3_estimate_acc, SE3_est)
    pose_estimate_list.append(se3_estimate_acc)


delta = 30
if (count - 1) - start_count >= delta:
    print(SE3.rmse_avg_raw(ground_truth_list,pose_estimate_list, delta))


visualizer = Visualizer.Visualizer(ground_truth_list,plot_steering=plot_steering, title=None)
visualizer.visualize_ground_truth(clear=True,draw=False)
if plot_steering:
    visualizer.visualize_steering(encoder_list,clear=False,draw=False)
if plot_vo:
    visualizer.visualize_poses(pose_estimate_list, draw= False)
print('visualizing..')
visualizer.show()