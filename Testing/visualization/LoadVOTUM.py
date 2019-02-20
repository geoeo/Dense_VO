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
count = 5

post_process_gt = PostProcessGroundTruth.PostProcessTUM_F2() # f1_d2
#post_process_gt = PostProcessGroundTruth.PostProcessTUM_F1()

output_dir = 'output/'
rgb_folder = 'rgb/'
depth_folder = 'depth/'
ext = '.png'
data_file = '1311868164.363181114_50_5e-08_2.0_0_False_True_False_False_301_1_solver_2_other_res_2_z_stand_with_duplicates_kernel_1_res_no_flag_pitch_neg'
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

match_dict = associate.read_file_list(match_text)
groundtruth_dict = associate.read_file_list(groundtruth_text)

rgb_folder_full = dataset_root+rgb_folder
depth_folder_full = dataset_root+depth_folder

rgb_files = ListGenerator.get_files_from_directory(rgb_folder_full, delimiter='.')
depth_files = ListGenerator.get_files_from_directory(depth_folder_full, delimiter='.')

rgb_file_total = len(rgb_files)
depth_file_total = len(depth_files)

#image_groundtruth_dict = dict(associate.match(rgb_text, groundtruth_text,max_difference=0.2,with_duplicates=True))
image_groundtruth_dict = dict(associate.match(rgb_text, groundtruth_text)) #f1_d2


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


delta = 30
if (count - 1) - start_count >= delta:

    print(SE3.rmse_avg_raw(ground_truth_list,pose_estimate_list, delta))

visualizer = Visualizer.Visualizer(ground_truth_list,plot_steering=False)
visualizer.visualize_ground_truth(clear=True,draw=False)
if plot_vo:
    visualizer.visualize_poses(pose_estimate_list, draw= False)
print('visualizing..')
visualizer.show()