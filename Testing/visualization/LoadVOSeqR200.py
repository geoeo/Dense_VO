import numpy as np
from Numerics import Utils, SE3
from Camera import Intrinsic, Camera
from VisualOdometry import Frame, SolverThreadManager
from Benchmark import Parser, associate, ListGenerator, FileIO
from Visualization import Visualizer
from MotionModels import Ackermann,SteeringCommand


bench_path = '/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/rccar_26_09_18/'
dataset = 'marc_4_full/'
output_dir = 'output/'
rgb_folder = 'color/'
depth_folder = 'depth_large/'
ext = '.png'
data_file = '967058.393566343_50_0.0005_0.0085_0_True_True_True_True_5_1'
data_ext = '.txt'

dataset_root = bench_path + dataset
output_dir_path = dataset_root + output_dir
rgb_text = dataset_root +'rgb.txt'
groundtruth_text = dataset_root+'groundtruth.txt'
rgb_encoder_text = dataset_root+'encoder_rgb.txt'
encoder_text = dataset_root+'encoder.txt'
data_file_path = output_dir_path+data_file+data_ext

groundtruth_dict = associate.read_file_list(groundtruth_text)
rgb_encoder_dict = associate.read_file_list(rgb_encoder_text)
encoder_dict = associate.read_file_list(encoder_text)

rgb_folder_full = dataset_root+rgb_folder
depth_folder_full = dataset_root+depth_folder

rgb_files = ListGenerator.get_files_from_directory(rgb_folder_full, delimiter='.')
depth_files = ListGenerator.get_files_from_directory(depth_folder_full, delimiter='.')

rgb_file_total = len(rgb_files)
depth_file_total = len(depth_files)

image_groundtruth_dict = dict(associate.match(rgb_text, groundtruth_text))

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
ground_truth_list = []
ref_image_list = []
target_image_list = []
encoder_list = []
vo_twist_list = []
pose_estimate_list_loaded, encoder_list_loaded = FileIO.load_vo_from_file(data_file_path)


start = ListGenerator.get_index_of_id(start_idx,rgb_files)

ref_id_list, target_id_list, ref_files_failed_to_load = ListGenerator.generate_files_to_load(
    rgb_files,
    start=start,
    max_count=max_count,
    offset=offset,
    ground_truth_dict=image_groundtruth_dict,
    reverse=False)

dt_list = ListGenerator.generate_time_step_list(
    rgb_files,
    start=start,
    max_count=max_count,
    offset=offset)

for i in range(0, len(ref_id_list)):

    ref_id = ref_id_list[i]
    target_id = target_id_list[i]

    SE3_ref_target = Parser.generate_ground_truth_se3(groundtruth_dict,image_groundtruth_dict,ref_id,target_id)
    im_greyscale_reference, im_depth_reference = Parser.generate_image_depth_pair(dataset_root,rgb_folder,depth_folder,ref_id, ext)
    im_greyscale_target, im_depth_target = Parser.generate_image_depth_pair(dataset_root,rgb_folder,depth_folder,target_id, ext)

    # TODO get this right
    # Optitrack/Rviz coversion capture X and Z are flipped
    rot = SE3.extract_rotation(SE3_ref_target)
    #conv = SE3.rotation_around_x(pi/2)
    #rot_new = np.matmul(conv,rot)
    euler = SE3.rotationMatrixToEulerAngles(rot)
    #rot_new = SE3.makeS03(euler[2],-euler[1],euler[0])
    rot_new = SE3.makeS03(euler[1],-euler[2],euler[0])
    SE3_ref_target[0:3,0:3] = rot_new
    x = SE3_ref_target[0,3]
    y = SE3_ref_target[1,3]
    z = SE3_ref_target[2,3]
    SE3_ref_target[0,3] = -y
    SE3_ref_target[1,3] = -z
    SE3_ref_target[2,3] = -x
    #SE3_ref_target[0:3,3] *= 10 # mm -> meters ?

    ground_truth_acc = np.matmul(ground_truth_acc,SE3_ref_target)
    ground_truth_list.append(ground_truth_acc)

    ref_image_list.append((im_greyscale_reference, im_depth_reference))
    target_image_list.append((im_greyscale_target, im_depth_target))

    encoder_ts = float(rgb_encoder_dict[ref_id][0])
    encoder_values = encoder_dict[encoder_ts]
    encoder_values_float = [float(encoder_values[0]),float(encoder_values[1])]

    encoder_list.append(encoder_values_float)



visualizer = Visualizer.Visualizer(ground_truth_list,plot_steering=plot_steering)
visualizer.visualize_ground_truth(clear=True,draw=False)
if plot_steering:
    visualizer.visualize_steering(encoder_list,clear=False,draw=False)
visualizer.visualize_poses(pose_estimate_list_loaded, draw= False)
visualizer.show()