import numpy as np
from Numerics import Utils, SE3
from Camera import Intrinsic, Camera
from VisualOdometry import Frame, SolverThreadManager
from Benchmark import Parser, associate, ListGenerator, FileIO
from Visualization import Visualizer
from MotionModels import Ackermann,SteeringCommand
from Visualization import PostProcessGroundTruth

#dataset 1
#start_idx = 298516.028956271

#data 2
#start_idx = 298679.112609803
#start_idx = 298698.117996584
#start_idx = 298698.350589178 # rect
start_idx = 298703.302012647 # rect

# dataset 3 # prior no gt
#start_idx = 299206.609285928
#start_idx = 299208.004564834

#dataset4
#start_idx = 299336.878361490
#start_idx = 299338.172738053
#start_idx = 299341.094225178




bench_path = '/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/rccar_15_11_18/'
dataset = 'marc_2_full/'
output_dir = 'output/'

rgb_folder = 'color_rect/'
depth_folder = 'depth_large_rect/'
rgb_match = 'rgb_rect'
depth_match = 'depth_large_rect'
match_match = 'matches_rect'
encoder_match = 'encoder_rgb_rect'

ext = '.png'
data_ext = '.txt'

dataset_root = bench_path + dataset
output_dir_path = dataset_root + output_dir

rgb_text = dataset_root +rgb_match + data_ext
depth_text = dataset_root +depth_match + data_ext
match_text = dataset_root+match_match + data_ext
rgb_encoder_text = dataset_root+encoder_match + data_ext

groundtruth_text = dataset_root+'groundtruth' + data_ext
encoder_text = dataset_root+'encoder' + data_ext


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

image_groundtruth_dict = dict(associate.match(rgb_text, groundtruth_text))

depth_factor = 5000.0
#depth_factor = 1.0
use_ndc = True
calc_vo = True
plot_steering = True

max_count = 10
offset = 1

name = f"{start_idx:.9f}"

max_its = 200
eps = 0.0001  # 0.0005, 0.0001, 0.0057
alpha_step = 0.005  # 0.002 ds3, 0.0055, 0.0085 - motion pri 0.01
gradient_monitoring_window_start = 1
image_range_offset_start = 0
use_ndc = use_ndc
use_robust = True
track_pose_estimates = True
use_motion_prior = False
use_ackermann = False
debug = False

use_paper_cov = False
use_ackermann_cov = False
use_paper_ackermann_cov = False

if use_motion_prior:
    assert (use_paper_cov or use_ackermann_cov or use_paper_ackermann_cov)

additional_info = f"{use_paper_cov}" + '_' + f"{use_ackermann_cov}" + '_' + f"{use_paper_ackermann_cov}"
additional_info += '_' + 'all_norm' + '_' + rgb_match + '_' + depth_match

info = '_' + f"{max_its}" \
       + '_' + f"{eps}" \
       + '_' + f"{alpha_step}" \
       + '_' + f"{image_range_offset_start}" \
       + '_' + f"{use_ndc}" \
       + '_' + f"{use_robust}" \
       + '_' + f"{use_motion_prior}" \
       + '_' + f"{use_ackermann}" \
       + '_' + f"{max_count}" \
       + '_' + f"{offset}"

if additional_info:
    info += '_' + additional_info

ground_truth_acc = np.identity(4,Utils.matrix_data_type)
#ground_truth_acc[0:3,0:3] = so3_prior
se3_estimate_acc = np.identity(4,Utils.matrix_data_type)
ground_truth_list = []
pose_estimate_list = []
ref_image_list = []
target_image_list = []
encoder_list = []
vo_twist_list = []

post_process_gt = PostProcessGroundTruth.PostProcessTUW()

start = ListGenerator.get_index_of_id(start_idx,rgb_files)

ref_id_list, target_id_list, ref_files_failed_to_load = ListGenerator.generate_files_to_load_match(
    rgb_files,
    start=start,
    max_count=max_count,
    offset=offset,
    ground_truth_dict=image_groundtruth_dict,
    match_dict = match_dict,
    reverse=False)

dt_list = ListGenerator.generate_time_step_list(
    rgb_files,
    start=start,
    max_count=max_count,
    offset=offset)


for i in range(0, len(ref_id_list)):

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



im_greyscale_reference_1, im_depth_reference_1 = ref_image_list[0]
(image_height, image_width) = im_greyscale_reference_1.shape
se3_identity = np.identity(4, dtype=Utils.matrix_data_type)
# image gradient induces a coordiante system where y is flipped i.e have to flip it here

intrinsic_identity = Intrinsic.Intrinsic(619.225, 618.836, 317.603, 244.876)
if use_ndc:
    #intrinsic_identity = Intrinsic.Intrinsic(1, 1, 1/2, 1/2) # for ndc
    intrinsic_identity = Intrinsic.Intrinsic(1, 619.225/618.836, 317.603/image_width, 244.876/image_height) # for ndc


camera_reference = Camera.Camera(intrinsic_identity, se3_identity)
camera_target = Camera.Camera(intrinsic_identity, se3_identity)

steering_commands = list(map(lambda cmd: SteeringCommand.SteeringCommands(cmd[0],cmd[1]), encoder_list))
ackermann_motion = Ackermann.Ackermann(steering_commands, dt_list)
ackermann_cov_list = ackermann_motion.covariance_dead_reckoning_for_command_list(steering_commands,dt_list)

motion_cov_inv = np.identity(6,dtype=Utils.matrix_data_type)
#motion_cov_inv = np.zeros((6,6),dtype=Utils.matrix_data_type)
#for i in range(0,6):
#    motion_cov_inv[i,i] = Utils.covariance_zero
twist_prior = np.zeros((6,1),dtype=Utils.matrix_data_type)

#TODO plot ackerman pose against prediction to test dt
for i in range(0, len(ref_image_list)):
    im_greyscale_reference, im_depth_reference = ref_image_list[i]
    im_greyscale_target, im_depth_target = target_image_list[i]

    im_depth_reference /= depth_factor
    im_depth_target /= depth_factor

    max_depth = np.amax(im_depth_reference)

    # We only need the gradients of the target frame
    frame_reference = Frame.Frame(im_greyscale_reference, im_depth_reference, camera_reference, False)
    frame_target = Frame.Frame(im_greyscale_target, im_depth_target, camera_target, True)

    ackermann_cov = ackermann_cov_list[i]
    ackermann_cov_large = Ackermann.generate_6DOF_cov_from_motion_model_cov(ackermann_cov)
    ackermann_cov_large_inv = np.linalg.inv(ackermann_cov_large)
    ackermann_twist = ackermann_motion.motion_delta_list[i].get_6dof_twist(normalize=False)
    #ackermann_twist[4] *= -1

    # OWN with motion prior = False
    #motion_cov_inv = ackermann_cov_large_inv
    #motion_cov_inv = Utils.norm_covariance_row(motion_cov_inv)
    #twist_prior = ackermann_twist

    solver_manager = SolverThreadManager.Manager(1,
                                                 "Solver Manager",
                                                 frame_reference,
                                                 frame_target,
                                                 max_its=max_its,
                                                 eps=eps,  #0.0008, 0.0001, 0.0057
                                                 alpha_step=alpha_step,  # 0.002 ds3, 0.0055, 0.0085 - motion pri 0.01
                                                 gradient_monitoring_window_start=1,
                                                 image_range_offset_start=image_range_offset_start,
                                                 max_depth=max_depth,
                                                 twist_prior=twist_prior,
                                                 motion_cov_inv = motion_cov_inv,
                                                 use_ndc=use_ndc,
                                                 use_robust=use_robust,
                                                 track_pose_estimates=track_pose_estimates,
                                                 use_motion_prior=use_motion_prior,
                                                 ackermann_pose_prior=ackermann_twist,
                                                 use_ackermann=use_ackermann,
                                                 debug=False)

    if calc_vo:
        solver_manager.start()
        solver_manager.join()  # wait to complete
        print('iteration ', i+1, ' complete')

        # PAPER
        if use_paper_cov:
            motion_cov_inv = solver_manager.motion_cov_inv_final
        # ACKERMANN
        elif use_ackermann_cov:
            motion_cov_inv = ackermann_cov_large_inv
        else:
            motion_cov_inv = solver_manager.motion_cov_inv_final
            motion_cov_inv[2,:] = ackermann_cov_large_inv[2,:]

        twist_prior = np.multiply(1.0,solver_manager.twist_final)



    #twist_prior = ackermann_twist

    #twist_prior = np.add(twist_prior,solver_manager.twist_final)
    #se3_estimate_acc = np.matmul(solver_manager.SE3_est_final,se3_estimate_acc)

 #  SE3_est = SE3.twist_to_SE3(ackermann_twist)
        SE3_est = solver_manager.SE3_est_final
        se3_estimate_acc = np.matmul(se3_estimate_acc, SE3_est)
        pose_estimate_list.append(se3_estimate_acc)
        vo_twist_list.append(solver_manager.twist_final)
print("visualizing..")
SE3.post_process_pose_list_for_display_in_mem(pose_estimate_list)

if calc_vo:
    FileIO.write_vo_output_to_file(name,info,output_dir_path,vo_twist_list)

visualizer = Visualizer.Visualizer(ground_truth_list,plot_steering=plot_steering)
visualizer.visualize_ground_truth(clear=True,draw=False)
if plot_steering:
    visualizer.visualize_steering(encoder_list,clear=False,draw=False)
if calc_vo:
    visualizer.visualize_poses(pose_estimate_list, draw= False)
visualizer.show()