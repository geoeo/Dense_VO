import numpy as np
from Numerics import Utils, SE3
from Camera import Intrinsic, Camera
from VisualOdometry import Frame, SolverThreadManager
from Benchmark import Parser, associate, ListGenerator, FileIO
from Visualization import Visualizer, PostProcessGroundTruth

# start
#start_idx = 1311868164.363181 # 2965

#start_idx = 1311868164.430940 # 3rd

#start_idx = 1311868164.531025 # 6th

#start_idx = 1311868165.063108 # 20

#start_idx = 1311868166.331189 # 60 - rmse starts to go bad

#start_idx = 1311868174.699578 # 300 - try this

start_idx = 1311868250.648757 # try this

#start_idx = 1311868165.999133

# Y Up
#start_idx = 1311868216.474357

# Y Down - motion prior
#start_idx = 1311868219.010718
#start_idx = 1311868219.210391
#start_idx = 1311868219.342858 # good

#X Right
#start_idx = 1311868164.899132
#start_idx = 1311868166.631287
# rotation fools into thinking its x translation
#start_idx = 1311868169.199452
#start_idx = 1311868169.163498

#start_idx = 1311868171.399409

# good x/y values - motin prior, shows bad recovery, better with higher offset (2 with prior)
#start_idx = 1311868235.279710

bench_path = '/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Bench/'
xyz_dataset = 'rgbd_dataset_freiburg2_desk/'
rgb_folder = 'rgb/'
depth_folder = 'depth/'
output_dir = 'output/'

dataset_root = bench_path+xyz_dataset
output_dir_path = dataset_root + output_dir
rgb_text = dataset_root +'rgb.txt'
depth_text = dataset_root+'depth.txt'
match_text = dataset_root+'matches.txt'
groundtruth_text = dataset_root+'groundtruth.txt'

groundtruth_dict = associate.read_file_list(groundtruth_text)

rgb_folder = dataset_root+rgb_folder
depth_folder = dataset_root+depth_folder

rgb_files = ListGenerator.get_files_from_directory(rgb_folder, delimiter='.')
depth_files = ListGenerator.get_files_from_directory(depth_folder, delimiter='.')

rgb_file_total = len(rgb_files)
depth_file_total = len(depth_files)



ground_truth_acc = np.identity(4,Utils.matrix_data_type)
se3_estimate_acc = np.identity(4,Utils.matrix_data_type)
ground_truth_list = []
pose_estimate_list = []
ref_image_list = []
target_image_list = []
vo_twist_list = []

depth_factor = 5000.0
#depth_factor = 1.0
use_ndc = False
calc_vo = True
plot_steering = False

max_count = 301
offset = 1

#TODO investigate index after rounding
name = f"{start_idx:.9f}"

max_its = 50
eps = 0.00000005  #0.001, 0.00001, 0.00005, 0.00000001
alpha_step = 1.5  # 0.001, 0.008 - motion pri
gradient_monitoring_window_start = 1
image_range_offset_start = 0
use_ndc = use_ndc
use_robust = True
track_pose_estimates = True
use_motion_prior = False
use_ackermann = False

divide_depth = True
debug = False

additional_info = ''
additional_info += 'solver_1_other_res_2_z_stand_with_duplicates_kernel_1_res_no_flag'
if not divide_depth:
    additional_info += '_no_depth_divide'


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

match_dict = associate.read_file_list(match_text)
image_groundtruth_dict = dict(associate.match(rgb_text, groundtruth_text, max_difference=0.2,with_duplicates=True))

post_process_gt = PostProcessGroundTruth.PostProcessTUM_F2()

print(name+'_'+info+'\n')

start = ListGenerator.get_index_of_id(start_idx,rgb_files)

ref_id_list, target_id_list, ref_files_failed_to_load = ListGenerator.generate_files_to_load_match(
    rgb_files,
    start=start,
    max_count=max_count,
    offset=offset,
    ground_truth_dict=image_groundtruth_dict,
    match_dict = match_dict,
    reverse=False)

if len(ref_files_failed_to_load) > 0:
    print(ref_files_failed_to_load)
    print('\n')

for i in range(0, len(ref_id_list)):

    ref_id = ref_id_list[i]
    target_id = target_id_list[i]

    SE3_ref_target = Parser.generate_ground_truth_se3(groundtruth_dict,image_groundtruth_dict,ref_id,target_id,None)
    im_greyscale_reference, im_depth_reference = Parser.generate_image_depth_pair_match(dataset_root,rgb_text,depth_text,match_text,ref_id)
    im_greyscale_target, im_depth_target = Parser.generate_image_depth_pair_match(dataset_root,rgb_text,depth_text,match_text,target_id)

    post_process_gt.post_process_in_mem(SE3_ref_target)

    ground_truth_acc = np.matmul(ground_truth_acc,SE3_ref_target)

    ground_truth_list.append(ground_truth_acc)
    ref_image_list.append((im_greyscale_reference, im_depth_reference))
    target_image_list.append((im_greyscale_target, im_depth_target))


im_greyscale_reference_1, im_depth_reference_1 = ref_image_list[0]
(image_height, image_width) = im_greyscale_reference_1.shape
se3_identity = np.identity(4, dtype=Utils.matrix_data_type)

intrinsic_identity = Intrinsic.Intrinsic(520.9, 521.0, 321.5, 249.7) # freiburg_2
if use_ndc:
    #intrinsic_identity = Intrinsic.Intrinsic(1, 1, 1/2, 1/2) # for ndc
    intrinsic_identity = Intrinsic.Intrinsic(1, 521.0/520.9, 321.5/image_width, 249.7/image_height) # for ndc


camera_reference = Camera.Camera(intrinsic_identity, se3_identity)
camera_target = Camera.Camera(intrinsic_identity, se3_identity)

visualizer = Visualizer.Visualizer(ground_truth_list)

motion_cov_inv = np.identity(6,dtype=Utils.matrix_data_type)
#motion_cov_inv = np.zeros((6,6),dtype=Utils.matrix_data_type)
twist_prior = np.zeros((6,1),dtype=Utils.matrix_data_type)

print('starting...\n')

for i in range(0, len(ref_image_list)):
    im_greyscale_reference, im_depth_reference = ref_image_list[i]
    im_greyscale_target, im_depth_target = target_image_list[i]

    if divide_depth:
        im_depth_reference /= depth_factor
        im_depth_target /= depth_factor

    max_depth = np.amax(im_depth_reference)

    # We only need the gradients of the target frame
    frame_reference = Frame.Frame(im_greyscale_reference, im_depth_reference, camera_reference, False)
    frame_target = Frame.Frame(im_greyscale_target, im_depth_target, camera_target, True)

    solver_manager = SolverThreadManager.Manager(1,
                                                 "Solver Manager",
                                                 frame_reference,
                                                 frame_target,
                                                 max_its=max_its,
                                                 eps=eps,  #0.001, 0.00001, 0.00005, 0.00000001
                                                 alpha_step=alpha_step,  # 0.001, 0.008 - motion pri
                                                 gradient_monitoring_window_start=gradient_monitoring_window_start,
                                                 image_range_offset_start=image_range_offset_start,
                                                 max_depth=max_depth,
                                                 twist_prior=twist_prior,
                                                 motion_cov_inv = motion_cov_inv,
                                                 use_ndc=use_ndc,
                                                 use_robust=use_robust,
                                                 track_pose_estimates=track_pose_estimates,
                                                 use_motion_prior=use_motion_prior,
                                                 debug=debug)

    if calc_vo:
        solver_manager.start()
        solver_manager.join()  # wait to complete
        print('iteration ', i + 1, ' complete')

        motion_cov_inv = solver_manager.motion_cov_inv_final
        #motion_cov_inv = np.add(motion_cov_inv,solver_manager.motion_cov_inv_final)
        twist_prior = np.multiply(1.0,solver_manager.twist_final)
        #twist_prior = np.add(twist_prior,solver_manager.twist_final)
        #se3_estimate_acc = np.matmul(solver_manager.SE3_est_final,se3_estimate_acc)
        se3_estimate_acc = np.matmul(se3_estimate_acc,solver_manager.SE3_est_final)
        pose_estimate_list.append(se3_estimate_acc)
        vo_twist_list.append(solver_manager.twist_final)
print("visualizing..")

if calc_vo:
    FileIO.write_vo_output_to_file(name,info,output_dir_path,vo_twist_list)

visualizer.visualize_ground_truth(clear=True,draw=False)
if calc_vo:
    visualizer.visualize_poses(pose_estimate_list, draw= False)
visualizer.show()







