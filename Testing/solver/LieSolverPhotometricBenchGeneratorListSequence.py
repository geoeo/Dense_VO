import numpy as np
from Numerics import Utils, SE3
from Camera import Intrinsic, Camera
from VisualOdometry import Frame, SolverThreadManager
from Benchmark import Parser, associate, ListGenerator
from Visualization import Visualizer
from math import pi



bench_path = '/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Bench/'
xyz_dataset = 'rgbd_dataset_freiburg1_xyz/'
rgb_folder = 'rgb/'
depth_folder = 'depth/'

dataset_root = bench_path+xyz_dataset
rgb_text = dataset_root +'rgb.txt'
depth_text = dataset_root+'depth.txt'
match_text = dataset_root+'matches.txt'
groundtruth_text = dataset_root+'groundtruth.txt'

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

depth_factor = 5000.0
#depth_factor = 1.0
use_ndc = True

match_dict = associate.read_file_list(match_text)
image_groundtruth_dict = dict(associate.match(rgb_text, groundtruth_text))

#se3_ground_truth_prior = np.transpose(SE3.quaternion_to_s03(0.6132, 0.5962, -0.3311, -0.3986))
se3_ground_truth_prior = SE3.makeS03(0,0,pi)
se3_ground_truth_prior = np.append(se3_ground_truth_prior,np.zeros((3,1),dtype=Utils.matrix_data_type),axis=1)
se3_ground_truth_prior = SE3.append_homogeneous_along_y(se3_ground_truth_prior)
#se3_ground_truth_prior = SE3.invert(se3_ground_truth_prior)
se3_ground_truth_prior[0:3,3] = 0

# start
#start = ListGenerator.get_index_of_id(1305031102.175304,rgb_files)

# Y Trans
#start = ListGenerator.get_index_of_id(1305031119.079223,rgb_files)

# X Trans
start = ListGenerator.get_index_of_id(1305031108.143334,rgb_files)

ref_id_list, target_id_list, ref_files_failed_to_load = ListGenerator.generate_files_to_load(
    rgb_files,
    start=start,
    max_count=5,
    offset=1,
    ground_truth_dict=image_groundtruth_dict,
    match_dict = match_dict)

for i in range(0, len(ref_id_list)):

    ref_id = ref_id_list[i]
    target_id = target_id_list[i]

    SE3_ref_target = Parser.generate_ground_truth_se3(groundtruth_text,image_groundtruth_dict,ref_id,target_id,None)
    im_greyscale_reference, im_depth_reference = Parser.generate_image_depth_pair(dataset_root,rgb_text,depth_text,match_text,ref_id)
    im_greyscale_target, im_depth_target = Parser.generate_image_depth_pair(dataset_root,rgb_text,depth_text,match_text,target_id)

    ground_truth_acc = np.matmul(SE3_ref_target,ground_truth_acc)

    ground_truth_list.append(ground_truth_acc)
    ref_image_list.append((im_greyscale_reference, im_depth_reference))
    target_image_list.append((im_greyscale_target, im_depth_target))


im_greyscale_reference_1, im_depth_reference_1 = ref_image_list[0]
(image_height, image_width) = im_greyscale_reference_1.shape
se3_identity = np.identity(4, dtype=Utils.matrix_data_type)
# image gradient induces a coordiante system where y is flipped i.e have to flip it here
intrinsic_identity = Intrinsic.Intrinsic(-517.3, -516.5, 318.6, 239.5) # freiburg_1
if use_ndc:
    #intrinsic_identity = Intrinsic.Intrinsic(1, 1, 1/2, 1/2) # for ndc
    intrinsic_identity = Intrinsic.Intrinsic(-1, -516.5/517.3, 318.6/image_width, 239.5/image_height) # for ndc


camera_reference = Camera.Camera(intrinsic_identity, se3_identity)
camera_target = Camera.Camera(intrinsic_identity, se3_identity)

visualizer = Visualizer.Visualizer(ground_truth_list)

motion_cov_inv = np.identity(6,dtype=Utils.matrix_data_type)
twist_prior = np.zeros((6,1),dtype=Utils.matrix_data_type)

for i in range(0, len(ref_image_list)):
    im_greyscale_reference, im_depth_reference = ref_image_list[i]
    im_greyscale_target, im_depth_target = target_image_list[i]

    im_depth_reference /= depth_factor
    im_depth_target /= depth_factor


    #depth_t = (im_depth_reference != 0).astype(Utils.depth_data_type_float)
    #im_depth_reference = np.add(im_depth_reference,depth_t)
    #depth_t = (im_depth_target != 0).astype(Utils.depth_data_type_float)
    #im_depth_target = np.add(im_depth_target,depth_t)

    # We only need the gradients of the target frame
    frame_reference = Frame.Frame(im_greyscale_reference, im_depth_reference, camera_reference, False)
    frame_target = Frame.Frame(im_greyscale_target, im_depth_target, camera_target, True)

    solver_manager = SolverThreadManager.Manager(1,
                                                 "Solver Manager",
                                                 frame_reference,
                                                 frame_target,
                                                 max_its=50,
                                                 eps=0.002,  #0.001, 0.00001, 0.00005, 0.00000001
                                                 alpha_step=1.0,  # 0.1, 0.015, 0.04, 0.005, 0.55 - motion prior
                                                 gradient_monitoring_window_start=0,
                                                 image_range_offset_start=0,
                                                 twist_prior=twist_prior,
                                                 motion_cov_inv = motion_cov_inv,
                                                 use_ndc=use_ndc,
                                                 use_robust=True,
                                                 track_pose_estimates=True,
                                                 use_motion_prior=True,
                                                 debug=False)

    solver_manager.start()
    solver_manager.join()  # wait to complete

    motion_cov_inv = solver_manager.motion_cov_inv_final
    #motion_cov_inv = np.add(motion_cov_inv,solver_manager.motion_cov_inv_final)
    twist_prior = np.multiply(1.0,solver_manager.twist_final)
    #twist_prior = np.add(twist_prior,solver_manager.twist_final)
    se3_estimate_acc = np.matmul(solver_manager.SE3_est_final,se3_estimate_acc)
    pose_estimate_list.append(se3_estimate_acc)
visualizer.visualize_poses(pose_estimate_list, draw= False)
visualizer.show()






