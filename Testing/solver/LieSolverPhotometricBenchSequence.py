import numpy as np
import Numerics.Utils as Utils
import Camera.Intrinsic as Intrinsic
import Camera.Camera as Camera
import VisualOdometry.Solver as Solver
from VisualOdometry import Frame, SolverThreadManager
from Benchmark import Parser
from Benchmark import associate
from Visualization import Visualizer


bench_path = '/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Bench/'
xyz_dataset = 'rgbd_dataset_freiburg1_xyz/'
rgb_folder = 'rgb/'
depth_folder = 'depth/'

dataset_root = bench_path+xyz_dataset
rgb_text = dataset_root +'rgb.txt'
depth_text = dataset_root+'depth.txt'
match_text = dataset_root+'matches.txt'
groundtruth_text = dataset_root+'groundtruth.txt'

rgb_id_ref = 1305031102.175304
rgb_id_target = 1305031102.211214

rgb_id_ref_2 = 1305031102.175304
rgb_id_target_2 = 1305031102.211214

rgb_id_ref_3 = 1305031102.175304
rgb_id_target_3 = 1305031102.211214

rgb_id_ref_4 = 1305031102.175304
rgb_id_target_4 = 1305031102.211214

rgb_id_ref_5 = 1305031102.175304
rgb_id_target_5 = 1305031102.211214

ref_id_list = [rgb_id_ref]
target_id_list = [rgb_id_target]

#ref_id_list = [rgb_id_ref, rgb_id_ref_2, rgb_id_ref_3, rgb_id_ref_4, rgb_id_ref_5]
#target_id_list = [rgb_id_target, rgb_id_target_2, rgb_id_target_3, rgb_id_target_4, rgb_id_target_5]


ground_truth_list = []
pose_estimate_list = []
ref_image_list = []
target_image_list = []

depth_factor = 5000.0
use_ndc = True


image_groundtruth_dict = dict(associate.match(rgb_text,groundtruth_text))


for i in range(0, len(ref_id_list)):

    ref_id = ref_id_list[i]
    target_id = target_id_list[i]

    SE3_ref_target = Parser.generate_ground_truth_se3(groundtruth_text,image_groundtruth_dict,ref_id,target_id)
    im_greyscale_reference, im_depth_reference = Parser.generate_image_depth_pair(dataset_root,rgb_text,depth_text,match_text,ref_id)
    im_greyscale_target, im_depth_target = Parser.generate_image_depth_pair(dataset_root,rgb_text,depth_text,match_text,target_id)

    ground_truth_list.append(SE3_ref_target)
    ref_image_list.append((im_greyscale_reference, im_depth_reference))
    target_image_list.append((im_greyscale_target, im_depth_target))


im_greyscale_reference_1, im_depth_reference_1 = ref_image_list[0]
(image_height, image_width) = im_greyscale_reference_1.shape
se3_identity = np.identity(4, dtype=Utils.matrix_data_type)
# image gradient induces a coordiante system where y is flipped i.e have to flip it here
intrinsic_identity = Intrinsic.Intrinsic(-1, -1, image_width/2, image_height/2)
if use_ndc:
    intrinsic_identity = Intrinsic.Intrinsic(-1, -1, 1/2, 1/2) # for ndc

camera_reference = Camera.Camera(intrinsic_identity, se3_identity)
camera_target = Camera.Camera(intrinsic_identity, se3_identity)

visualizer = Visualizer.Visualizer(ground_truth_list)

for i in range(0, len(ref_image_list)):
    im_greyscale_reference, im_depth_reference = ref_image_list[i]
    im_greyscale_target, im_depth_target = target_image_list[i]

    im_depth_reference /= depth_factor
    im_depth_target /= depth_factor

    # We only need the gradients of the target frame
    frame_reference = Frame.Frame(im_greyscale_reference, im_depth_reference, camera_reference, False)
    frame_target = Frame.Frame(im_greyscale_target, im_depth_target, camera_target, True)


    # TODO: Dont allocate new class every iteraiton
    solver_manager = SolverThreadManager.Manager(1,
                                                 "Solver Manager",
                                                 frame_reference,
                                                 frame_target,
                                                 max_its=20000,
                                                 eps=0.00005,
                                                 alpha_step=0.05,
                                                 gradient_monitoring_window_start=0,
                                                 use_ndc=use_ndc,
                                                 use_robust=True,
                                                 track_pose_estimates=True,
                                                 debug=False)


    solver_manager.start()
    solver_manager.join()  # wait to complete
    pose_estimate_list.append(solver_manager.SE3_est_final)
    visualizer.visualize_poses(pose_estimate_list, draw= False)
    visualizer.show()







#SE3_final = solver_manager.pose_estimate_list[len(solver_manager.pose_estimate_list)-1]

#euler_angles_XYZ = SE3.rotationMatrixToEulerAngles(SE3.extract_rotation(SE3_final))
#euler_angles_gt_XYZ = SE3.rotationMatrixToEulerAngles(SE3.extract_rotation(SE3_ref_target))

#print('*'*80)
#print('GROUND TRUTH\n')
#print(SE3_ref_target)
#print(Utils.radians_to_degrees(euler_angles_gt_XYZ[0]),
#      Utils.radians_to_degrees(euler_angles_gt_XYZ[1]),
#      Utils.radians_to_degrees(euler_angles_gt_XYZ[2]))
#print('*'*80)

#print(SE3_final)
#print(Utils.radians_to_degrees(euler_angles_XYZ[0]),
#      Utils.radians_to_degrees(euler_angles_XYZ[1]),
#      Utils.radians_to_degrees(euler_angles_XYZ[2]))



