import numpy as np
import Numerics.Utils as Utils
import Camera.Intrinsic as Intrinsic
import Camera.Camera as Camera
import VisualOdometry.Solver as Solver
from VisualOdometry import Frame
from Numerics import ImageProcessing
from Benchmark import Parser
from Benchmark import associate
from Numerics import SE3
from Visualization import Visualizer
import cv2

bench_path = '/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Bench/'
xyz_dataset = 'rgbd_dataset_freiburg1_xyz/'
rgb_folder = 'rgb/'
depth_folder = 'depth/'

dataset_root = bench_path+xyz_dataset
rgb_text = dataset_root +'rgb.txt'
depth_text = dataset_root+'depth.txt'
match_text = dataset_root+'matches.txt'
groundtruth_text = dataset_root+'groundtruth.txt'

#rgb_id_ref = 1305031102.175304
#rgb_id_target = 1305031102.211214

#rgb_id_ref = 1305031102.275326
#rgb_id_target = 1305031102.311267
#rgb_id_target = 1305031102.311267

rgb_id_ref = 1305031112.643246
rgb_id_target = 1305031112.679952
#rgb_id_target = 1305031112.743245

#rgb_id_ref = 1305031105.575449
#rgb_id_target = 1305031105.611378


#rgb_id_ref = 1305031108.743502
#rgb_id_target = 1305031108.775493
#rgb_id_target = 1305031108.811244

#rgb_id_ref = 1305031119.615017
#rgb_id_target = 1305031119.647903

#rgb_id_ref = 1305031106.675279
#rgb_id_target = 1305031106.711508


image_groundtruth_dict = dict(associate.match(rgb_text,groundtruth_text))

groundtruth_ts_ref = image_groundtruth_dict[rgb_id_ref]
groundtruth_data_ref = associate.return_groundtruth(groundtruth_text,groundtruth_ts_ref)
SE3_ref = Parser.generate_se3_from_groundtruth(groundtruth_data_ref)

groundtruth_ts_target = image_groundtruth_dict[rgb_id_target]
groundtruth_data_target = associate.return_groundtruth(groundtruth_text,groundtruth_ts_target)
SE3_target = Parser.generate_se3_from_groundtruth(groundtruth_data_target)

SE3_ref_target = SE3.pose_pose_composition_inverse(SE3_ref,SE3_target)

rgb_ref_file_path , depth_ref_file_path = associate.return_rgb_depth_from_rgb_selection(rgb_text,depth_text, match_text, dataset_root, rgb_id_ref)
rgb_target_file_path , depth_target_file_path = associate.return_rgb_depth_from_rgb_selection(rgb_text,depth_text, match_text, dataset_root, rgb_id_target)

im_greyscale_reference = cv2.imread(rgb_ref_file_path,cv2.IMREAD_GRAYSCALE).astype(Utils.image_data_type)
im_depth_reference = cv2.imread(depth_ref_file_path,cv2.IMREAD_ANYDEPTH).astype(Utils.depth_data_type_float)

im_greyscale_target = cv2.imread(rgb_target_file_path,cv2.IMREAD_GRAYSCALE).astype(Utils.image_data_type)
im_depth_target = cv2.imread(depth_target_file_path,cv2.IMREAD_ANYDEPTH).astype(Utils.depth_data_type_float)

im_greyscale_reference = ImageProcessing.z_standardise(im_greyscale_reference)
im_greyscale_target = ImageProcessing.z_standardise(im_greyscale_target)

(image_height,image_width) = im_greyscale_reference.shape

depth_factor = 5000.0
use_ndc = True

im_depth_reference /= depth_factor
im_depth_target /= depth_factor


se3_identity = np.identity(4, dtype=Utils.matrix_data_type)

intrinsic_identity = Intrinsic.Intrinsic(1, 1, image_width/2, image_height/2)
if use_ndc:
    intrinsic_identity = Intrinsic.Intrinsic(1, 1, 1/2, 1/2) # for ndc

camera_reference = Camera.Camera(intrinsic_identity, se3_identity)
camera_target = Camera.Camera(intrinsic_identity, se3_identity)

# We only need the gradients of the target frame
frame_reference = Frame.Frame(im_greyscale_reference, im_depth_reference, camera_reference, False)
frame_target = Frame.Frame(im_greyscale_target, im_depth_target, camera_target, True)

#visualizer = Visualizer.Visualizer(photometric_solver)

SE3_est = Solver.solve_photometric(frame_reference,
                                   frame_target,
                                   threadLock=None,
                                   pose_estimate_list=None,
                                   max_its=20000,
                                   eps = 0.0005,
                                   alpha_step=1.0,
                                   gradient_monitoring_window_start=20,
                                   image_range_offset_start=0,
                                   use_ndc=use_ndc,
                                   use_robust = True,
                                   track_pose_estimates = False,
                                   debug=False)

euler_angles_XYZ = SE3.rotationMatrixToEulerAngles(SE3.extract_rotation(SE3_est))
euler_angles_gt_XYZ = SE3.rotationMatrixToEulerAngles(SE3.extract_rotation(SE3_ref_target))

print('*'*80)
print('GROUND TRUTH\n')
print(SE3_ref_target)
print(Utils.radians_to_degrees(euler_angles_gt_XYZ[0]),
      Utils.radians_to_degrees(euler_angles_gt_XYZ[1]),
      Utils.radians_to_degrees(euler_angles_gt_XYZ[2]))
print('*'*80)

print(SE3_est)
print(Utils.radians_to_degrees(euler_angles_XYZ[0]),
      Utils.radians_to_degrees(euler_angles_XYZ[1]),
      Utils.radians_to_degrees(euler_angles_XYZ[2]))




#cv2.imshow('rgb ref',im_greyscale_reference)
#cv2.imshow('depth ref',im_depth_reference)

#cv2.imshow('rgb target',im_greyscale_target)
#cv2.imshow('depth target',im_depth_target)

#while True:
#    k = cv2.waitKey(5) & 0xFF
#    # ESC
#    if k == 27:
#        break

#cv2.destroyAllWindows()

