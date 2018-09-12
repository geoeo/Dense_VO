import numpy as np
import cv2
import Numerics.Utils as Utils
import Camera.Intrinsic as Intrinsic
import Camera.Camera as Camera
import VisualOdometry.Solver as Solver
from VisualOdometry import Frame
from Numerics import ImageProcessing
from Numerics import SE3
from Benchmark import associate
import cv2

bench_path = '/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Bench/'
xyz_dataset = 'rgbd_dataset_freiburg1_xyz/'
rgb_folder = 'rgb/'
depth_folder = 'depth/'

dataset_root = bench_path+xyz_dataset
rgb_text = dataset_root +'rgb.txt'
depth_text = dataset_root+'depth.txt'
match_text = dataset_root+'matches.txt'

rgb_id_ref = 1305031102.175304
rgb_id_target = 1305031102.211214

rgb_ref_file_path , depth_ref_file_path = associate.return_rgb_depth_from_rgb_selection(rgb_text,depth_text, match_text, dataset_root, rgb_id_ref)
rgb_target_file_path , depth_target_file_path = associate.return_rgb_depth_from_rgb_selection(rgb_text,depth_text, match_text, dataset_root, rgb_id_target)

im_greyscale_reference = cv2.imread(rgb_ref_file_path,cv2.IMREAD_GRAYSCALE)
im_depth_reference = cv2.imread(depth_ref_file_path,cv2.IMREAD_ANYDEPTH).astype(Utils.depth_data_type)

im_greyscale_target = cv2.imread(rgb_target_file_path,cv2.IMREAD_GRAYSCALE)
im_depth_target = cv2.imread(depth_target_file_path,cv2.IMREAD_ANYDEPTH).astype(Utils.depth_data_type)

im_greyscale_reference = ImageProcessing.z_standardise(im_greyscale_reference)
im_greyscale_target = ImageProcessing.z_standardise(im_greyscale_target)

(image_height,image_width) = im_greyscale_reference.shape

depth_factor = 5000
use_ndc = False

im_depth_reference /= depth_factor
im_depth_target /= depth_factor


se3_identity = np.identity(4, dtype=Utils.matrix_data_type)

intrinsic_identity = Intrinsic.Intrinsic(-1, -1, image_width/2, image_height/2)
if use_ndc:
    intrinsic_identity = Intrinsic.Intrinsic(-1, -1, 1/2, 1/2) # for ndc

camera_reference = Camera.Camera(intrinsic_identity, se3_identity)
camera_target = Camera.Camera(intrinsic_identity, se3_identity)

# We only need the gradients of the target frame
frame_reference = Frame.Frame(im_greyscale_reference, im_depth_reference, camera_reference, False)
frame_target = Frame.Frame(im_greyscale_target, im_depth_target, camera_target, True)

SE3_est = Solver.solve_photometric(frame_reference, frame_target, 20000, 0.22, alpha_step=1.0, use_ndc=use_ndc, debug = False)
euler_angles_XYZ = SE3.rotationMatrixToEulerAngles(SE3.extract_rotation(SE3_est))

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

