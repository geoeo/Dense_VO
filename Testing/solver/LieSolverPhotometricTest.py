import numpy as np
import cv2
import Numerics.Utils as Utils
import Camera.Intrinsic as Intrinsic
import Camera.Camera as Camera
import VisualOdometry.Solver as Solver
from VisualOdometry import Frame
from Numerics import ImageProcessing
from Numerics import SE3

# TODO use raw depth values

#im_greyscale_reference = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Synthetic/framebuffer_fov_90_square_Y.png',cv2.IMREAD_GRAYSCALE)
#im_greyscale_reference = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Synthetic/framebuffer_fov_90_square.png',cv2.IMREAD_GRAYSCALE)
#im_greyscale_reference = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Synthetic/framebuffer_fov_90_square_negative.png',cv2.IMREAD_GRAYSCALE)
im_greyscale_reference = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Home_Images/Images_ZR300_X_Trans_Depth_Aligned_Scaled/image_40.png',cv2.IMREAD_GRAYSCALE)
#im_greyscale_reference = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Home_Images/Images_ZR300_Y_Trans_Depth_Aligned_Scaled/image_40.png',cv2.IMREAD_GRAYSCALE)
#im_greyscale_reference = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Home_Images/Images_ZR300_Board_X_Trans_Depth_Aligned_Scaled/image_126.png',cv2.IMREAD_GRAYSCALE)

im_greyscale_reference = ImageProcessing.z_standardise(im_greyscale_reference)


#im_greyscale_target = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Synthetic/framebuffer_translated_fov_90_square_Y.png',cv2.IMREAD_GRAYSCALE)
#im_greyscale_target = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Synthetic/framebuffer_translated_fov_90_square.png',cv2.IMREAD_GRAYSCALE)
#im_greyscale_target = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Synthetic/framebuffer_left_90_fov_90_square.png',cv2.IMREAD_GRAYSCALE)
#im_greyscale_target = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Synthetic/framebuffer_translated_fov_90_square_negative.png',cv2.IMREAD_GRAYSCALE)
im_greyscale_target = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Home_Images/Images_ZR300_X_Trans_Depth_Aligned_Scaled/image_41.png',cv2.IMREAD_GRAYSCALE)
#im_greyscale_target = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Home_Images/Images_ZR300_Y_Trans_Depth_Aligned_Scaled/image_41.png',cv2.IMREAD_GRAYSCALE)
#im_greyscale_target = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Home_Images/Images_ZR300_Board_X_Trans_Depth_Aligned_Scaled/image_127.png',cv2.IMREAD_GRAYSCALE)

im_greyscale_target = ImageProcessing.z_standardise(im_greyscale_target)

#depth_reference = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Synthetic/depthbuffer_fov_90_square_Y.png',cv2.IMREAD_ANYDEPTH).astype(Utils.depth_data_type)
#depth_reference = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Synthetic/depthbuffer_fov_90_square.png',cv2.IMREAD_ANYDEPTH).astype(Utils.depth_data_type)
#depth_reference = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Synthetic/depthbuffer_fov_90_square_negative.png',cv2.IMREAD_ANYDEPTH).astype(Utils.depth_data_type)
depth_reference = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Home_Images/Images_ZR300_X_Trans_Depth_Aligned_Scaled/image_depth_aligned_40.png',cv2.IMREAD_ANYDEPTH).astype(Utils.depth_data_type_int)
#depth_reference = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Home_Images/Images_ZR300_Y_Trans_Depth_Aligned_Scaled/image_depth_aligned_40.png',cv2.IMREAD_ANYDEPTH).astype(Utils.depth_data_type)
#depth_reference = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Home_Images/Images_ZR300_Board_X_Trans_Depth_Aligned_Scaled/image_depth_aligned_126.png',cv2.IMREAD_ANYDEPTH).astype(Utils.depth_data_type)
#depth_reference = ImageProcessing.z_standardise(depth_reference)

#depth_target = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Synthetic/depthbuffer_translated_fov_90_square_Y.png',cv2.IMREAD_ANYDEPTH).astype(Utils.depth_data_type)
#depth_target = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Synthetic/depthbuffer_translated_fov_90_square.png',cv2.IMREAD_ANYDEPTH).astype(Utils.depth_data_type)
#depth_target = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Synthetic/depthbuffer_left_90_fov_90_square.png',cv2.IMREAD_ANYDEPTH).astype(Utils.depth_data_type)
depth_target = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Home_Images/Images_ZR300_X_Trans_Depth_Aligned_Scaled/image_depth_aligned_41.png',cv2.IMREAD_ANYDEPTH).astype(Utils.depth_data_type_int)
#depth_target = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Home_Images/Images_ZR300_Y_Trans_Depth_Aligned_Scaled/image_depth_aligned_41.png',cv2.IMREAD_ANYDEPTH).astype(Utils.depth_data_type)
#depth_target = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Home_Images/Images_ZR300_Board_X_Trans_Depth_Aligned_Scaled/image_depth_aligned_127.png',cv2.IMREAD_ANYDEPTH).astype(Utils.depth_data_type)
#depth_target = ImageProcessing.z_standardise(depth_target)

(image_height,image_width) = im_greyscale_reference.shape

# Some depth image were aquired without scaling i.e. scale here
depth_factor = 1000
use_ndc = True

#depth_target *= depth_factor
#depth_reference *= depth_factor

se3_identity = np.identity(4, dtype=Utils.matrix_data_type)

# fx and fy affect the resulting coordiante system of the se3 matrix
intrinsic_identity = Intrinsic.Intrinsic(-1, -1, image_width/2, image_height/2)
if use_ndc:
    intrinsic_identity = Intrinsic.Intrinsic(-1, -1, 1/2, 1/2) # for ndc


# reference frame is assumed to be the origin
# target frame SE3 is unknown i.e. what we are trying to solve
camera_reference = Camera.Camera(intrinsic_identity, se3_identity)
camera_target = Camera.Camera(intrinsic_identity, se3_identity)

# We only need the gradients of the target frame
frame_reference = Frame.Frame(im_greyscale_reference, depth_reference, camera_reference, False)
frame_target = Frame.Frame(im_greyscale_target, depth_target, camera_target, True)

SE3_est = Solver.solve_photometric(frame_reference, frame_target, 20000, 0.01, alpha_step=1.0, use_ndc = use_ndc, debug = False)
euler_angles_XYZ = SE3.rotationMatrixToEulerAngles(SE3.extract_rotation(SE3_est))

print(SE3_est)
print(Utils.radians_to_degrees(euler_angles_XYZ[0]),
      Utils.radians_to_degrees(euler_angles_XYZ[1]),
      Utils.radians_to_degrees(euler_angles_XYZ[2]))



