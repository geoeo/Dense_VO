import numpy as np
import cv2
from Numerics import Utils
import Camera.Intrinsic as Intrinsic
import Camera.Camera as Camera
import  Solver_Cython
from VisualOdometry import Frame
from Numerics import ImageProcessing

# TODO use raw depth values

im_greyscale_reference = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Synthetic/framebuffer_fov_90_square_Y.png',0)
#im_greyscale_reference = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Synthetic/framebuffer_fov_90_square.png',0)
#im_greyscale_reference = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Synthetic/framebuffer_fov_90_square_negative.png',0)
#im_greyscale_reference = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Home_Images/Images_ZR300_X_Trans_Depth/image_25.png',0)
im_greyscale_reference = ImageProcessing.z_standardise(im_greyscale_reference)

im_greyscale_target = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Synthetic/framebuffer_translated_fov_90_square_Y.png',0)
#im_greyscale_target = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Synthetic/framebuffer_translated_fov_90_square.png',0)
#im_greyscale_target = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Synthetic/framebuffer_left_90_fov_90_square.png',0)
#im_greyscale_target = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Synthetic/framebuffer_translated_fov_90_square_negative.png',0)
#im_greyscale_target = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Home_Images/Images_ZR300_X_Trans_Depth/image_30.png',0)
im_greyscale_target = ImageProcessing.z_standardise(im_greyscale_target)

depth_reference = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Synthetic/depthbuffer_fov_90_square_Y.png',0).astype(
    Utils.depth_data_type)
#depth_reference = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Synthetic/depthbuffer_fov_90_square.png',0).astype(Utils.depth_data_type)
#depth_reference = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Synthetic/depthbuffer_fov_90_square_negative.png',0).astype(Utils.depth_data_type)
#depth_reference = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Home_Images/Images_ZR300_X_Trans_Depth/image_depth_25.png',0).astype(Utils.depth_data_type)
#depth_reference = ImageProcessing.z_standardise(depth_reference)

depth_target = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Synthetic/depthbuffer_translated_fov_90_square_Y.png',0).astype(
    Utils.depth_data_type)
#depth_target = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Synthetic/depthbuffer_translated_fov_90_square.png',0).astype(Utils.depth_data_type)
#depth_target = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Synthetic/depthbuffer_left_90_fov_90_square.png',0).astype(Utils.depth_data_type)
#depth_target = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Synthetic/depthbuffer_translated_fov_90_square_negative.png',0).astype(Utils.depth_data_type)
#depth_target = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Home_Images/Images_ZR300_X_Trans_Depth/image_depth_30.png',0).astype(Utils.depth_data_type)
#depth_target = ImageProcessing.z_standardise(depth_target)

(image_height,image_width) = im_greyscale_reference.shape

se3_identity = np.identity(4, dtype=Utils.matrix_data_type)
# fx and fy affect the resulting coordiante system of the se3 matrix
intrinsic_identity = Intrinsic.Intrinsic(-1, 1, image_width/2, image_height/2)
#intrinsic_identity = Intrinsic.Intrinsic(-1, -1, 1/2, 1/2) # for ndc

# reference frame is assumed to be the origin
# target frame SE3 is unknown i.e. what we are trying to solve
camera_reference = Camera.Camera(intrinsic_identity, se3_identity)
camera_target = Camera.Camera(intrinsic_identity, se3_identity)

# We only need the gradients of the target frame
frame_reference = Frame.Frame(im_greyscale_reference, depth_reference, camera_reference, False)
frame_target = Frame.Frame(im_greyscale_target, depth_target, camera_target, True)

SE3_est = Solver_Cython.solve_photometric(frame_reference, frame_target, 20000, 0.5, debug = False)

print(SE3_est)



