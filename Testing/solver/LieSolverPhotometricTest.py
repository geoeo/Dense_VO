import numpy as np
import cv2
import Numerics.ImageProcessing as ImageProcessing
import Numerics.Utils as Utils
import Camera.Intrinsic as Intrinsic
import Camera.Camera as Camera
import Numerics.Solver as Solver
import Frame

# TODO use raw depth values

im_greyscale_reference = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Synthetic/framebuffer_fov_90_square.png',0)
im_greyscale_reference = ImageProcessing.z_standardise(im_greyscale_reference)

im_greyscale_target = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Synthetic/framebuffer_translated_fov_90_square.png',0)
im_greyscale_target = ImageProcessing.z_standardise(im_greyscale_target)

depth_reference = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Synthetic/depthbuffer_fov_90_square.png',0).astype(Utils.depth_data_type)
#depth_reference = ImageProcessing.z_standardise(depth_reference)

depth_target = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Synthetic/depthbuffer_translated_fov_90_square.png',0).astype(Utils.depth_data_type)
#depth_target = ImageProcessing.z_standardise(depth_target)

(image_height,image_width) = im_greyscale_reference.shape

se3_identity = np.identity(4, dtype=Utils.matrix_data_type)
intrinsic_identity = Intrinsic.Intrinsic(-1, -1, image_width/2, image_height/2)
#intrinsic_identity = Intrinsic.Intrinsic(-1, -1, 0, 0)

# reference frame is assumed to be the origin
# target frame SE3 is unknown i.e. what we are trying to solve
camera_reference = Camera.Camera(intrinsic_identity, se3_identity)
camera_target = Camera.Camera(intrinsic_identity, se3_identity)

# We only need the gradients of the target frame
frame_reference = Frame.Frame(im_greyscale_reference,depth_reference,camera_reference,False)
frame_target = Frame.Frame(im_greyscale_target,depth_target,camera_target,True)

SE3_est = Solver.solve_photometric(frame_reference,frame_target,20000,0.01)



