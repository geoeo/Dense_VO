import numpy as np
import cv2
import Camera.Intrinsic as Intrinsic
import Camera.Camera as Camera
from VisualOdometry import Frame
from Numerics import ImageProcessing, Utils

#im_greyscale = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Home_Images/Images_ZR300_XTrans/image_1.png',0)
#im_greyscale = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/rccar_26_09_18/marc_1_full/color/966816.173323313.png',cv2.IMREAD_GRAYSCALE)
im_greyscale = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Bench/rgbd_dataset_freiburg2_desk/rgb/1311868164.363181.png',cv2.IMREAD_GRAYSCALE)
#im_greyscale = im_greyscale.astype(Utils.image_data_type)

pixels_standardised = ImageProcessing.z_standardise(im_greyscale)
pixels_norm = im_greyscale.astype(np.float64)

pixels_normalized_disp = ImageProcessing.normalize_to_image_space(pixels_standardised)
pixels_disp = ImageProcessing.normalize_to_image_space(pixels_norm)
depth_image = pixels_standardised.astype(Utils.depth_data_type_int)

se3_identity = np.identity(4, dtype=Utils.matrix_data_type)
intrinsic_identity = Intrinsic.Intrinsic(-1, -1, 0, 0)
camera_identity = Camera.Camera(intrinsic_identity, se3_identity)

frame = Frame.Frame(pixels_standardised, depth_image, camera_identity, True)

#cv2.imshow('sobel x',frame.grad_x)
#cv2.imshow('neg sobel x',-frame.grad_x)
#cv2.imshow('sobel y',frame.grad_y)
cv2.imshow('image',pixels_disp)
cv2.imshow('image z-standard',pixels_normalized_disp)


while True:
    k = cv2.waitKey(5) & 0xFF
    # ESC
    if k == 27:
        break

cv2.destroyAllWindows()