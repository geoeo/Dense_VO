import numpy as np
import Camera.Camera as Camera
import Numerics.Utils as Utils
import Numerics.ImageProcessing as ImageProcessing
import cv2

"""An Object which encodes a Frame.

Attributes:
  image: 2d numpy array representing the pixels of an image 
  camera: an object 
"""


class Frame:
    def __init__(self, pixel_image : np.ndarray, depth_image : np.ndarray, camera : Camera, compute_gradients):
        if pixel_image.dtype != Utils.image_data_type:
            raise TypeError('Camera pixels are not of type float64 and probably not z standardised')
        if depth_image.dtype != Utils.depth_data_type_float and depth_image.dtype != Utils.depth_data_type_int:
            raise TypeError('Depth image is not of type float16 or uint16')

        self.pixel_image = ImageProcessing.z_standardise(pixel_image)
        self.pixel_depth = depth_image
        self.camera = camera
        if compute_gradients:
            #https://docs.opencv.org/3.4.1/d5/d0f/tutorial_py_gradients.html
            #self.grad_x = np.absolute(cv2.Sobel(pixel_image, Utils.image_data_type_open_cv, 1, 0))
            #self.grad_x = cv2.Sobel(pixel_image, Utils.image_data_type_open_cv, 1, 0, ksize=3)
            #self.grad_y = cv2.Sobel(pixel_image, Utils.image_data_type_open_cv, 0, 1, ksize=3)
            self.grad_x = cv2.Scharr(pixel_image, Utils.image_data_type_open_cv, 1, 0)
            self.grad_y = cv2.Scharr(pixel_image, Utils.image_data_type_open_cv, 0, 1)
            #self.grad_y = np.absolute(cv2.Sobel(pixel_image, Utils.image_data_type_open_cv, 0, 1))

    def scale_frame_by(self,scale_factor):
        self.pixel_image = cv2.resize(self.pixel_image, (0, 0), fx=scale_factor, fy=scale_factor)
        self.camera = self.camera.intrinsic.scale_by(scale_factor)

    def convolve_frame_by(self,scale_factor):
        (rows,cols) = self.pixel_image.size
        dst_size = (int(scale_factor * rows), int(scale_factor * cols))
        dst_size_flipped = (int(scale_factor * cols), int(scale_factor * rows))
        output_image = np.zeros(dst_size)
        cv2.pyrDown(self.pixel_image,output_image,dst_size_flipped)
        self.pixel_image = output_image
        self.camera = self.camera.intrinsic.scale_by(scale_factor)

    #TODO: TEST
    # Assumes that image has been down sampled with 2x2 image blocks i.e scale = 0.5
    def reconstruct_depth(self,x_low,y_low,layers_between):
        two_l = 2**layers_between
        two_l_minus_one = 0.5*(two_l - 1)
        x_high = two_l*x_low + two_l_minus_one
        y_high = two_l*y_low + two_l_minus_one
        return x_high , y_high