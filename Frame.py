import numpy as np
import Camera.Camera as Camera
import cv2

"""An Object which encodes a Frame.

Attributes:
  image: 2d numpy array representing the pixels of an image 
  camera: an object 
"""


class Frame:
    def __init__(self, pixel_image : np.ndarray, camera : Camera):
        if pixel_image.dtype != 'float64':
            raise TypeError('Camera pixels are not of type float64 and probably not z standardised')

        self.pixel_image = pixel_image
        self.camera = camera

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