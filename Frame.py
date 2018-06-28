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