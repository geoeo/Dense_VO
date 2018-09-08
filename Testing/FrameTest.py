#https://docs.python.org/3/library/unittest.html

import unittest
import numpy as np
from VisualOdometry import Frame
import Camera.Camera as Camera
import Camera.Intrinsic as Intrinsic
import Numerics.Utils as Utils

class TestFrameMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pixels_uint8 = np.array([[1, 2], [3, 4]]).astype(np.uint8)
        cls.pixels_float32 = np.array([[1, 2], [3, 4]]).astype(Utils.image_data_type)
        cls.depth_float32 = np.array([[0.1, 0.5], [1, 2]]).astype(Utils.image_data_type)
        cls.se3_identity = np.identity(4, dtype=Utils.matrix_data_type)
        cls.intrinsic_identity = Intrinsic.Intrinsic(-1,-1,0,0)
        cls.camera_identity = Camera.Camera(cls.intrinsic_identity,cls.se3_identity)

    def test_init(self):
        Frame.Frame(self.pixels_float32, self.depth_float32, self.camera_identity, False)

    def test_init_raise(self):
        with self.assertRaises(TypeError):
            Frame.Frame(self.pixels_uint8, self.depth_float32, self.camera_identity, False)

if __name__ == '__main__':
    unittest.main()