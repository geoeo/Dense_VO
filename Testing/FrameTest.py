#https://docs.python.org/3/library/unittest.html

import unittest
import numpy as np
import Frame
import Camera.Camera as Camera
import Camera.Intrinsic as Intrinsic
import Numerics.SE3 as SE3

class TestFrameMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pixels_uint8 = np.array([[1, 2], [3, 4]]).astype(np.uint8)
        cls.pixels_float32 = np.array([[1, 2], [3, 4]]).astype(np.float64)
        cls.se3_identity = SE3.SE3(np.identity(4,dtype='float32'))
        cls.intrinsic_identity = Intrinsic.Intrinsic(np.identity(3))
        cls.camera_identity = Camera.Camera(cls.intrinsic_identity,cls.se3_identity)

    def test_init(self):
        Frame.Frame(self.pixels_float32,self.camera_identity)

    def test_init_raise(self):
        with self.assertRaises(TypeError):
            Frame.Frame(self.pixels_uint8,self.camera_identity)



if __name__ == '__main__':
    unittest.main()