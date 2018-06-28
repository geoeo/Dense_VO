import Camera.Intrinsic as Intrinsic
import Numerics.SE3 as SE3
"""An Object which encodes a Camera.

Attributes:
  intrinsic: An object of type IntrnsicsCamera
  extrinsic: An object of type SE3 
"""


class Camera:
    def __init__(self, intrinsic : Intrinsic, extrinsic : SE3):
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic




