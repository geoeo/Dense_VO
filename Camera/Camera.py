import numpy as np
import Camera.Intrinsic as Intrinsic
import Numerics.SE3 as SE3
import Numerics.Utils as Utils


#TODO: Test
# https://referencesource.microsoft.com/#System.Numerics/System/Numerics/Matrix4x4.cs,b82966e485b5a306
def lookAt(camera_position: np.ndarray, camera_target: np.ndarray, camera_up: np.ndarray):
    z_axis = Utils.normalize(np.array(camera_position - camera_target))
    x_axis = Utils.normalize(np.cross(camera_up,z_axis))
    y_axis = np.cross(z_axis,x_axis)

    return np.array(x_axis,y_axis,z_axis,order='F')


"""An Object which encodes a Camera.

Attributes:
  intrinsic: An object of type IntrnsicsCamera
  extrinsic: An object of type SE3 
"""
class Camera:
    def __init__(self, intrinsic : Intrinsic, extrinsic : SE3):
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic




