import numpy as np
import Camera.Intrinsic as Intrinsic
import Numerics.SE3 as SE3
import Numerics.Utils as Utils


#TODO: Test this
# https://referencesource.microsoft.com/#System.Numerics/System/Numerics/Matrix4x4.cs,b82966e485b5a306
def lookAt(camera_position: np.ndarray, camera_target: np.ndarray, camera_up: np.ndarray):
    z_axis = Utils.normalize(np.array(camera_position - camera_target))
    x_axis = Utils.normalize(np.cross(camera_up,z_axis))
    y_axis = np.cross(z_axis,x_axis)

    translation_x = -np.dot(x_axis,camera_position)
    translation_y = -np.dot(y_axis,camera_position)
    translation_z = -np.dot(z_axis,camera_position)
    translation = np.array([translation_x,translation_y,translation_z,1.0])

    return np.array(x_axis,y_axis,z_axis,translation,order='F')



"""An Object which encodes a Camera.

Attributes:
  intrinsic: An object of type IntrnsicsCamera
  extrinsic: An object of type SE3 
"""
class Camera:
    def __init__(self, intrinsic : Intrinsic, extrinsic : SE3):
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic

    def to_camera_space(self,point_3D_world):
        return np.matmul(self.extrinsic.se3,point_3D_world)

    def to_world_space(self,point_3D_camera):
        return np.matmul(self.extrinsic.se3_inv, point_3D_camera)

    def apply_intrinsics(self,point_3D_camera):
        return np.matmul(self.intrinsic.K,point_3D_camera)

    def apply_perspective_projection_to_camera(self,point_3D_camera):
        point_pesrp = np.matmul(self.intrinsic.K,point_3D_camera)
        return np.array([point_pesrp[0]/point_pesrp[2],point_pesrp[1]/point_pesrp[2],1])

    def apply_perspetive_pipeline_to_world(self,point_3D_world):
        return self.apply_perspective_projection_to_camera(self.to_camera_space(point_3D_world))




