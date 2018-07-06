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
    translation = np.array([translation_x,translation_y,translation_z]).reshape((3,1))

    x_axis = x_axis.reshape((3,1))
    y_axis = y_axis.reshape((3,1))
    z_axis = z_axis.reshape((3,1))
    translation = translation.reshape((3,1))

    #TODO: Make this more efficient
    mat = np.append(np.append(np.append(x_axis,y_axis,axis=1),z_axis,axis=1),translation,axis=1)

    return np.append(mat,Utils.homogenous_for_SE3(),axis=0)



"""An Object which encodes a Camera.

Attributes:
  intrinsic: An object of type IntrnsicsCamera
  extrinsic: An object of type SE3 
"""
class Camera:
    def __init__(self, intrinsic : Intrinsic, se3 : np.ndarray):
        self.intrinsic = intrinsic
        self.se3 = se3
        self.se3_inv = SE3.invert(se3)

    def to_camera_space(self,point_3D_world):
        return np.matmul(self.se3,point_3D_world)

    def to_world_space(self,point_3D_camera):
        return np.matmul(self.se3_inv, point_3D_camera)

    def apply_intrinsics(self,point_3D_camera):
        return np.matmul(self.intrinsic.K,point_3D_camera)

    def apply_perspective_projection_to_camera(self,point_3D_camera):
        point_persp = np.matmul(self.intrinsic.K,point_3D_camera)
        return np.array([point_persp[0]/point_persp[2],point_persp[1]/point_persp[2],1])

    def perspective_pipeline(self):
        return np.matmul(self.intrinsic.K,self.se3)

    def apply_perspective_pipeline(self,point_3D_world):
        persp = np.matmul(self.perspective_pipeline(),point_3D_world)
        (dim,N) = persp.shape
        for i in range(0,N):
            persp[:,i] = [persp[0,i]/persp[2,i],persp[1,i]/persp[2,i],1]
        return persp




