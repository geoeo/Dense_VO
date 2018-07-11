import numpy as np
import Camera.Intrinsic as Intrinsic
import Numerics.SE3 as SE3
import Numerics.Utils as Utils
from math import tan
import warnings


# https://referencesource.microsoft.com/#System.Numerics/System/Numerics/Matrix4x4.cs,b82966e485b5a306
def look_at_matrix(camera_position: np.ndarray, camera_target: np.ndarray, camera_up: np.ndarray):
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


def normalized_camera():
    look_at = look_at_matrix(np.array([0, 0, 0]), np.array([0, 0, -1]), np.array([0, 1, 0]))
    intrinsics = Intrinsic.Intrinsic(-1, -1, 0, 0)
    return Camera(intrinsics, look_at)



"""An Object which encodes a Camera.

Attributes:
  intrinsic: An object of type IntrnsicsCamera
  extrinsic: An object of type SE3 
"""
class Camera:
    def __init__(self, intrinsic : Intrinsic, se3 : np.ndarray):
        if intrinsic.extract_fx() > 0 or intrinsic.extract_fy() > 0:
            warnings.warn("focal length is positive in right handed coordinate system, may lead to inverted image", RuntimeWarning)

        self.intrinsic = intrinsic
        self.se3 = se3
        self.se3_inv = SE3.invert(se3)
        self.origin_ws = np.matmul(self.se3_inv,np.array([0,0,0,1]).reshape((4,1)))

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

    #TODO Test camera
    # ndc is [0, 1] from top left
    def pixel_to_ndc(self,px,py,width,height):
        return (px + 0.5)/width , (py+0.5)/height

    # screen space is [-1,1] from top left
    def ndc_to_screen(self,x_ndc,y_ndc):
        return 2.0*x_ndc-1.0 , 2.0*y_ndc-1.0

    def aspect_ratio(self,width,height):
        return width/height

    # pixel coordiantes / sample points (X,Y) in camera space (3D)
    # focal length is implicit in the tan(alpha / 2) calc
    def screen_to_camera(self,x_screen,y_screen,aspect_ratio,fov):
        alpha = fov/2
        return x_screen*aspect_ratio*tan(alpha) , x_screen*aspect_ratio*tan(alpha)

    def pixel_to_camera(self,x,y,width,height,fov):
        (ndc_x,ndc_y) = self.pixel_to_ndc(x,y,width,height)
        aspect_ratio = self.aspect_ratio(width,height)
        (screen_x,screen_y) = self.ndc_to_screen(ndc_x,ndc_y)
        return self.screen_to_camera(screen_x,screen_y,aspect_ratio,fov)

    # assumes normalized camera with focal length 1 looking towards z = -1
    def camera_ray_direction_camera_space(self,camera_pixel_x,camera_pixel_y):
        ray_cs = np.array([camera_pixel_x,camera_pixel_y,-1]).reshape(3,1)
        return Utils.normalize(ray_cs)







