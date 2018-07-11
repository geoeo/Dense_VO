import numpy as np
import Camera.Camera as Camera
import Numerics.SE3 as SE3
import Numerics.Utils as Utils
import Raytracer.Ray as Ray
import Raytracer.Geometry as Geometry


class Scene:
    def __init__(self,x_resolution,y_resolution,number_of_spheres,camera : Camera):
        self.resolution = (y_resolution,x_resolution)
        self.frame_buffer = np.zeros(self.resolution,dtype=np.float64)
        self.depth_buffer = np.zeros(self.resolution,dtype=np.float64)
        self.spheres = np.zeros((4,1,number_of_spheres))
        self.camera = camera

    def render(self):
        (height,width) = self.resolution
        for x in range(0,width):
            for y in range(0,height):
                ray_direction_camera_space = self.camera.camera_ray_direction_camera_space(x,y)
                camera_to_world = self.camera.se3_inv
                rot = SE3.extract_rotation(camera_to_world)
                ray_world_space = Utils.normalize(np.matmul(rot,ray_direction_camera_space))
                ray = Ray.Ray(self.camera.origin_ws,ray_world_space)

    def find_closest_intersection(self,x,y,ray):
        real_solution_exists = False
        t = -1
        sphere = Geometry.Sphere(np.zeros(3,1),1.0)




