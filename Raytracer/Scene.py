import numpy as np
import Camera.Camera as Camera
import Numerics.SE3 as SE3
import Numerics.Utils as Utils
import Raytracer.Geometry as Geometry
from math import pi, fabs


def phong_shading(light_ws, position_ws, normal):
    view = Utils.normalize(light_ws - position_ws)
    return Utils.fast_dot(view,normal)

class Scene:
    def __init__(self,x_resolution,y_resolution,spheres,camera : Camera):
        self.resolution = (y_resolution,x_resolution)
        self.frame_buffer = np.zeros(self.resolution,dtype=np.float64)
        self.depth_buffer = np.zeros(self.resolution,dtype=np.float64)
        self.spheres = spheres
        self.camera = camera
        self.fov = pi / 3
        self.light_ws = np.array([0,0,10]).reshape((3,1))

    def render(self):
        (height,width) = self.resolution
        for x in range(0,width):
            for y in range(0,height):
                ray_direction_camera_space = self.camera.camera_ray_direction_camera_space(x,y,width,height,self.fov)
                camera_to_world = self.camera.se3_inv
                rot = SE3.extract_rotation(camera_to_world)
                ray_world_space = Utils.normalize(np.matmul(rot,ray_direction_camera_space))
                ray = Geometry.Ray(self.camera.origin_ws[0:3],ray_world_space)
                (b,t,sphere) = self.find_closest_intersection(ray)
                if sphere.is_intersection_acceptable(b,t):
                    intersection_point = Geometry.point_for_ray(ray,t)
                    normal = sphere.normal_for_point(intersection_point)
                    depth = intersection_point[2]
                    self.depth_buffer[y,x] = fabs(depth)
                    self.frame_buffer[y,x] = phong_shading(self.light_ws,intersection_point,normal)


    def find_closest_intersection(self,ray):
        real_solution_exists = False
        t_min = Geometry.t_max
        sphere_best = Geometry.empty_sphere
        for sphere in self.spheres:
            (b,t) = sphere.intersect(ray)
            if sphere.is_intersection_acceptable(b,t) and t < t_min:
                t_min = t
                sphere_best = sphere
                real_solution_exists = True

        return real_solution_exists , t_min , sphere_best







