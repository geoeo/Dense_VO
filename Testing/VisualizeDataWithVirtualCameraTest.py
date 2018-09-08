import Numerics.Generator as Generator
import Visualization.Plot3D as Plot3D
import numpy as np
import Numerics.Utils as Utils
import Camera.Camera as Camera
from math import pi


N = 20

(X,Y,Z) = Generator.generate_3d_plane(1, 1, -30, N, 4)
H = np.repeat(1,N)

points = np.transpose(np.array(list(map(lambda x: list(x),list(zip(X,Y,Z,H))))))

SE3 = Generator.generate_random_se3(-5, 5, pi / 18, pi / 10, 0, pi / 10)

perturbed_points_gt = np.matmul(SE3, points)

camera = Camera.normalized_camera()

points_persp = camera.apply_perspective_pipeline(points)

#SE3_est = Solver.solve_SE3(points,perturbed_points_gt,20000,0.01)

#perturbed_points_est = np.matmul(SE3_est,points)

(X_orig,Y_orig,Z_orig) = list(Utils.points_into_components(points))
(X_new,Y_new,Z_new) = list(Utils.points_into_components(perturbed_points_gt))
#(X_est,Y_est,Z_est) = list(Utils.points_into_components(perturbed_points_est))
(X_persp,Y_persp,Z_persp) = list(Utils.points_into_components(points_persp))


#Plot3D.scatter_plot([(X_orig,Y_orig,Z_orig),(X_new,Y_new,Z_new),(X_est,Y_est,Z_est)],['original','perturbed','estimate'])
Plot3D.scatter_plot_sub([(X_orig,Y_orig,Z_orig)],[(X_persp,Y_persp,Z_persp)],['original'],['projected'])



