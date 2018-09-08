import Numerics.Generator as Generator
import Visualization.Plot3D as Plot3D
import numpy as np
import Numerics.Utils as NumUtils
import VisualOdometry.Solver as Solver
from math import pi

N = 100
(X,Y,Z) = Generator.generate_3d_plane(1, 1, -10, N, 4)
H = np.repeat(1,N)

points = np.transpose(np.array(list(map(lambda x: list(x),list(zip(X,Y,Z,H))))))

SE3 = Generator.generate_random_se3(-10, 10, 4, pi / 2, 0, pi / 2)

rotated_points_gt = np.matmul(SE3,points)

(X_gt,Y_gt,Z_gt) = list(NumUtils.points_into_components(rotated_points_gt))

SE3_est = Solver.solve_SE3(points, rotated_points_gt, 20000, 0.01)

rotated_points_est = np.matmul(SE3_est,points)

(X_est,Y_est,Z_est) = list(NumUtils.points_into_components(rotated_points_est))

Plot3D.scatter_plot([(X,Y,Z),(X_gt,Y_gt,Z_gt),(X_est,Y_est,Z_est)],['original','ground truth','estimate'])



