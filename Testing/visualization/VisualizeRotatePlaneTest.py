import Numerics.Generator as Generator
import Visualization.Plot3D as Plot3D
import numpy as np
import Numerics.Utils as NumUtils

(X,Y,Z) = Generator.generate_3d_plane(1, 1, -30, 20, 4)
H = np.repeat(1,20)

points = np.transpose(np.array(list(map(lambda x: list(x),list(zip(X,Y,Z,H))))))

SE3 = Generator.generate_random_se3(-1, 1, 4, 0, 0, 0)

rotated_points = np.matmul(SE3,points)

(X_new,Y_new,Z_new) = list(NumUtils.points_into_components(rotated_points))

#Plot3D.scatter_plot(X_new,Y_new,Z_new)
Plot3D.scatter_plot_array([(X,Y,Z),(X_new,Y_new,Z_new)],['original','perturbed'])
#Plot3D.plot_array_lines([(X,Y,Z),(X_new,Y_new,Z_new)])




