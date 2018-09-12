import Numerics.Generator as Generator
import Visualization.Plot3D as Plot3D

(X,Y,Z) = Generator.generate_3d_plane(1, 1, 0, 20, 4)

Plot3D.scatter_plot(X,Y,Z)



