import matplotlib.pyplot as plt
from Visualization import Plot3D
from Numerics import Generator, Utils
import numpy as np
import math

#X, Y, Z = [0,0,10,10] , [0,0,10,10] , [0,-1,0,-1]
X, Y, Z = [0,0], [0,0], [0,-1]
H = np.repeat(1,2)

pair = Utils.to_homogeneous_positions(X, Y, Z, H)

se3 = Generator.generate_random_se3(2,2,math.pi,math.pi/2,0,0)
se3_2 = Generator.generate_random_se3(2,2,math.pi,math.pi/2,0,0)

pair_transformed = np.matmul(se3,pair)
pair_transformed_2 = np.matmul(se3_2,pair)

points = np.append(pair,pair_transformed,axis=1)
points_xyz = points[0:3,:]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

Plot3D.plot_array_lines(points_xyz, ax)
pair_transformed = np.matmul(se3,pair)
