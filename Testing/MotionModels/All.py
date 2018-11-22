from MotionModels import Ackermann, SteeringCommand, MotionDelta, Pose
from Numerics import Utils, SE3
import matplotlib.pyplot as plt
from Visualization import Plot3D
import numpy as np
import math

X, Y, Z = [0,0], [0,0], [0,-1]
H = np.repeat(1,2)

origin = Utils.to_homogeneous_positions(X, Y, Z, H)

dt = 1.0
steering_command_straight = SteeringCommand.SteeringCommands(1.5, 0.0)
steering_commands = [steering_command_straight, steering_command_straight]
dt_list = list(map(lambda _: dt,steering_commands))
ackermann_motion = Ackermann.Ackermann(steering_commands, dt_list)


cov_list = ackermann_motion.covariance_dead_reckoning_for_command_list(steering_commands,dt_list)
ellipse_factor_list = Ackermann.get_standard_deviation_factors_from_covaraince_list(cov_list)


# TODO put this in visualizer
se3_list = SE3.generate_se3_from_motion_delta_list(ackermann_motion.motion_delta_list)
motion_delta = ackermann_motion.motion_delta_list[0]
se3 = SE3.generate_se3_from_motion_delta(motion_delta)
origin_transformed = np.matmul(se3, origin)
origin_transformed_2 = np.matmul(se3, origin_transformed)

points = np.append(np.append(origin, origin_transformed, axis=1), origin_transformed_2, axis=1)
points_xyz = points[0:3,:]

# testing inverse
#cov_inv = np.linalg.inv(motion_cov_small_1)
#t = np.matmul(cov_inv,motion_cov_small_1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#Plot3D.plot_wireframe_ellipsoid(1,1,1,ax,label_axes=True, clear=True,draw=False)
Plot3D.plot_array_lines(points_xyz, ax,clear=True,draw=False)
Plot3D.plot_wireframe_ellipsoid(0.1, ellipse_factor_list, se3_list , ax, label_axes=True, clear=False,draw=False)
ax.set_xlim(-10, 10)
ax.set_ylim(-1, 1)
ax.set_zlim(-10, 10)
Plot3D.show()



