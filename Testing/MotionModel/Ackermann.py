from MotionModels import Ackermann, SteeringCommands, MotionDelta, Pose
import numpy as np
from Numerics import Utils, SE3
import matplotlib.pyplot as plt
from Visualization import Plot3D

X, Y, Z = [0,0], [0,0], [0,-1]
H = np.repeat(1,2)

origin = Utils.to_homogeneous_positions(X, Y, Z, H)

dt = 1.0
pose = Pose.Pose()
steering_command_straight = SteeringCommands.SteeringCommands(1.0,0.0)

ackermann_motion = Ackermann.Ackermann()

new_motion_delta = ackermann_motion.ackermann_dead_reckoning(steering_command_straight)
pose.apply_motion(new_motion_delta,dt)

#TODO investigate which theta to use
# this might actually be better since we are interested in the uncertainty only in this timestep
theta = new_motion_delta.delta_theta
# traditional uses accumulated theta
#theta = pose.theta

motion_cov = ackermann_motion.covariance_dead_reckoning(steering_command_straight,theta,dt)

se3 = SE3.generate_se3_from_motion_delta(new_motion_delta)
origin_transformed = np.matmul(se3, origin)

points = np.append(origin, origin_transformed, axis=1)
points_xyz = points[0:3,:]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

Plot3D.plot_array_lines(points_xyz, ax)


