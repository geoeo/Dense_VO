from MotionModels import Ackermann, SteeringCommand, MotionDeltaRobot, Pose
from Numerics import Utils, SE3
import matplotlib.pyplot as plt
from Visualization import Plot3D
import numpy as np
import math

X, Y, Z = [0,0], [0,0], [0,-1]
H = np.repeat(1,2)

origin = Utils.to_homogeneous_positions(X, Y, Z, H)

dt = 1.0
pose = Pose.Pose()
steering_command_straight = SteeringCommand.SteeringCommands(1.5, 0.0)

ackermann_motion = Ackermann.Ackermann()

new_motion_delta = ackermann_motion.ackermann_dead_reckoning_delta(steering_command_straight)
pose.apply_motion(new_motion_delta,dt)
pose.apply_motion(new_motion_delta,dt)
pose.apply_motion(new_motion_delta,dt)

#TODO investigate which theta to use
# this might actually be better since we are interested in the uncertainty only in this timestep
#theta = new_motion_delta.delta_theta
# traditional uses accumulated theta
theta = pose.theta

motion_cov_small = ackermann_motion.covariance_dead_reckoning(steering_command_straight,theta,dt)

se3 = SE3.generate_se3_from_motion_delta(new_motion_delta)
origin_transformed = np.matmul(se3, origin)

w,v = Utils.covariance_eigen_decomp(motion_cov_small)
z_factor, x_factor = Ackermann.get_standard_deviation_factors_for_projection(w)

change_of_basis = np.identity(4,dtype=Utils.matrix_data_type)
#change_of_basis = SE3.rotation_around_x(-math.pi/2)

# testing inverse
cov_inv = np.linalg.inv(motion_cov_small)
t = np.matmul(cov_inv,motion_cov_small)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#Plot3D.plot_wireframe_ellipsoid(1,1,1,ax,label_axes=True, clear=True,draw=False)
Plot3D.plot_wireframe_ellipsoid(0.1, [(x_factor,z_factor)], [change_of_basis], ax, label_axes=True, clear=False,draw=True)


