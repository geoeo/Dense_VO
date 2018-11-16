from MotionModels.MotionDelta import MotionDelta
from MotionModels.SteeringCommands import SteeringCommands
from Numerics.Utils import matrix_data_type
import numpy as np
import math

def get_ackermann_eigenvalue_factors_for_projection(w):
    z = w[2]
    x = w[1]
    return z, x


class Ackermann:

    def __init__(self):

        self.wheel_base = 0.255 # meters
        self.linear_velocity_noise = 0.0
        self.steering_angle_noise = 0.0
        self.covariance_prev = np.identity(3, dtype=matrix_data_type)
        # 6 DOF row = {x,y,z,roll,pitch,yaw}
        self.covariance_current_large = np.identity(6, dtype=matrix_data_type)
        self.M = np.identity(2,dtype=matrix_data_type) # noise parameters
        self.G = np.identity(3, dtype=matrix_data_type)
        self.V = np.zeros((3,2), dtype=matrix_data_type)
        self.x_offset = 0
        self.z_offset = 2
        self.pitch_offset = 4


    # TODO test
    def ackermann_dead_reckoning(self, steering_input : SteeringCommands):

        new_motion_delta = MotionDelta()
        linear_velocity = 0.0
        steering_angle = 0.0

        steering_vel_cmd = steering_input.linear_velocity
        steering_angle_cmd = steering_input.steering_angle

        if math.fabs(steering_vel_cmd) > self.linear_velocity_noise:
            linear_velocity = steering_vel_cmd
        if math.fabs(steering_angle_cmd) > self.steering_angle_noise:
            steering_angle = steering_angle_cmd

        new_motion_delta.delta_theta = linear_velocity * math.tan(steering_angle) / self.wheel_base
        new_motion_delta.delta_x = linear_velocity

        return new_motion_delta

    # TODO test
    # Every row of the covariance has 3 entries
    # The order of values is [x,y,yaw] ([-z,x,yaw])
    # i.e. cov[0,1] = dx/dy , cov[2,1] = dpitch/dy
    def covariance_dead_reckoning(self, steering_input : SteeringCommands, theta, dt):

        steering_vel_cmd = steering_input.linear_velocity
        steering_angle_cmd = steering_input.steering_angle

        #self.G[0, 0] = 1.0
        self.G[0, 2] = -steering_vel_cmd * math.sin(theta) * dt
        #self.G[1, 1] = 1.0
        self.G[1, 2] = steering_vel_cmd * math.cos(theta) * dt
        #self.G[2, 2] = 1.0

        self.V[0, 0] = math.cos(theta) * dt
        self.V[1, 0] = math.sin(theta) * dt
        self.V[2, 0] = (math.tan(steering_angle_cmd) / self.wheel_base) * dt
        self.V[2, 1] = (steering_vel_cmd * dt) / (pow(math.cos(steering_angle_cmd), 2.0) * self.wheel_base)

        G_t = np.transpose(self.G)
        V_t = np.transpose(self.V)

        cov_est = np.matmul(self.G,np.matmul(self.covariance_prev,G_t)) + np.matmul(self.V,np.matmul(self.M,V_t))

        # set for next iteration
        self.covariance_prev = np.copy(cov_est)

        # Clear
        #for j in range(0,3):
        #    for i in range(0,3):
        #        self.G[j,i] = 0
        #    for i_2 in range (0,2):
        #        self.V[j,i_2] = 0

        new_cov = cov_est

        # TODO investigate coordiante system change
        # copy into 6Dof Covariance
        self.covariance_current_large[self.x_offset,self.x_offset] = new_cov[1,1]
        self.covariance_current_large[self.x_offset,self.z_offset] = new_cov[1,0]
        self.covariance_current_large[self.x_offset,self.pitch_offset] = new_cov[1,2]

        self.covariance_current_large[self.z_offset,self.x_offset] = new_cov[0,1]
        self.covariance_current_large[self.z_offset,self.z_offset] = new_cov[0,0]
        self.covariance_current_large[self.z_offset,self.pitch_offset] = new_cov[0,2]

        self.covariance_current_large[self.pitch_offset,self.x_offset] = new_cov[2,1]
        self.covariance_current_large[self.pitch_offset,self.z_offset] = new_cov[2,0]
        self.covariance_current_large[self.pitch_offset,self.pitch_offset] = new_cov[2,2]

        return new_cov, self.covariance_current_large

