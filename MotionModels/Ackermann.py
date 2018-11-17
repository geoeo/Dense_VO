from MotionModels.MotionDelta import MotionDelta
from MotionModels.SteeringCommand import SteeringCommands
from Numerics.Utils import matrix_data_type
import numpy as np
import math


# input are the eigen values of the covariance's EVD
def get_standard_deviation_factors_for_projection(w):
    z = math.sqrt(w[2])
    x = math.sqrt(w[1])
    return x, z


def get_standard_deviation_factors_for_projection_for_list(w_list):
    return list(map(lambda x: get_standard_deviation_factors_for_projection(x),w_list))


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

    def ackermann_dead_reckoning_for_list(self, steering_input_list : [SteeringCommands]):
        return list(map(lambda x: self.ackermann_dead_reckoning(x), steering_input_list))


    # TODO test
    # Every row of the covariance has 3 entries
    # The order of values is [x,y,yaw] ([-z,x,yaw])
    # i.e. cov[0,1] = dx/dy , cov[2,1] = dpitch/dy
    def covariance_dead_reckoning(self, steering_input : SteeringCommands, theta, dt):

        steering_vel_cmd = steering_input.linear_velocity
        steering_angle_cmd = steering_input.steering_angle

        self.G[0, 2] = -steering_vel_cmd * math.sin(theta) * dt
        self.G[1, 2] = steering_vel_cmd * math.cos(theta) * dt

        self.V[0, 0] = math.cos(theta) * dt
        self.V[1, 0] = math.sin(theta) * dt
        self.V[2, 0] = (math.tan(steering_angle_cmd) / self.wheel_base) * dt
        self.V[2, 1] = (steering_vel_cmd * dt) / (pow(math.cos(steering_angle_cmd), 2.0) * self.wheel_base)

        G_t = np.transpose(self.G)
        V_t = np.transpose(self.V)

        cov_est = np.matmul(self.G,np.matmul(self.covariance_prev,G_t)) + np.matmul(self.V,np.matmul(self.M,V_t))

        # set for next iteration
        self.covariance_prev = np.copy(cov_est)

        return cov_est


    def covariance_dead_reckoning_for_command_list(self, pose, steering_input_list, motion_delta_list, dt_list):
        steering_list_len = len(steering_input_list)
        assert steering_list_len == len(dt_list)
        assert steering_list_len == len(motion_delta_list)

        cov_list = []

        commands = zip(steering_input_list,motion_delta_list,dt_list)

        for (steering_command,motion_delta,dt) in commands:
            pose.apply_motion(motion_delta, dt)
            # TODO investigate which theta to use
            # this might actually be better since we are interested in the uncertainty only in this timestep
            # theta = motion_delta.delta_theta
            # traditional uses accumulated theta
            theta = pose.theta
            motion_cov = self.covariance_dead_reckoning(steering_command, theta, dt)
            cov_list.append(motion_cov)

        return cov_list


    def generate_6DOF_cov_from_motion_model_cov(self,cov_small):
        # copy into 6Dof Covariance
        self.covariance_current_large[self.x_offset,self.x_offset] = cov_small[1,1]
        self.covariance_current_large[self.x_offset,self.z_offset] = cov_small[1,0]
        self.covariance_current_large[self.x_offset,self.pitch_offset] = cov_small[1,2]

        self.covariance_current_large[self.z_offset,self.x_offset] = cov_small[0,1]
        self.covariance_current_large[self.z_offset,self.z_offset] = cov_small[0,0]
        self.covariance_current_large[self.z_offset,self.pitch_offset] = cov_small[0,2]

        self.covariance_current_large[self.pitch_offset,self.x_offset] = cov_small[2,1]
        self.covariance_current_large[self.pitch_offset,self.z_offset] = cov_small[2,0]
        self.covariance_current_large[self.pitch_offset,self.pitch_offset] = cov_small[2,2]

    def generate_6DOF_cov_from_motion_model_cov_list(self,cov_small_list):
        return list(map(lambda x: self.generate_6DOF_cov_from_motion_model_cov(x), cov_small_list))

