from MotionModels.MotionDeltaRobot import MotionDeltaRobot
from MotionModels.SteeringCommand import SteeringCommands
from MotionModels.Pose import Pose
from Numerics.Utils import matrix_data_type
from Numerics import Utils
import numpy as np
import math

x_offset = 0
z_offset = 2
pitch_offset = 4


# input are the eigen values of the covariance's EVD
def get_standard_deviation_factors_for_projection(w):
    z = math.sqrt(w[2])
    x = math.sqrt(w[1])
    return x, z


def get_standard_deviation_factors_for_projection_for_list(w_list):
    return list(map(lambda x: get_standard_deviation_factors_for_projection(x),w_list))

def get_standard_deviation_factors_from_covaraince_list(cov_list):
    evd_list = Utils.covariance_eigen_decomp_for_list(cov_list)
    eigen_value_list = Utils.eigen_values_from_evd_list(evd_list)
    return get_standard_deviation_factors_for_projection_for_list(eigen_value_list)


# copy into 6Dof Covariance
# 6 DOF row = {x,y,z,roll,pitch,yaw}
def generate_6DOF_cov_from_motion_model_cov(cov_small):

    covariance_current_large = np.identity(6, dtype=matrix_data_type)

    #for i in range(0,6):
    #    covariance_current_large[i,i] = Utils.covariance_zero

    covariance_current_large[x_offset,x_offset] = cov_small[1,1]
    covariance_current_large[x_offset,z_offset] = cov_small[1,0]
    covariance_current_large[x_offset,pitch_offset] = cov_small[1,2]

    covariance_current_large[z_offset,x_offset] = cov_small[0,1]
    covariance_current_large[z_offset,z_offset] = cov_small[0,0]
    covariance_current_large[z_offset,pitch_offset] = cov_small[0,2]

    covariance_current_large[pitch_offset,x_offset] = cov_small[2,1]
    covariance_current_large[pitch_offset,z_offset] = cov_small[2,0]
    covariance_current_large[pitch_offset,pitch_offset] = cov_small[2,2]

    return covariance_current_large

def generate_6DOF_cov_from_motion_model_cov_list(cov_small_list):
    return list(map(lambda x: generate_6DOF_cov_from_motion_model_cov(x), cov_small_list))



class Ackermann:

    def __init__(self, steering_command_list, dt_list):

        self.wheel_base = 0.255 # meters
        self.wheel_diameter = 0.0269 # meters
        self.linear_velocity_noise = 0.0
        self.steering_angle_noise = 0.0
        self.covariance_prev = np.identity(3, dtype=matrix_data_type)
        self.M = np.identity(2,dtype=matrix_data_type) # noise parameters
        self.G = np.identity(3, dtype=matrix_data_type)
        self.V = np.zeros((3,2), dtype=matrix_data_type)
        self.pose = Pose()
        self.steering_command_list = steering_command_list
        self.dt_list = dt_list
        self.cov_list = None # Will be populated when the motion model is run
        self.pose_delta_list = [] # Will be populated when the motion model is run
        #self.pose_list = None # Will be populated when the motion model is run



    def ackermann_dead_reckoning_delta(self, steering_input : SteeringCommands):

        new_motion_delta = MotionDeltaRobot()
        linear_velocity = 0.0
        steering_angle = 0.0

        steering_vel_cmd = steering_input.linear_velocity
        steering_angle_cmd = -steering_input.steering_angle

        if math.fabs(steering_vel_cmd) > self.linear_velocity_noise:
            linear_velocity = steering_vel_cmd
        if math.fabs(steering_angle_cmd) > self.steering_angle_noise:
            steering_angle = steering_angle_cmd

        new_motion_delta.delta_theta = linear_velocity * math.tan(steering_angle) / self.wheel_base
        new_motion_delta.delta_x = linear_velocity
        # wheel diameter TODO write about this
        new_motion_delta.delta_x *= 0.0269*math.pi

        return new_motion_delta

    def set_ackermann_dead_reckoning_for_list(self, steering_input_list : [SteeringCommands], dt_list):
        steering_list_length = len(steering_input_list)
        dt_list_length = len(dt_list)

        assert steering_list_length == dt_list_length
        theta_prev = 0.0

        for i in range(0,steering_list_length):
            dt = dt_list[i]
            steering_cmd = steering_input_list[i]
            pose = Pose()
            motion_delta_robot = self.ackermann_dead_reckoning_delta(steering_cmd)
            pose.apply_world_motion(motion_delta_robot,dt,theta_prev)
            self.pose_delta_list.append(pose)
            theta_prev += pose.delta_theta


    # TODO test
    # Every row of the covariance has 3 entries
    # The order of values is [x,y,yaw] ([-z,x,yaw])
    # i.e. cov[0,1] = dx/dy , cov[2,1] = dpitch/dy
    def covariance_dead_reckoning(self, steering_input : SteeringCommands, theta, dt):

        steering_vel_cmd = steering_input.linear_velocity
        steering_angle_cmd = -steering_input.steering_angle

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


    def covariance_dead_reckoning_for_command_list(self, steering_input_list, dt_list):
        self.set_ackermann_dead_reckoning_for_list(steering_input_list, dt_list)

        steering_list_len = len(steering_input_list)
        dt_list_len = len(dt_list)
        dt_steering_list_diff = dt_list_len - steering_list_len
        if dt_steering_list_diff > 0:
            dt_list = dt_list[:-dt_steering_list_diff]
        assert steering_list_len == len(dt_list)
        assert steering_list_len == len(self.pose_delta_list)

        cov_list = []

        commands = zip(steering_input_list, self.pose_delta_list, dt_list)

        for (steering_command,motion_delta,dt) in commands:
            self.pose.apply_motion(motion_delta, dt)
            # TODO investigate which theta to use
            # this might actually be better since we are interested in the uncertainty only in this timestep
            theta = motion_delta.delta_theta
            # traditional uses accumulated theta
            #theta = self.pose.theta
            motion_cov = self.covariance_dead_reckoning(steering_command, theta, dt)
            cov_list.append(motion_cov)

        return cov_list

