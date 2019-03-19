from MotionModels.LinearDeltaRobot import LinearDeltaRobot
from MotionModels.AccelerationCommand import AccelerationCommand
from MotionModels.PoseConstant import PoseConstant
from Numerics.Utils import matrix_data_type
from Numerics import Utils
import numpy as np
import math

x_offset = 0
y_offset = 1
z_offset = 2
pitch_offset = 4


#TODO maybe not necessary
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
def generate_6DOF_cov_from_motion_model_cov(cov_linear):

    covariance_current_large = np.identity(6, dtype=matrix_data_type)

    covariance_current_large[x_offset,x_offset] = cov_linear[0,0]
    covariance_current_large[x_offset,y_offset] = cov_linear[0,1]
    covariance_current_large[x_offset,z_offset] = cov_linear[0,2]

    covariance_current_large[y_offset,x_offset] = cov_linear[1,0]
    covariance_current_large[y_offset,y_offset] = cov_linear[1,1]
    covariance_current_large[y_offset,z_offset] = cov_linear[1,2]

    covariance_current_large[z_offset,x_offset] = cov_linear[2,0]
    covariance_current_large[z_offset,y_offset] = cov_linear[2,1]
    covariance_current_large[z_offset,z_offset] = cov_linear[2,2]

    return covariance_current_large

def generate_6DOF_cov_from_motion_model_cov_list(cov_small_list):
    return list(map(lambda x: generate_6DOF_cov_from_motion_model_cov(x), cov_small_list))



class Ackermann:

    def __init__(self, steering_command_list, dt_list):

        self.covariance_prev = np.identity(6, dtype=matrix_data_type)
        self.G = np.identity(6, dtype=matrix_data_type)
        self.V = np.zeros((6,4), dtype=matrix_data_type)
        #self.pose = Pose()
        self.steering_command_list = steering_command_list
        self.dt_list = dt_list
        self.cov_list = None # Will be populated when the motion model is run
        self.pose_delta_list = [] # Will be populated when the motion model is run
        #self.pose_list = None # Will be populated when the motion model is run



    def set_linear_velocity_for_list(self, acc_input_list : [AccelerationCommand], dt_list):
        acc_list_length = len(acc_input_list)
        dt_list_length = len(dt_list)

        assert acc_list_length == dt_list_length
        pose_prev = PoseConstant()

        for i in range(0,acc_list_length):
            dt = dt_list[i]
            acc_cmd = acc_input_list[i]
            pose = PoseConstant()
            pose.apply_world_motion(pose_prev, dt, acc_cmd.ax, acc_cmd.ay, acc_cmd.az)
            self.pose_delta_list.append(pose)
            pose_prev = pose


    def covariance_dead_reckoning(self, acceleration_input : AccelerationCommand, pose_prev, dt):

        ax = acceleration_input.ax
        ay = acceleration_input.ay
        az = acceleration_input.az

        vx = pose_prev.v_x
        vy = pose_prev.v_y
        vz = pose_prev.v_z

        self.G[0, 4] = dt
        self.G[1, 5] = dt
        self.G[2, 6] = dt

        self.V[0, 0] = vx + ax*dt
        self.V[0, 1] = 0.5*dt*dt
        self.V[1, 0] = vy + ay*dt
        self.V[1, 1] = 0.5*dt*dt
        self.V[2, 0] = vz + az*dt
        self.V[2, 1] = 0.5*dt*dt

        self.V[3, 0] = ax
        self.V[3, 1] = dt
        self.V[4, 0] = ay
        self.V[4, 1] = dt
        self.V[5, 0] = az
        self.V[5, 1] = dt

        G_t = np.transpose(self.G)
        V_t = np.transpose(self.V)

        cov_est = np.matmul(self.G,np.matmul(self.covariance_prev,G_t)) + np.matmul(self.V,V_t)

        # set for next iteration
        self.covariance_prev = np.copy(cov_est)

        return cov_est


    def covariance_dead_reckoning_for_command_list(self, acc_input_list, dt_list):
        self.set_linear_velocity_for_list(acc_input_list, dt_list)

        acc_list_len = len(acc_input_list)
        dt_list_len = len(dt_list)
        dt_steering_list_diff = dt_list_len - acc_list_len
        if dt_steering_list_diff > 0:
            dt_list = dt_list[:-dt_steering_list_diff]
        assert acc_list_len == len(dt_list)
        assert acc_list_len == len(self.pose_delta_list)

        cov_list = []

        commands = zip(acc_input_list, self.pose_delta_list, dt_list)

        for (steering_command,motion_delta,dt) in commands:
            #self.pose.apply_motion(motion_delta, dt)
            # TODO investigate which theta to use
            # this might actually be better since we are interested in the uncertainty only in this timestep
            theta = motion_delta.delta_theta
            # traditional uses accumulated theta
            #theta = self.pose.theta
            motion_cov = self.covariance_dead_reckoning(steering_command, theta, dt)
            cov_list.append(motion_cov)

        return cov_list

