from MotionModels.MotionDelta import MotionDelta
from MotionModels.SteeringCommands import SteeringCommands
from Numerics.Utils import matrix_data_type
import numpy as np


class Ackermann:

    def __init__(self):

        self.linear_velocity_noise = 0.0
        self.steering_angle_noise =  0.0
        self.covariance_prev = np.identity(3, dtype=matrix_data_type)

    # TODO
    def ackermann_dead_reckoning(self, input : SteeringCommands):

        new_motion = MotionDelta()
        linear_velocity = 0.0
        steering_angle = 0.0


    #TODO
    def covariance_dead_reckoning(self):

        G = np.zeros((3,3), dtype=matrix_data_type)
        V = np.zeros((3,3), dtype=matrix_data_type)