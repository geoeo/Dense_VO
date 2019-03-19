from MotionModels.MotionDeltaRobot import MotionDeltaRobot
from Numerics import Utils
import numpy as np
import math

class PoseLinear:

    def __init__(self):

        self.delta_x = 0.0
        self.delta_y = 0.0
        self.delta_z = 0.0

        self.delta_v_x = 0.0
        self.delta_v_y = 0.0
        self.delta_v_z = 0.0

    def apply_world_motion(self, linear_delta_robot, dt, ax, ay, az):
        delta_x = linear_delta_robot.delta_x
        delta_y = linear_delta_robot.delta_y
        delta_z = linear_delta_robot.delta_z

        delta_v_x = linear_delta_robot.delta_v_x
        delta_v_y = linear_delta_robot.delta_v_y
        delta_v_z = linear_delta_robot.delta_v_z

        self.delta_x = delta_v_x*dt + 0.5*ax*dt*dt
        self.delta_y = delta_v_y*dt + 0.5*ay*dt*dt
        self.delta_z = delta_v_z*dt + 0.5*az*dt*dt

        self.v_x = ax*dt
        self.v_y = ay*dt
        self.v_z = az*dt


    def get_6dof_twist(self, normalize=False):
        twist = np.array([[self.delta_x],[self.delta_y],[-self.delta_z],[0],[0],[0]],dtype=Utils.matrix_data_type)
        if normalize:
            twist /= np.linalg.norm(twist)
        return twist