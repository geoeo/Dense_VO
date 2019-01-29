from MotionModels.MotionDeltaRobot import MotionDeltaRobot
from Numerics import Utils
import math

class Pose:

    def __init__(self):
        self.x = 0
        self.y = 0
        self.theta = 0

    def apply_motion(self, motion_delta : MotionDeltaRobot, dt):
        self.x += motion_delta.delta_x * dt
        self.y += motion_delta.delta_y * dt
        self.theta += motion_delta.delta_theta * dt

    def apply_world_motion(self, motion_delta_robot, dt):
        delta_x_r = motion_delta_robot.delta_x
        delta_y_r = motion_delta_robot.delta_y
        delta_theta_r = motion_delta_robot.delta_theta

        self.delta_x = (delta_x_r*math.cos(delta_theta_r) - delta_y_r*math.sin(delta_theta_r))*dt
        self.delta_y = (delta_x_r*math.sin(delta_theta_r) + delta_y_r*math.cos(delta_theta_r))*dt
        self.delta_theta = delta_theta_r*dt


    def get_6dof_twist(self, normalize=False):
        twist = np.array([[self.delta_y],[0],[self.delta_x],[0],[self.delta_theta],[0]],dtype=Utils.matrix_data_type)
        if normalize:
            twist /= np.linalg.norm(twist)
        return twist