import numpy as np
from Numerics import Utils


class MotionDelta:

    # In Ackermann space
    def __init__(self):
        self.delta_x = 0.0
        self.delta_y = 0.0
        self.delta_theta = 0.0

    def get_6dof_twist(self, normalize=False):
        twist = np.array([[-self.delta_y],[0],[-self.delta_x],[0],[self.delta_theta],[0]],dtype=Utils.matrix_data_type)
        if normalize:
            twist /= np.linalg.norm(twist)
        return twist