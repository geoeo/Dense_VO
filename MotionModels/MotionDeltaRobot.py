import numpy as np
import math
from Numerics import Utils

# In robot space
class MotionDeltaRobot:

    # In Ackermann space
    def __init__(self):
        self.delta_x = 0.0
        self.delta_y = 0.0
        self.delta_theta = 0.0


