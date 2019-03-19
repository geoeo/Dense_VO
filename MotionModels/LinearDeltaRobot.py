import numpy as np
import math
from Numerics import Utils

# In robot space
class LinearDeltaRobot:

    # In Ackermann space
    def __init__(self):
        self.delta_x = 0.0
        self.delta_y = 0.0
        self.delta_z = 0.0

        self.delta_v_x = 0.0
        self.delta_v_y = 0.0
        self.delta_v_z = 0.0


