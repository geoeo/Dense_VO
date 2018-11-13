from MotionModels.MotionDelta import MotionDelta

class Pose:

    def __init__(self):
        self.x = 0
        self.y = 0
        self.theta = 0

    def apply_motion(self, motion_delta : MotionDelta, dt):
        self.x += motion_delta.delta_X*dt
        self.y += motion_delta.delta_Y*dt
        self.theta += motion_delta.delta_Theta * dt