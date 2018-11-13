
class SteeringCommands:
    def __init__(self):
        self.linear_velocity = 0.0
        self.steering_angle = 0.0

    def set(self, velocity, alpha):
        self.linear_velocity = velocity
        self.steering_angle = alpha