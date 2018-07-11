import numpy as np

class Sphere:
    def __init__(self,origin : np.ndarray,radius : float):
        if origin.shape != (3,1):
            raise TypeError('Sphere origin not shape (3,1)')
        self.origin = origin
        self.radius = radius