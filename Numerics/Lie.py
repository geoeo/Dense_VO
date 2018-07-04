import numpy as np


def generator_x():
    return np.array([[0,0,0,1],
                     [0,0,0,0],
                     [0,0,0,0],
                     [0,0,0,0]],dtype=np.float64)


def generator_y():
    return np.array([[0,0,0,0],
                     [0,0,0,1],
                     [0,0,0,0],
                     [0,0,0,0]],dtype=np.float64)


def generator_z():
    return np.array([[0,0,0,0],
                     [0,0,0,0],
                     [0,0,0,1],
                     [0,0,0,0]],dtype=np.float64)


# Rotation Around X
def generator_roll():
    return np.array([[0,0,0,0],
                     [0,0,-1,0],
                     [0,1,0,0],
                     [0,0,0,0]],dtype=np.float64)


# Rotation Around Y
def generator_pitch():
    return np.array([[0,0,1,0],
                     [0,0,0,0],
                     [-1,0,0,0],
                     [0,0,0,0]],dtype=np.float64)


# Rotation Around Z
def generator_yaw():
    return np.array([[0,-1,0,0],
                     [1,0,0,0],
                     [0,0,0,0],
                     [0,0,0,0]],dtype=np.float64)