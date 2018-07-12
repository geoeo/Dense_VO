import numpy as np

matrix_data_type = np.float32

def points_into_components(points):
    X = points[0,:]
    Y = points[1,:]
    Z = points[2,:]

    return(X,Y,Z)

def homogenous_for_SE3():
    return np.array([[0,0,0,1]])

def padding_for_generator_jacobi():
    return np.array([[0,0,0,0,0,0]])

def Unit_SE3():
    t = np.array([0, 0, 0],dtype=np.float32).reshape((3,1))
    R = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]],dtype=np.float32)

    return np.append(np.append(R,t,axis=1),homogenous_for_SE3(),axis=0)

def skew_symmetric(a,b,c):
    return np.array([[0, -c, b],
                     [c, 0, -a],
                     [-b, a, 0]], dtype=np.float64)

def normalize(vector: np.ndarray):
    return vector/np.linalg.norm(vector)

def move_data_along_z(points, magnitude):
    translation = np.reshape(np.array([0,0,magnitude]),(3,1))
    se3_translation_only = np.append(np.append(np.identity(3,dtype=matrix_data_type),translation,axis=1),homogenous_for_SE3(),axis=0)

    return np.matmul(se3_translation_only,points)

def fast_dot(a,b):
    return a[0,0]*b[0,0]+a[1,0]*b[1,0]+a[2,0]*b[2,0]


