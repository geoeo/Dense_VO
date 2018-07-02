import numpy as np
import Numerics.Lie
import Numerics.Utils as Utils

def solve_SE3(X,Y,max_its,eps):
    #init
    # array for twist values x, y, z, roll, pitch, yaw
    w = np.array([0, 0, 0, 0, 0, 0],dtype=np.float32)
    t_est = np.array([0, 0, 0],dtype=np.float32).reshape((3,1))
    R_est = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]],dtype=np.float32)

    SE_3_est = np.append(np.append(R_est,t_est,axis=1),Utils.homogenous_for_SE3(),axis=0)

    for it in range(0,max_its,1):
        Y_est = np.matmul(SE_3_est,X)
        diff = Y - Y_est
        v = np.sum(np.sum(np.square(diff),axis=0))

        if v < eps:
            print('done')
            break