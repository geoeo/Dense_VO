import threading
import matplotlib.pyplot as plt
import time
import numpy as np
from Numerics import Utils
from Visualization import Plot3D

class VisualizerThread(threading.Thread):

    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.pose_estimate_list = []
        self.threadLock = threading.Lock()
        self.running = True
        self.pose_estimate_list.append(np.identity(4,dtype=Utils.matrix_data_type))
        self.figure = plt.figure()
        self.graph = self.figure.add_subplot(111, projection='3d')

        # These points will be plotted with incomming se3 matricies
        X, Y, Z = [0, 0], [0, 0], [0, -1]
        H = np.repeat(1, 2)

        self.point_pair = Utils.to_homogeneous_positions(X, Y, Z, H)

    # performs visualization
    def run(self):
        while self.running:
            print("Visualizing " + self.name)
            if not self.threadLock.acquire(blocking=False):
                print("pose estimate list being updated. Skipping this run")

            points_to_be_graphed = self.point_pair[0:3, :]
            # for se3 in self.pose_estimate_list:
            #    print('*'*80)
            #    print(se3)
            #    print('*'*80)

            Plot3D.plot_array_lines(points_to_be_graphed,self.graph)




            self.threadLock.release()
            time.sleep(1)

        print("Exiting " + self.name)

    def add_pose_to_list(self, se3):
        self.threadLock.acquire()
        self.pose_estimate_list.append(se3)
        self.threadLock.release()

    def stop(self):
        self.running = False

class Visualizer():

    def __init__(self, solver_thread_manager, ground_truth = None):
        self.solver_thread_manager = solver_thread_manager
        self.ground_truth = ground_truth
        self.figure = plt.figure()
        self.graph = self.figure.add_subplot(111, projection='3d')

        # These points will be plotted with incomming se3 matricies
        X, Y, Z = [0, 0], [0, 0], [0, -1]
        H = np.repeat(1, 2)

        self.point_pair = Utils.to_homogeneous_positions(X, Y, Z, H)


    def visualize_poses(self,pose_list):
        if len(pose_list) == 0:
            print('pose list is empty, skipping')
            return

        se3_init = pose_list[0]
        points_to_be_graphed = np.matmul(se3_init,self.point_pair)[0:3,:]

        if not self.ground_truth is None:
            points_transformed = np.matmul(self.ground_truth, self.point_pair)[0:3,:]
            Plot3D.plot_array_lines(points_transformed, self.graph, '-go', clear = True, draw=False)
        for i in range(1,len(pose_list)):
            se3 = pose_list[i]
            points_transformed = np.matmul(se3,self.point_pair)[0:3,:]
            # identity gets transformed twice
            points_to_be_graphed = np.append(points_to_be_graphed,points_transformed,axis=1)

        Plot3D.plot_array_lines(points_to_be_graphed,self.graph,clear=False,draw=True)

    # performs visualization
    def visualize(self):
        print("Visualizing...")

        while self.solver_thread_manager.is_running:
            if not self.solver_thread_manager.threadLock.acquire(blocking=False):
                print("pose estimate list being updated. Skipping this run")
                time.sleep(0.1)
                continue

            self.visualize_poses(self.solver_thread_manager.pose_estimate_list)

            self.solver_thread_manager.threadLock.release()
            time.sleep(5)

        plt.show()
        print("Exiting Visualizer")


