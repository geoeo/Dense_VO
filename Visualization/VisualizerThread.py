import threading
import matplotlib.pyplot as plt
import time
import numpy as np
from Numerics import Utils, SE3
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
        self.se3_graph = self.figure.add_subplot(211, projection='3d')
        self.x_graph = self.figure.add_subplot(234, projection='2d')
        self.y_graph = self.figure.add_subplot(235, projection='2d')
        self.z_graph = self.figure.add_subplot(236, projection='2d')

        self.x_graph.set_title("x")
        self.y_graph.set_title("y")
        self.z_graph.set_title("z")

        # These points will be plotted with incomming se3 matricies
        #X, Y, Z = [0, 0], [0, 0], [0, -1]
        X, Y, Z = [0, 0], [0, 0], [0, 1]
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

            Plot3D.plot_array_lines(points_to_be_graphed, self.se3_graph)

            self.threadLock.release()
            time.sleep(1)

        print("Exiting " + self.name)

    def add_pose_to_list(self, se3):
        self.threadLock.acquire()
        self.pose_estimate_list.append(se3)
        self.threadLock.release()

    def stop(self):
        self.running = False