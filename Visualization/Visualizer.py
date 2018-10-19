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

class Visualizer():

    def __init__(self, ground_truth_list = None):
        self.ground_truth_list = []
        if ground_truth_list is not None:
            self.ground_truth_list = ground_truth_list
        self.figure = plt.figure()
        self.se3_graph = self.figure.add_subplot(311, projection='3d')
        self.x_graph = self.figure.add_subplot(334)
        self.y_graph = self.figure.add_subplot(335)
        self.z_graph = self.figure.add_subplot(336)
        self.rmse_graph = self.figure.add_subplot(313)

        self.se3_graph.set_title("relative pose estimate")
        self.x_graph.set_title("x")
        self.y_graph.set_title("y")
        self.z_graph.set_title("z")
        self.rmse_graph.set_title("drift per frame (pose error)")

        # These points will be plotted with incomming se3 matricies
        X, Y, Z = [0, 0], [0, 0], [0, -1]
        H = np.repeat(1, 2)

        self.point_pair = Utils.to_homogeneous_positions(X, Y, Z, H)


    def show(self):
        Plot3D.show()

    def visualize_poses(self, pose_list, draw = True):
        if len(pose_list) == 0:
            print('pose list is empty, skipping')
            return

        if len(self.ground_truth_list) > 0:
            gt_init = self.ground_truth_list[0]
            points_gt_to_be_graphed = np.matmul(gt_init,self.point_pair)[0:3,:]

            for i in range(1,len(self.ground_truth_list)):
                se3 = self.ground_truth_list[i]
                points_transformed = np.matmul(se3,self.point_pair)[0:3,:]
                # identity gets transformed twice
                points_gt_to_be_graphed = np.append(points_gt_to_be_graphed,points_transformed,axis=1)

            Plot3D.plot_array_lines(points_gt_to_be_graphed, self.se3_graph, '-go', clear=True, draw=False)

        se3_init = pose_list[0]
        points_to_be_graphed = np.matmul(se3_init,self.point_pair)[0:3,:]

        for i in range(1,len(pose_list)):

            se3 = pose_list[i]

            points_transformed = np.matmul(se3,self.point_pair)[0:3,:]
            # identity gets transformed twice
            points_to_be_graphed = np.append(points_to_be_graphed,points_transformed,axis=1)

        Plot3D.plot_array_lines(points_to_be_graphed, self.se3_graph, clear=False, draw=draw)

        Plot3D.plot_translation_component(0, self.ground_truth_list, pose_list, self.x_graph, clear=False, draw=draw)
        Plot3D.plot_translation_component(1, self.ground_truth_list, pose_list, self.y_graph, clear=False, draw=draw)
        Plot3D.plot_translation_component(2, self.ground_truth_list, pose_list, self.z_graph, clear=False, draw=draw)
        Plot3D.plot_rmse(self.ground_truth_list, pose_list, self.rmse_graph, clear=False, draw=draw)

    # performs visualization
    def visualize(self,solver_thread_manager):
        print("Visualizing...")

        while solver_thread_manager.is_running:
            if not solver_thread_manager.threadLock.acquire(blocking=False):
                print("pose estimate list being updated. Skipping this run")
                time.sleep(0.1)
                continue

            self.visualize_poses(solver_thread_manager.pose_estimate_list)

            solver_thread_manager.threadLock.release()
            time.sleep(1)

        plt.show()
        print("Exiting Visualizer")


