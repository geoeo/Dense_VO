import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import numpy as np
from Numerics import Utils, SE3
from Visualization import Plot3D


def make_patch(color, label):
    return mpatches.Patch(color=color, label=label)

class Visualizer():

    def __init__(self, ground_truth_list = None, plot_steering = False, title = None, plot_trajectory=True, plot_rmse=True):
        self.ground_truth_list = []
        if ground_truth_list is not None:
            self.ground_truth_list = ground_truth_list
        self.figure = plt.figure()
        self.plot_steering = plot_steering
        self.plot_trajectory = plot_trajectory
        self.plot_rmse = plot_rmse

        if title:
            plt.title(title)
        if not plot_trajectory and not plot_rmse:
            if not plot_steering:
                self.x_graph = self.figure.add_subplot(131)
                self.y_graph = self.figure.add_subplot(132)
                self.z_graph = self.figure.add_subplot(133)
            else:
                self.x_graph = self.figure.add_subplot(231)
                self.y_graph = self.figure.add_subplot(232)
                self.z_graph = self.figure.add_subplot(233)
                self.rev = self.figure.add_subplot(224)
                self.steer = self.figure.add_subplot(225)

        elif not plot_steering:
            if plot_trajectory:
                self.se3_graph = self.figure.add_subplot(311, projection='3d')
            self.x_graph = self.figure.add_subplot(334)
            self.y_graph = self.figure.add_subplot(335)
            self.z_graph = self.figure.add_subplot(336)
            if self.plot_rmse:
                self.rmse_graph = self.figure.add_subplot(313)
        else:
            if plot_trajectory:
                self.se3_graph = self.figure.add_subplot(411, projection='3d')
            self.x_graph = self.figure.add_subplot(434)
            self.y_graph = self.figure.add_subplot(435)
            self.z_graph = self.figure.add_subplot(436)
            if self.plot_rmse:
                self.rmse_graph = self.figure.add_subplot(413)
            self.rev = self.figure.add_subplot(427)
            self.steer = self.figure.add_subplot(428)

            self.rev.set_title("rev cmd")
            self.steer.set_title("str cmd")


        #self.se3_graph.set_aspect('equal')
        if self.plot_trajectory:
            self.se3_graph.set_title("relative pose estimate")
        self.x_graph.set_title("X")
        self.y_graph.set_title("Y")
        self.z_graph.set_title("Z")
        if self.plot_rmse:
            self.rmse_graph.set_title("drift per frame (pose error)")

        # These points will be plotted with incomming se3 matricies
        X, Y, Z = [0, 0], [0, 0], [0, -1]
        #X, Y, Z = [0, 0], [0, 0], [0, 1]
        H = np.repeat(1, 2)

        self.point_pair = Utils.to_homogeneous_positions(X, Y, Z, H)


    def show(self):
        Plot3D.show()

    def legend(self, handles):
        Plot3D.legend(handles)

    def visualize_steering(self, encoder_list, clear = False, draw = False):
        assert self.plot_steering
        Plot3D.plot_steering_commands(encoder_list, self.rev, self.steer, style='-rx', clear=clear, draw=draw)

    def visualize_ground_truth(self, clear= True, draw=False):
        if len(self.ground_truth_list) > 0:
            gt_init = self.ground_truth_list[0]
            points_gt_to_be_graphed = np.matmul(gt_init,self.point_pair)[0:3,:]

            for i in range(1,len(self.ground_truth_list)):
                se3 = self.ground_truth_list[i]
                points_transformed = np.matmul(se3,self.point_pair)[0:3,:]
                # identity gets transformed twice
                points_gt_to_be_graphed = np.append(points_gt_to_be_graphed,points_transformed,axis=1)

            if self.plot_trajectory:
                Plot3D.plot_array_lines(points_gt_to_be_graphed, self.se3_graph, '-go', clear=clear, draw=False)

            Plot3D.plot_translation_component(0, self.ground_truth_list, self.x_graph, style='-gx', clear=False, draw=draw)
            Plot3D.plot_translation_component(1, self.ground_truth_list, self.y_graph, style='-gx', clear=False, draw=draw)
            Plot3D.plot_translation_component(2, self.ground_truth_list, self.z_graph, style='-gx', clear=False, draw=draw)

    def visualize_poses(self, pose_list, draw = True, style='-rx'):
        if len(pose_list) == 0:
            print('pose list is empty, skipping')
            return

        se3_init = pose_list[0]
        points_to_be_graphed = np.matmul(se3_init,self.point_pair)[0:3,:]

        for i in range(1,len(pose_list)):

            se3 = pose_list[i]

            points_transformed = np.matmul(se3,self.point_pair)[0:3,:]
            # identity gets transformed twice
            points_to_be_graphed = np.append(points_to_be_graphed,points_transformed,axis=1)

        if self.plot_trajectory:
            Plot3D.plot_array_lines(points_to_be_graphed, self.se3_graph, clear=False, draw=draw)

        Plot3D.plot_translation_component(0, pose_list, self.x_graph, style=style, clear=False, draw=draw)
        Plot3D.plot_translation_component(1, pose_list, self.y_graph, style=style, clear=False, draw=draw)
        Plot3D.plot_translation_component(2, pose_list, self.z_graph, style=style, clear=False, draw=draw)
        if self.plot_rmse:
            Plot3D.plot_rmse(self.ground_truth_list, pose_list, self.rmse_graph, clear=False, draw=draw, offset=1)

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


