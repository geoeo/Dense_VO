import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Numerics import ImageProcessing
import numpy as np
import cv2

def show():
    plt.show()

def scatter_plot(X,Y,Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X,Y,Z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    #ax.set_zlabel('Z Label')

    plt.show()


def scatter_plot_array(points,labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    N = len(points)
    for i in range(0,N):
        X,Y,Z = points[i]
        label = labels[i]
        ax.scatter(X,Y,Z,label=label)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    #ax.set_zlabel('Z Label')

    plt.legend(loc=2)
    plt.show()

def plot_array_lines(points, ax, style = '-ro',clear = True, draw = True):
    #for i in range(0,N):
    if clear:
        ax.clear()

    X,Y,Z = points[0, :], points[1, :], points[2, :]
    for i in range(0,len(X), 2):
        ax.plot(X[i:i+2],Y[i:i+2],Z[i:i+2],style)
        ax.text(X[i], Y[i], Z[i], '%s' % (i/2))

    #ax.set_xlabel('X Label')
    #ax.set_ylabel('Y Label')
    #ax.set_zlabel('Z Label')

    #plt.legend(loc=2)
    if draw:
        plt.draw()
        plt.pause(1)

def plot_translation_component(index, se3_gt_list, se3_est_list, ax,style_gt='-gx' ,style_est ='-rx', clear = False, draw = True):
    #for i in range(0,N):
    if clear:
        ax.clear()

    for i in range(0,len(se3_gt_list), 1):
        se3_gt = se3_gt_list[i]
        se3_est = se3_est_list[i]
        translation_comp_gt = se3_gt[index,3]
        translation_comp_est = se3_est[index,3]

        ax.plot([i],[translation_comp_gt],style_gt)
        ax.plot([i],[translation_comp_est],style_est)
        #ax.text(X[i], Y[i], Z[i], '%s' % (i/2))

    #ax.set_xlabel('X Label')
    #ax.set_ylabel('Y Label')
    #ax.set_zlabel('Z Label')

    #plt.legend(loc=2)
    if draw:
        plt.draw()
        plt.pause(1)


def draw():
    plt.draw()
    plt.pause(1)

def scatter_plot_sub(plot_1,plot_2,labels_1,labels_2):
    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')
    N = len(plot_1)
    for i in range(0,N):
        X,Y,Z = plot_1[i]
        label = labels_1[i]
        ax.scatter(X,Y,Z,label=label)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')

    plt.legend(loc=2)

    ax_2 = fig.add_subplot(212, projection='3d')
    N = len(plot_2)
    for i in range(0,N):
        X,Y,Z = plot_2[i]
        label = labels_2[i]
        ax_2.scatter(X,Y,Z,label=label)
    ax_2.set_xlabel('X Label')
    ax_2.set_ylabel('Y Label')
    #ax.set_zlabel('Z Label')

    plt.legend(loc=2)



    plt.show()

def save_projection_of_back_projected(height,width,frame_reference,X_back_projection):
    N = width*height
    # render/save image of projected, back projected points
    projected_back_projected = frame_reference.camera.apply_perspective_pipeline(X_back_projection)
    # scale ndc if applicable
    #projected_back_projected[0,:] = projected_back_projected[0,:]*width
    #projected_back_projected[1,:] = projected_back_projected[1,:]*height
    debug_buffer = np.zeros((height,width), dtype=np.float64)
    for i in range(0,N,1):
        u = projected_back_projected[0,i]
        v = projected_back_projected[1,i]

        if not np.isnan(u) and not np.isnan(v):
            debug_buffer[int(v),int(u)] = 1.0
    cv2.imwrite("debug_buffer.png", ImageProcessing.normalize_to_image_space(debug_buffer))