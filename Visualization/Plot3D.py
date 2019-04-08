import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # used implicitly for projection = '3d'!
from Numerics import ImageProcessing, SE3
import numpy as np
import cv2

def show():
    plt.show()

def legend(h):
    plt.legend(handles=h)

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

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
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

    # TODO move this out
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    #ax.set_zlabel('Z Label')

    if draw:
        plt.show()


def plot_translation_component(index, se3_list, ax, style, clear = False, draw = True):
    #for i in range(0,N):
    if clear:
        ax.clear()

    for i in range(0,len(se3_list), 1):
        se3 = se3_list[i]
        translation_comp = se3[index,3]

        ax.plot([i],[translation_comp],style)

    if draw:
        plt.draw()
        plt.pause(1)

def plot_steering_commands(encoder_list, rev_ax,steer_ax, style, clear = False, draw = True):
    if clear:
        rev_ax.clear()
        steer_ax.clear()

    for i in range(0,len(encoder_list), 1):
        cmd = encoder_list[i]

        rev_ax.plot(i,cmd[0], style)
        steer_ax.plot(i,cmd[1], style)

    if draw:
        plt.draw()
        plt.pause(1)


def plot_rmse(se3_gt_list, se3_est_list, ax,  style = 'bx', clear = False, draw = True, offset = 1):
    if clear:
        ax.clear()

    #rmse_list = SE3.root_mean_square_error_for_entire_list(se3_gt_list,se3_est_list)
    rmse_list = SE3.root_mean_square_error_for_consecutive_frames_accumulated(se3_gt_list, se3_est_list, offset=offset)

    for i in range(0,len(rmse_list), 1):
        rmse = rmse_list[i]
        ax.plot([i],[rmse],style)


    if draw:
        plt.draw()
        plt.pause(1)


# http://jakevdp.github.io/mpl_tutorial/tutorial_pages/tut5.html
# https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
def plot_wireframe_ellipsoid(y_fac, ellipse_factor_list, se3_list, ax, label_axes = False,clear = False, draw = True):
    list_len = len(se3_list)
    assert len(ellipse_factor_list) == list_len

    if clear:
        ax.clear()
    u = np.linspace(0, np.pi, 30)
    v = np.linspace(0, 2 * np.pi, 30)

    x = np.outer(np.sin(u), np.sin(v))
    y = np.outer(np.sin(u), np.cos(v))
    z = np.outer(np.cos(u), np.ones_like(v))

    x_plot = np.zeros(x.shape)
    y_plot = np.zeros(y.shape)
    z_plot = np.zeros(z.shape)

    se3_acc = np.identity(4)

    r,c = x.shape

    for i in range(0,list_len):
        (x_fac,z_fac) = ellipse_factor_list[i]
        change_of_basis = se3_list[i]
        se3_acc = np.matmul(se3_acc,change_of_basis)
        for j in range(0,r):
            for i in range(0,c):
                point = np.array([x_fac * x[j,i], y_fac * y[j,i], z_fac * z[j,i], 1.0])
                new_point = np.matmul(se3_acc,point)
                x_plot[j,i] = new_point[0]
                y_plot[j,i] = new_point[1]
                z_plot[j,i] = new_point[2]

        ax.plot_wireframe(x_plot, y_plot, z_plot)

    if label_axes:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    if draw:
        #plt.axis('equal')
        plt.show()



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