import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def scatter_plot(X,Y,Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X,Y,Z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    #ax.set_zlabel('Z Label')

    plt.show()

def scatter_plot(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for point in points:
        X,Y,Z = point
        ax.scatter(X,Y,Z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    #ax.set_zlabel('Z Label')

    plt.show()