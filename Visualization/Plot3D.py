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

def scatter_plot(points,labels):
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