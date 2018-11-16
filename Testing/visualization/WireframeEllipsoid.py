import matplotlib.pyplot as plt
from Visualization import Plot3D
import numpy as np

id = np.identity(3)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')





#Plot3D.plot_wireframe_ellipsoid(1,1,1,id,ax,label_axes=True, clear=True,draw=False)
Plot3D.plot_wireframe_ellipsoid(2,0.1,1, id,ax,label_axes=True, clear=False,draw=True)



