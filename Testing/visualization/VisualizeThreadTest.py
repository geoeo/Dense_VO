from Visualization import Visualizer
import time

visualizer = Visualizer.VisualizerThread(1,"Visualizer")
visualizer.start()

time.sleep(5)

visualizer.stop()