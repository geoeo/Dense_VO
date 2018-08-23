import numpy as np
import cv2
import Numerics.ImageProcessing as ImageProcessing
import Numerics.Generator as Generator
import Raytracer.Scene as Scene
import Raytracer.Geometry as Geometry
import Camera.Camera as Camera
import Numerics.Utils as Utils
import Visualization.Plot3D as Plot3D

#gray_scale = np.zeros((320,640),dtype=np.float64)

#for x in range(0,640):
#    for y in range(0, 320):
#        gray_scale[y,x] = x/640.0


#grayscale_image = ImageProcessing.normalize_to_image_space(gray_scale)
#cv2.imwrite("grayscale.png",grayscale_image)

N = 20
(X,Y,Z) = Generator.generate_3d_plane(1,1,-30,N,4)
H = np.repeat(1,N)
image_width = 320
image_height = 320

points = np.transpose(np.array(list(map(lambda x: list(x),list(zip(X,Y,Z,H))))))

spheres = Geometry.generate_spheres(points)

camera = Camera.normalized_camera(0,0,image_width/2,image_height/2)
camera_translated = Camera.normalized_camera(0.5,0,image_width/2,image_height/2)

##############

points_persp = camera.apply_perspective_pipeline(points)

(X_orig,Y_orig,Z_orig) = list(Utils.points_into_components(points))
(X_persp,Y_persp,Z_persp) = list(Utils.points_into_components(points_persp))

##############

scene = Scene.Scene(image_width,image_height,spheres,camera)

scene.render()

frame_buffer_image = ImageProcessing.normalize_to_image_space(scene.frame_buffer)
depth_buffer_image = scene.depth_buffer

cv2.imwrite("framebuffer.png",frame_buffer_image)
cv2.imwrite("depthbuffer.png",depth_buffer_image)

###############

scene_translated = Scene.Scene(image_width,image_height,spheres,camera_translated)

scene_translated.render()

frame_buffer_image = ImageProcessing.normalize_to_image_space(scene_translated.frame_buffer)
depth_buffer_image = scene_translated.depth_buffer

cv2.imwrite("framebuffer_translated.png",frame_buffer_image)
cv2.imwrite("depthbuffer_translated.png",depth_buffer_image)


################

Plot3D.scatter_plot_sub([(X_orig,Y_orig,Z_orig)],[(X_persp,Y_persp,Z_persp)],['original'],['projected'])


