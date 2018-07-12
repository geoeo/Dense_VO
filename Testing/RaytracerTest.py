import numpy as np
import cv2
import Numerics.ImageProcessing as ImageProcessing
import Numerics.Generator as Generator
import Raytracer.Scene as Scene
import Raytracer.Geometry as Geometry
import Camera.Camera as Camera

#gray_scale = np.zeros((320,640),dtype=np.float64)

#for x in range(0,640):
#    for y in range(0, 320):
#        gray_scale[y,x] = x/640.0


#grayscale_image = ImageProcessing.normalize_to_image_space(gray_scale)
#cv2.imwrite("grayscale.png",grayscale_image)

N = 20
(X,Y,Z) = Generator.generate_3d_plane(1,1,-30,N,4)
H = np.repeat(1,N)

points = np.transpose(np.array(list(map(lambda x: list(x),list(zip(X,Y,Z,H))))))

spheres = Geometry.generate_spheres(points)

camera = Camera.normalized_camera()

scene = Scene.Scene(640,320,spheres,camera)

scene.render()

frame_buffer_image = ImageProcessing.normalize_to_image_space(scene.frame_buffer)
depth_buffer_image = ImageProcessing.normalize_to_image_space(ImageProcessing.z_standardise(scene.depth_buffer))

cv2.imwrite("framebuffer.png",frame_buffer_image)
cv2.imwrite("depthbuffer.png",depth_buffer_image)




