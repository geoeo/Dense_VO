import numpy as np
import cv2
import Numerics as ImageProcessing

gray_scale = np.zeros((320,640),dtype=np.float64)

for x in range(0,640):
    for y in range(0, 320):
        gray_scale[y,x] = x/640.0


grayscale_image = ImageProcessing.normalize_to_image_space(gray_scale)
cv2.imwrite("grayscale.png",grayscale_image)


