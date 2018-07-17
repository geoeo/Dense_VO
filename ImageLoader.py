import numpy as np
import cv2
import Numerics.ImageProcessing as ImageProcessing

im_greyscale = cv2.imread('/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Home_Images/Images_ZR300_XTrans/image_1.png',0)

pixels_standardised = ImageProcessing.z_standardise(im_greyscale)
pixels_normalized_disp = ImageProcessing.normalize_to_image_space(pixels_standardised)

#im_gc_normalized = Image.fromarray(pixels_normalized_disp)
#m_gc_normalized.show()

level_1 = cv2.resize(pixels_standardised, (0,0), fx=0.5, fy=0.5)
level_2 = cv2.resize(level_1, (0,0), fx=0.5, fy=0.5)
level_3 = cv2.resize(level_2, (0,0), fx=0.5, fy=0.5)
level_4 = cv2.resize(level_3, (0,0), fx=0.5, fy=0.5)
level_5 = cv2.resize(level_4, (0,0), fx=0.5, fy=0.5)

(y_1,x_1) = pixels_standardised.shape
dst_size_level_1 = (int(0.5*y_1),int(0.5*x_1))
dst_size_level_1_flipped = (int(0.5*x_1),int(0.5*y_1))
output_image_level_1 = np.zeros(dst_size_level_1)

(y_2,x_2) = output_image_level_1.shape
dst_size_level_2 = (int(0.5*y_2),int(0.5*x_2))
dst_size_level_2_flipped = (int(0.5*x_2),int(0.5*y_2))
output_image_level_2 = np.zeros(dst_size_level_2)

(y_3,x_3) = output_image_level_2.shape
dst_size_level_3 = (int(0.5*y_3),int(0.5*x_3))
dst_size_level_3_flipped = (int(0.5*x_3),int(0.5*y_3))
output_image_level_3 = np.zeros(dst_size_level_3)

(y_4,x_4) = output_image_level_3.shape
dst_size_level_4 = (int(0.5*y_4),int(0.5*x_4))
dst_size_level_4_flipped = (int(0.5*x_4),int(0.5*y_4))
output_image_level_4 = np.zeros(dst_size_level_4)

(y_5,x_5) = output_image_level_4.shape
dst_size_level_5 = (int(0.5*y_5),int(0.5*x_5))
dst_size_level_5_flipped = (int(0.5*x_5),int(0.5*y_5))
output_image_level_5 = np.zeros(dst_size_level_5)


cv2.pyrDown(pixels_standardised,output_image_level_1,dst_size_level_1_flipped)
cv2.pyrDown(output_image_level_1,output_image_level_2,dst_size_level_2_flipped)
cv2.pyrDown(output_image_level_2,output_image_level_3,dst_size_level_3_flipped)
cv2.pyrDown(output_image_level_3,output_image_level_4,dst_size_level_4_flipped)
cv2.pyrDown(output_image_level_4,output_image_level_5,dst_size_level_5_flipped)


print(im_greyscale.shape)
print(level_1.shape)
print(level_2.shape)
print(level_3.shape)
print(level_4.shape)
print(level_5.shape)

print(output_image_level_1.shape)
print(output_image_level_2.shape)
print(output_image_level_3.shape)
print(output_image_level_4.shape)
print(output_image_level_5.shape)

cv2.imshow('Level 0 (Norm to 255)',pixels_normalized_disp)
cv2.imshow('Level 0',pixels_standardised)

cv2.imshow('Level 1',level_1)
cv2.imshow('Level 2',level_2)
cv2.imshow('Level 3',level_3)
cv2.imshow('Level 4',level_4)
cv2.imshow('Level 5',level_5)

cv2.imshow('Level 1 Conv',output_image_level_1)
cv2.imshow('Level 2 Conv',output_image_level_2)
cv2.imshow('Level 3 Conv',output_image_level_3)
cv2.imshow('Level 4 Conv',output_image_level_4)
cv2.imshow('Level 5 Conv',output_image_level_5)

while True:
    k = cv2.waitKey(5) & 0xFF
    # ESC
    if k == 27:
        break

cv2.destroyAllWindows()