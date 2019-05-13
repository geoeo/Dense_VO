import cv2 as cv
import numpy as np
rgb_ref_file_path = '/Volumes/Sandisk/Diplomarbeit/Results/TUW/dataset_3/color/299206.044653428.png'
rgb_target_file_path = '/Volumes/Sandisk/Diplomarbeit/Results/TUW/dataset_3/color/299206.642492522.png'


#ret, frame1 = cap.read()
#prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
frame1= cv.imread(rgb_ref_file_path, cv.IMREAD_ANYCOLOR)
frame1_gray = cv.imread(rgb_ref_file_path, cv.IMREAD_GRAYSCALE)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

#ret, frame2 = cap.read()
#next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
frame2 = cv.imread(rgb_target_file_path, cv.IMREAD_ANYCOLOR)
frame2_gray = cv.imread(rgb_target_file_path, cv.IMREAD_GRAYSCALE)
flow = cv.calcOpticalFlowFarneback(frame1_gray,frame2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
hsv[...,0] = ang*180/np.pi/2
hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
#hsv[...,2] = 255
bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
cv.imshow('frame2',bgr)
k = cv.waitKey(0) & 0xff
#elif k == ord('s'):
    #cv.imwrite('opticalfb.png',frame2)
    #cv.imwrite('opticalhsv.png',bgr)
cv.destroyAllWindows()