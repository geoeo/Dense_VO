#https://docs.opencv.org/3.4/d7/d8b/tutorial_py_lucas_kanade.html
import numpy as np
import cv2 as cv
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.1,
                       minDistance = 3,
                       blockSize = 3 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))
#rgb_ref_file_path = '/Volumes/Sandisk/Diplomarbeit/Results/TUM/f2d1/rgb/1311868164.363181.png'
#rgb_target_file_path = '/Volumes/Sandisk/Diplomarbeit/Results/TUM/f2d1/rgb/1311868164.399026.png'
rgb_ref_file_path = '/Volumes/Sandisk/Diplomarbeit/Results/TUW/dataset_3/color/299206.044653428.png'
rgb_target_file_path = '/Volumes/Sandisk/Diplomarbeit/Results/TUW/dataset_3/color/299206.642492522.png'
# Take first frame and find corners in it
#ret, old_frame = cap.read()
#old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
im_greyscale_reference = cv.imread(rgb_ref_file_path, cv.IMREAD_GRAYSCALE)
p0 = cv.goodFeaturesToTrack(im_greyscale_reference, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(im_greyscale_reference)

#ret,frame = cap.read()
#frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
frame_gray = cv.imread(rgb_target_file_path, cv.IMREAD_GRAYSCALE)
# calculate optical flow
p1, st, err = cv.calcOpticalFlowPyrLK(im_greyscale_reference, frame_gray, p0, None, **lk_params)
# Select good points
good_new = p1[st==1]
good_old = p0[st==1]
# draw the tracks
for i,(new,old) in enumerate(zip(good_new,good_old)):
    a,b = new.ravel()
    c,d = old.ravel()
    mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    im_greyscale_reference = cv.circle(im_greyscale_reference,(a,b),5,color[i].tolist(),-1)
img = cv.add(im_greyscale_reference,mask)
cv.imshow('frame',img)
k = cv.waitKey(0) & 0xff
cv.destroyAllWindows()