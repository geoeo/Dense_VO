import numpy as np
from PIL import Image

im = Image.open("/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Home_Images/Images_ZR300_XTrans/image_1.png")
im_greyscale = im.convert('L')
im_greyscale.show()

np_pixels = np.asarray(im_greyscale)

mean = np.mean(np_pixels)
std_dev = np.std(np_pixels)

pixels_normalized = (np_pixels - mean) / std_dev

min = np.min(pixels_normalized)
max = np.max(pixels_normalized)

#https://en.wikipedia.org/wiki/Normalization_(image_processing)
pixels_normalized_disp = (pixels_normalized - min)*(255.0/(max-min))

im_gc_normalized = Image.fromarray(pixels_normalized_disp)
im_gc_normalized.show()