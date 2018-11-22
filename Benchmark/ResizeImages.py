from Benchmark import ListGenerator
from Numerics import Utils, ImageProcessing
import os
import cv2
import numpy as np


bench_path = '/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/rccar_26_09_18/'
dataset = 'marc_4_full/'
img_source_dir = 'depth/'
img_target_dir = 'depth_large_norm/'

#img_source_dir = 'depth_rect/'
#img_target_dir = 'depth_rect_large/'

full_source_path = bench_path + dataset + img_source_dir
full_target_path = bench_path + dataset + img_target_dir

compression_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]

timestamps = ListGenerator.get_files_from_directory(full_source_path, '.')

source_files = list(map(lambda float: full_source_path+f'{float:.9f}'+'.png',timestamps))
target_files = list(map(lambda float: full_target_path+f'{float:.9f}'+'.png',timestamps))

zipped_files = zip(source_files,target_files)

if not os.path.exists(full_target_path):
    os.mkdir(full_target_path,mode=0o0755)

for source_path,target_path in zipped_files:
    image = cv2.imread(source_path, cv2.IMREAD_ANYDEPTH)
    count = np.count_nonzero(image)
    resized_image = cv2.resize(image,(640,480),interpolation=cv2.INTER_CUBIC)
    # Used for visualization only
    resized_norm_image = cv2.normalize(resized_image, None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16UC1)
    cv2.imwrite(target_path,resized_norm_image,compression_params)
    #cv2.imwrite(target_path,resized_image,compression_params)