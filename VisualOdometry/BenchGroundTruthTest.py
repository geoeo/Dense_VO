import numpy as np
import Numerics.Utils as Utils
import Camera.Intrinsic as Intrinsic
import Camera.Camera as Camera
import VisualOdometry.Solver as Solver
from VisualOdometry import Frame
from Numerics import ImageProcessing
from Benchmark import Parser
from Benchmark import associate
from Numerics import SE3
import cv2

bench_path = '/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Bench/'
xyz_dataset = 'rgbd_dataset_freiburg1_xyz/'
rgb_folder = 'rgb/'
depth_folder = 'depth/'

dataset_root = bench_path+xyz_dataset
rgb_text = dataset_root +'rgb.txt'
depth_text = dataset_root+'depth.txt'
match_text = dataset_root+'matches.txt'
groundtruth_text = dataset_root+'groundtruth.txt'

#rgb_id_ref = 1305031102.175304
#rgb_id_target = 1305031102.211214

rgb_id_ref = 1305031108.743502
rgb_id_target = 1305031108.775493

#rgb_id_ref = 1305031119.615017
#rgb_id_target = 1305031119.647903

image_groundtruth_dict = dict(associate.match(rgb_text,groundtruth_text))

groundtruth_ts_ref = image_groundtruth_dict[rgb_id_ref]
groundtruth_data_ref = associate.return_groundtruth(groundtruth_text,groundtruth_ts_ref)
SE3_ref = Parser.generate_se3_from_groundtruth(groundtruth_data_ref)

groundtruth_ts_target = image_groundtruth_dict[rgb_id_target]
groundtruth_data_target = associate.return_groundtruth(groundtruth_text,groundtruth_ts_target)
SE3_target = Parser.generate_se3_from_groundtruth(groundtruth_data_target)

SE3_ref_target = SE3.pose_pose_composition_inverse(SE3_ref,SE3_target)

print('*'*80)
print('GROUND TRUTH\n')
print(SE3_ref_target)
print('*'*80)






