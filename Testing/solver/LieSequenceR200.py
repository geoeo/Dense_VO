import numpy as np
from Numerics import Utils, SE3
from Camera import Intrinsic, Camera
from VisualOdometry import Frame, SolverThreadManager
from Benchmark import Parser, associate, ListGenerator
from Visualization import Visualizer


bench_path = '/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/rccar_26_09_18/'
xyz_dataset = 'marc_1_full/'
rgb_folder = 'color/'
depth_folder = 'depth_full/'

dataset_root = bench_path+xyz_dataset
rgb_text = dataset_root +'rgb.txt'
groundtruth_text = dataset_root+'groundtruth.txt'

rgb_folder = dataset_root+rgb_folder
depth_folder = dataset_root+depth_folder

rgb_files = ListGenerator.get_files_from_directory(rgb_folder, delimiter='.')
depth_files = ListGenerator.get_files_from_directory(depth_folder, delimiter='.')

rgb_file_total = len(rgb_files)
depth_file_total = len(depth_files)

image_groundtruth_dict = dict(associate.match(rgb_text, groundtruth_text))

start = ListGenerator.get_index_of_id(966815.749698534,rgb_files)