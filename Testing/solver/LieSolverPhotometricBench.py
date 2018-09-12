from Benchmark import associate
import cv2

bench_path = '/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Bench/'
xyz_dataset = 'rgbd_dataset_freiburg1_xyz/'
rgb_folder = 'rgb/'
depth_folder = 'depth/'

dataset_root = bench_path+xyz_dataset
rgb_text = dataset_root +'rgb.txt'
depth_text = dataset_root+'depth.txt'
match_text = dataset_root+'matches.txt'

rgb_id_ref = 1305031102.175304
rgb_id_target = 1305031102.411258

rgb_ref_file_path , depth_ref_file_path = associate.return_rgb_depth_from_rgb_selection(rgb_text,depth_text, match_text, dataset_root, rgb_id_ref)
rgb_target_file_path , depth_target_file_path = associate.return_rgb_depth_from_rgb_selection(rgb_text,depth_text, match_text, dataset_root, rgb_id_target)

im_greyscale_reference = cv2.imread(rgb_ref_file_path,cv2.IMREAD_GRAYSCALE)
im_depth_reference = cv2.imread(depth_ref_file_path,cv2.IMREAD_ANYDEPTH)

im_greyscale_target = cv2.imread(rgb_target_file_path,cv2.IMREAD_GRAYSCALE)
im_depth_target = cv2.imread(depth_target_file_path,cv2.IMREAD_ANYDEPTH)

cv2.imshow('rgb ref',im_greyscale_reference)
cv2.imshow('depth ref',im_depth_reference)

cv2.imshow('rgb target',im_greyscale_target)
cv2.imshow('depth target',im_depth_target)

while True:
    k = cv2.waitKey(5) & 0xFF
    # ESC
    if k == 27:
        break

cv2.destroyAllWindows()

