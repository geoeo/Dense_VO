from Benchmark import associate
import os


bench_path = '/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/rccar_15_11_18/'
xyz_dataset = 'marc_5_full/'
rgb_folder = 'color_rect/'
depth_folder = 'depth_rect_reg/'

rgb_text = bench_path+xyz_dataset+'rgb.txt'
depth_text = bench_path+xyz_dataset+'depth.txt'

matches = associate.match(rgb_text,depth_text)
match_file = bench_path+xyz_dataset+'matches_rect_reg.txt'

try:
    os.remove(match_file)
except OSError:
    pass

with open(match_file, 'w') as f:
    f.write('# rgb_timestamp depth_timestamp\n')
    for (rgb_ts,depth_ts) in matches:
        f.write("%s %s\n" % (rgb_ts,depth_ts))




