from Benchmark import associate
import os


bench_path = '/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/rccar_15_11_18/'
xyz_dataset = 'marc_1_full/'


rgb_text = bench_path+xyz_dataset+'rgb_rect.txt'
depth_text = bench_path+xyz_dataset+'encoder.txt'
with_duplicates_for_steering = True

matches = associate.match(rgb_text,depth_text,with_duplicates=with_duplicates_for_steering)
match_file = bench_path+xyz_dataset+'encoder_rgb_rect.txt'

try:
    os.remove(match_file)
except OSError:
    pass

with open(match_file, 'w') as f:
    f.write('# rgb_timestamp depth_timestamp\n')
    for (rgb_ts,depth_ts) in matches:
        f.write("%s %s\n" % (rgb_ts,depth_ts))




