from Benchmark import associate
import os


bench_path = '/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/rccar_26_09_18/'
xyz_dataset = 'marc_4_full/'


rgb_text = bench_path+xyz_dataset+'rgb.txt'
depth_text = bench_path+xyz_dataset+'depth_large.txt'
with_duplicates_for_steering = True

matches = associate.match(rgb_text,depth_text,with_duplicates=with_duplicates_for_steering)
match_file = bench_path+xyz_dataset+'matches_rgb.txt'

try:
    os.remove(match_file)
except OSError:
    pass

with open(match_file, 'w') as f:
    f.write('# rgb_timestamp depth_timesamp\n')
    for (rgb_ts,depth_ts) in matches:
        rgb_ts_string = f'{rgb_ts:.9f}'
        depth_ts_string = f'{depth_ts:.9f}'
        f.write("%s %s\n" % (rgb_ts_string,depth_ts_string))




