from Benchmark import associate
import os


#bench_path = '/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/rccar_15_11_18/'
bench_path = '/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/VO_Bench/'
dataset = 'rgbd_dataset_freiburg1_xyz/'


rgb_text = bench_path + dataset + 'rgb.txt'
depth_text = bench_path + dataset + 'depth.txt'
with_duplicates_for_steering = True

matches = associate.match(rgb_text,depth_text,with_duplicates=with_duplicates_for_steering)
match_file = bench_path + dataset + 'matches_with_duplicate.txt'

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




