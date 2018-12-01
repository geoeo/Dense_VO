from Benchmark import ListGenerator
import os


bench_path = '/Users/marchaubenstock/Workspace/Diplomarbeit_Resources/rccar_26_09_18/'
dataset = 'marc_6_full/'

#rgb_folder = 'depth/'
#rgb_text = 'depth.txt'
#rgb_folder = 'depth_rect/'
#rgb_text = 'depth_rect.txt'
rgb_folder = 'depth_rect_reg/'
rgb_text = 'depth_rect_reg.txt'
#rgb_folder = 'color_rect/'
#rgb_text = 'rgb_rect.txt'
#rgb_folder = 'color/'
#rgb_text = 'rgb.txt'

rgb_text_path = bench_path+dataset+rgb_text

full_path = bench_path + dataset + rgb_folder

# file names are the time stamps
timestamps = ListGenerator.get_files_from_directory(full_path, '.')

files = list(map(lambda float: rgb_folder+f'{float:.9f}'+'.png',timestamps))

zipped = zip(timestamps,files)


try:
    os.remove(rgb_text_path)
except OSError:
    pass

with open(rgb_text_path, 'w') as f:
    f.write('#timestamp file_name\n')
    for (ts,rgb_path) in zipped:
        f.write("%s %s\n" % (ts,rgb_path))




