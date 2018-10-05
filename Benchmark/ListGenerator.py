from os import listdir
from os.path import isfile, join


def get_files_from_directory(dir_path, delimiter=''):
    if not delimiter:
        files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    else:
        files = [concat_all_but_last(f.split(delimiter)) for f in listdir(dir_path) if isfile(join(dir_path, f))]
    return files

def generate_files_to_load(rgb_list, start, ref_max, offset, ground_truth_dict):
    ref_files_to_load = []
    target_files_to_load = []
    ref_file_failed_to_load = []

    i_ref = start
    i_target = i_ref+offset

    while i_ref < ref_max:
        ref_file = rgb_list[i_ref]
        target_file = rgb_list[i_target]

        if validate(ref_file, ground_truth_dict):
            ref_files_to_load.append(ref_file)
            i_ref+=offset
        else:
            ref_file_failed_to_load.append(ref_file)
            i_ref+=1

        if validate(target_file, ground_truth_dict):
            target_files_to_load.append(target_file)
            i_target+=offset
        else:
            i_target+=1

        assert i_ref < i_target

    assert len(ref_files_to_load) == len(target_files_to_load)


    return ref_files_to_load, target_files_to_load , ref_file_failed_to_load


def validate(key_string, ground_truth_dict):
    return float(key_string) in ground_truth_dict

def get_index_of_id(id, file_list):
    id_ret = -1
    for i in range(0,len(file_list)):
        if id == file_list[i]:
            id_ret = i
            break
    return id_ret

def concat_all_but_last(split_list):
    all_but_last = split_list[:-1]
    return '.'.join(all_but_last)


