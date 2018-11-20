from os import listdir
from os.path import isfile, join


def get_files_from_directory(dir_path, delimiter=''):
    if not delimiter:
        files = [f for f in listdir(dir_path) if isfile(join(dir_path, f)) and is_valid_file(f)]
    else:
        files = [concat_all_but_last(f.split(delimiter)) for f in listdir(dir_path) if isfile(join(dir_path, f)) and is_valid_file(f)]

    files_float = list(map(lambda string: float(string),files))
    files_float.sort()
    return files_float

def generate_files_to_load_match(rgb_list, start, max_count, offset, ground_truth_dict, match_dict, reverse=False):
    ref_files_to_load = []
    target_files_to_load = []
    ref_file_failed_to_load = []

    i_ref = start
    i_target = i_ref+offset
    max = start + max_count

    while i_ref < max:
        ref_file = rgb_list[i_ref]
        target_file = rgb_list[i_target]

        if validate_match(ref_file, ground_truth_dict,match_dict):
            ref_files_to_load.append(ref_file)
            i_ref+=offset
        else:
            ref_file_failed_to_load.append(ref_file)
            i_ref+=1
            i_target+=1
            continue

        if validate_match(target_file, ground_truth_dict,match_dict):
            target_files_to_load.append(target_file)
            i_target+=offset
        else:
            i_target+=1

        assert i_ref < i_target

    ref_len = len(ref_files_to_load)
    target_len = len(target_files_to_load)

    # fill up if the last i_target happens to be invalid
    while target_len < ref_len:
        target_file = rgb_list[i_target]
        if validate(target_file, ground_truth_dict,match_dict):
            target_files_to_load.append(target_file)
            i_target+=offset
            target_len += 1
        else:
            i_target += 1

    assert len(ref_files_to_load) == len(target_files_to_load)

    if reverse:
        t = ref_files_to_load[::-1]
        ref_files_to_load = target_files_to_load[::-1]
        target_files_to_load = t
        ref_file_failed_to_load = ref_file_failed_to_load[::-1]


    return ref_files_to_load, target_files_to_load , ref_file_failed_to_load


def generate_files_to_load(rgb_list, start, max_count, offset, ground_truth_dict):
    ref_files_to_load = []
    target_files_to_load = []
    ref_file_failed_to_load = []

    i_ref = start
    i_target = i_ref+offset
    max = start + max_count

    while i_ref < max:
        ref_file = rgb_list[i_ref]
        target_file = rgb_list[i_target]

        if validate(ref_file, ground_truth_dict):
            ref_files_to_load.append(ref_file)
            i_ref+=offset
        else:
            ref_file_failed_to_load.append(ref_file)
            i_ref+=1
            i_target+=1
            continue

        if validate(target_file, ground_truth_dict):
            target_files_to_load.append(target_file)
            i_target+=offset
        else:
            i_target+=1

        assert i_ref < i_target

    ref_len = len(ref_files_to_load)
    target_len = len(target_files_to_load)

    # fill up if the last i_target happens to be invalid
    while target_len < ref_len:
        target_file = rgb_list[i_target]
        if validate(target_file, ground_truth_dict):
            target_files_to_load.append(target_file)
            i_target+=offset
            target_len += 1
        else:
            i_target += 1

    assert len(ref_files_to_load) == len(target_files_to_load)

    return ref_files_to_load, target_files_to_load , ref_file_failed_to_load


def validate_match(key, ground_truth_dict,match_dict):
    does_ground_truth_exist = key in ground_truth_dict
    does_depth_match_exist = key in match_dict
    return does_ground_truth_exist and does_depth_match_exist

def validate(key, ground_truth_dict):
    does_ground_truth_exist = key in ground_truth_dict
    return does_ground_truth_exist

def is_valid_file(file_name):
    return not file_name == '.DS_Store'

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


