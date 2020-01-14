import os
import numpy as np
import re, time
from osureader import test_process_path, read_and_save_osu_file
from lib import read_some_npzs_and_preprocess, read_npz_list, train_test_split, \
    read_some_npzs_and_preprocess_m, read_npz_list_m, train_test_split_m
from config import *

def save_map_data(maplist_dir, level):
    maplist_name = os.path.join(maplist_dir, "maplist_{}.txt".format(level))
    with open(maplist_name) as fp:
        fcont = fp.readlines()

    results = [];
    for line in fcont:
        results.append(line)

    print("Number of {} beatmaps: {}".format(level, len(results)))

    data_path = os.path.join(mapdata_path, level)
    if os.path.exists(data_path) == False:
        os.mkdir(data_path)

    for k, mname in enumerate(results):
        try:
            start = time.time()
            read_and_save_osu_file(mname.strip(), filename=os.path.join(data_path, str(k)), divisor=divisor);
            end = time.time()
            print("Map data #" + str(k) + " saved! time = " + str(end - start) + " secs");
        except Exception as e:
            print("Error on #{}, path = {}, error = {}".format(str(k), mname.strip(), e));

def write_maplist(map_dir, level, maplist_dir, unzip=False):

    osz_dir = os.path.join(map_dir, 'osz')

    if unzip:
        # mkdir
        if os.path.exists(osz_dir) == False:
            os.mkdir(osz_dir)
        # unzip .rar
        files = os.listdir(map_dir)
        for name in files:
            if name.endswith('.rar'):
                os.system("unrar x -y '{}' '{}'".format(os.path.join(map_dir, name), osz_dir))
                os.system("mv '{}' ~/Trash/".format(os.path.join(map_dir, name)))

        # unzip .osz
        osz_files = os.listdir(osz_dir)
        for file in osz_files:
            id = file.split(' ')[0]
            name = file[:-4]
            folder = os.path.join(map_dir, name)
            if os.path.exists(folder) == False:
                os.mkdir(folder)
            file_f = os.path.join(osz_dir, file)
            cmd = "unzip -d '{}' -n '{}' '*.mp3'".format(folder, file_f)
            os.system(cmd)
            cmd = "unzip -d '{}' -n '{}' '*.osu'".format(folder, file_f)
            os.system(cmd)
            os.system("mv '{}' ~/Trash/".format(file_f))

    maplist_name = os.path.join(maplist_dir, "maplist_{}.txt".format(level))
    with open(maplist_name, 'w') as F:
        folders = os.listdir(map_dir)
        for folder in folders:
            if os.path.isdir(os.path.join(map_dir, folder)) == False:
                continue
            files = os.listdir(os.path.join(map_dir, folder))
            for file in files:
                if file.endswith('.osu') and level in file:
                    line = os.path.join(map_dir, folder, file) + '\n'
                    print(line)
                    F.write(line)
                    break # one (if any) beatmap per song

    return

def data_split_save(filename):
    if model_choose == "ConvLstm":
        train_file_list = read_npz_list()
        train_data2, div_data2, train_labels2 = read_some_npzs_and_preprocess(train_file_list)
        (new_train_data, new_div_data, new_train_labels), (test_data, test_div_data, test_labels) = train_test_split(
            train_data2, div_data2, train_labels2)

        np.savez_compressed(filename + 'train.npz', spec=new_train_data, div=new_div_data, label=new_train_labels)
        np.savez_compressed(filename + 'test.npz', spec=test_data, div=test_div_data, label=test_labels)
    else:
        train_file_list = read_npz_list_m()
        train_data2, div_data2, train_labels2 = read_some_npzs_and_preprocess_m(train_file_list)
        (new_train_data, new_div_data, new_train_labels), (test_data, test_div_data, test_labels) = train_test_split_m(
            train_data2, div_data2, train_labels2)

        np.savez_compressed(filename + 'train_m.npz', spec=new_train_data, div=new_div_data, label=new_train_labels)
        np.savez_compressed(filename + 'test_m.npz', spec=test_data, div=test_div_data, label=test_labels)


if __name__ == "__main__":
    # create maplist
    # write_maplist(map_dir=map_path, level=level, maplist_dir=mapdata_path, unzip=False)
    # test node
    # test_process_path("node")
    # save map data
    # save_map_data(maplist_dir=mapdata_path, level=level)
    # split training / testing and save
    data_split_save(data_path) # change level in lib.py
    print("Done!")