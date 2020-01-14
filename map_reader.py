import os, re, time
from osureader import *
import sys

DIVISOR = 4

def read_new_map(file_path):
    start = time.time()
    read_and_save_osu_tester_file(file_path.strip(), filename="mapthis", divisor=DIVISOR)
    end = time.time()
    print("Map data saved! time = " + str(end - start) + " secs.")

def set_ffmpeg_path(ffmpeg_path):

    GLOBAL_VARS["ffmpeg_path"] = ffmpeg_path
    test_process_path(GLOBAL_VARS["ffmpeg_path"])

def read_main(file_path, ffmpeg_path):
    set_ffmpeg_path(ffmpeg_path)
    read_new_map(file_path)


if __name__ == "__main__":
    osu_fn = sys.argv[1]
    ffmpeg_path = sys.argv[2]
    read_main(osu_fn, ffmpeg_path)