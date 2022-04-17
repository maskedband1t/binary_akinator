#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, pickle
from pathlib import Path

pickle_directory = "pickled_files"
script_path = Path(__file__).parent.parent
"""
  strips filename to first reverse occurance of sep, 
  params: 
        filename
        sep
        reverse (bool) -> set to False if forward order towards last occurance of sep
  ex. f/hello.txt 
                reverse=True -> hello.txt
                reverse=False -> f      
"""


def strip_filename(filename: str, sep: str, reverse: bool) -> str:
    if reverse:
        filename_base = filename.split(sep).pop()
    else:
        filename_base = (filename.rpartition(sep))[0]
    return filename_base


"""
  reads pickle and returns object, in our case we are storing strings
"""


def read_pickle(filename: str) -> str:
    full_path = "%s/%s/%s" % (script_path, pickle_directory, filename)
    with open(full_path, "rb") as p:
        unpickled = pickle.load(p)  # append to list of strings
        print("read pickle @ %s" % (full_path))
        return unpickled


file_hash = sys.argv[1]
set_of_strings = []
set_of_hashnames = []
print("back in main")
for filename in os.listdir("../pickled_files"):
    if file_hash in filename:
        imported_lib_string = read_pickle(filename)
        result = open("pickle_validator_result.txt", "w")
        result.write(imported_lib_string)
        result.close()

print("done loading in library, created sets")
