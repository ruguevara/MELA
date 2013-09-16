# coding: utf-8
from glob import glob
import os
import sys
import numpy as np

MAPPING = {
    '.': 0,
    '%': 1
}

BASE_PATH = 'mazes'

def convert_map(maze_path):
    def converter(char):
        return MAPPING.get(char, 0)
    rows = []
    for line in open(maze_path):
        if line.startswith('m'):
            line = line[1:].strip()
            row = [converter(char) for char in line]
            rows.append(row)
    maze = np.array(rows, dtype=np.uint8)
    _, maze_name = os.path.split(maze_path)
    base_name, ext = os.path.splitext(maze_name)
    our_path = os.path.join(BASE_PATH, base_name)
    np.save(our_path, maze)

def convert_maps(dir):
    for maze in glob(os.path.join(dir, '*.map')):
        convert_map(maze)

if __name__=="__main__":
    convert_maps(sys.argv[1])
