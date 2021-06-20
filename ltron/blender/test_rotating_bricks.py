#!/usr/bin/env python
import re
import os

src_dir = '/media/awalsman/data_drive/brick-gym/data/splendor/meshes'
check_dir = '/media/awalsman/data_drive/brick-gym/data/splendor/meshes_new_all'

src_files = sorted([f for f in os.listdir(src_dir) if f[-4:] == '.obj'])

matches = 0
mismatches = 0
for src_file in src_files:
    src_path = os.path.join(src_dir, src_file)
    check_path = os.path.join(check_dir, src_file)
    
    '''
    src_contents = open(src_path).read()
    src_contents = re.sub('o [^\n]*\n', src_contents, '')
    check_contents = open(check_path).read()
    check_contents = re.sub('o [^\n]*\n', check_contents, '')
    '''
    src_lines = open(src_path).readlines()
    src_lines = [line for line in src_lines if not line.startswith('o ')]
    src_contents = ''.join(src_lines)
    check_lines = open(check_path).readlines()
    check_lines = [line for line in check_lines if not line.startswith('o ')]
    check_contents = ''.join(check_lines)
    
    if src_contents == check_contents:
        matches += 1
    else:
        print('No match!', src_file)
        mismatches += 1

print('%i matches'%matches)
print('%i mismatches'%mismatches)
