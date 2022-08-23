#!/usr/bin/env python
import json

data = json.load(open('model_stat_full.json'))

print('number of entries: %i'%len(data.keys()))

min_parts = 16
max_parts = 256

num_too_small = 0
num_just_right = 0
num_too_big = 0

for model_name, contents in data.items():
    if model_name == 'error':
        continue
    
    if contents['count'] < min_parts:
        continue
    
    elif contents['count'] < max_parts:
        num_just_right += 1
        continue
    
    else:
        for submodel, count in contents['submodel'].items():
            if count > min_parts and count < max_parts:
                num_just_right += 1

print('number of good submodels: %i'%num_just_right)
