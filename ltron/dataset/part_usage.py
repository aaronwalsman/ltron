#!/usr/bin/env python
import os
import json

import numpy

import tqdm

import ltron.config as config
from ltron.bricks.brick_scene import BrickScene

omr_ldraw_directory = os.path.join(config.paths['omr'], 'ldraw')

file_names = list(sorted(os.listdir(omr_ldraw_directory)))

path_data = {}

for file_name in tqdm.tqdm(file_names):
    path = os.path.join(omr_ldraw_directory, file_name)
    scene = BrickScene(track_snaps=True)
    try:
        scene.import_ldraw(path)
    except:
        print('Unable to load path: %s'%path)
        continue
    
    path_data[file_name] = {}
    
    path_data[file_name]['brick_counts'] = {}
    for instance_id, instance in scene.instances.items():
        brick_type = str(instance.brick_type)
        if brick_type not in path_data[file_name]['brick_counts']:
            path_data[file_name]['brick_counts'][brick_type] = 0
        path_data[file_name]['brick_counts'][brick_type] += 1
    
    edges = scene.get_all_edges()
    path_data[file_name]['edge_data'] = {}
    for a, b in edges.T:
        instance_a = scene.instances[a]
        brick_type_a = str(instance_a.brick_type)
        transform_a = instance_a.transform
        
        instance_b = scene.instances[b]
        brick_type_b = str(instance_b.brick_type)
        transform_b = instance_b.transform
        
        ab = numpy.dot(numpy.linalg.inv(transform_a), transform_b)
        
        edge_string = '%s,%s'%(brick_type_a, brick_type_b)
        if edge_string not in path_data[file_name]['edge_data']:
            path_data[file_name]['edge_data'][edge_string] = []
        path_data[file_name]['edge_data'][edge_string].append(ab.tolist())

with open(os.path.join(config.paths['omr'], 'scene_data.json'), 'w') as f:
    json.dump(path_data, f, indent=2)
