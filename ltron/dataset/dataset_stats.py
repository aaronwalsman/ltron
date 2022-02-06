#!/usr/bin/env python
import os
import json

import numpy

import tqdm

from PIL import Image

import ltron.settings as settings
from ltron.dataset.paths import get_dataset_paths, get_dataset_info
from ltron.bricks.brick_scene import BrickScene

dataset = 'tiny_turbos2'
subset = 64

train_paths = get_dataset_paths(dataset, 'train', subset=subset)
test_paths = get_dataset_paths(dataset, 'test', subset=subset)
info = get_dataset_info(dataset)

max_shape_id = max(info['shape_ids'].values())

omr_path = settings.paths['omr']
with open(os.path.join(omr_path, 'scene_data.json')) as f:
    scene_data = json.load(f)

def get_brick_stats(paths):
    brick_shapes = set()
    for path in tqdm.tqdm(paths):
        scene = BrickScene(track_snaps=True)
        scene.import_ldraw(path)
        path_brick_shapes = set(scene.shape_library.keys())
        brick_shapes |= path_brick_shapes
    
    return brick_shapes

def get_extant_edges(paths):
    extant_edge_matrix = numpy.zeros(
            (max_shape_id+1, max_shape_id+1), dtype=numpy.long)
    
    edge_counts = {}
    
    for path in tqdm.tqdm(paths):
        scene = BrickScene(track_snaps=True)
        scene.import_ldraw(path)
        
        edges = scene.get_assembly_edges()
        
        for a, b in edges.T:
            instance_a = scene.instances[a]
            brick_shape_a = str(instance_a.brick_shape)
            id_a = info['shape_ids'][brick_shape_a]
            
            instance_b = scene.instances[b]
            brick_shape_b = str(instance_b.brick_shape)
            id_b = info['shape_ids'][brick_shape_b]
            
            extant_edge_matrix[id_a, id_b] = 1
            
            if (id_a, id_b) not in edge_counts:
                edge_counts[id_a, id_b] = 0
            edge_counts[id_a, id_b] += 1
    
    return extant_edge_matrix, edge_counts

train_edge_matrix, train_edge_counts = get_extant_edges(train_paths)
test_edge_matrix, test_edge_counts = get_extant_edges(test_paths)

train_types_per_brick_shape = numpy.sum(train_edge_matrix, axis=0)
print(train_types_per_brick_shape)
print('Average connection types per brick in the train set: %f'%
        numpy.mean(train_types_per_brick_shape))

test_file_names = [
        os.path.split(path)[-1].split(':')[0]
        for path in test_paths]

train_file_names = [
        os.path.split(path)[-1].split(':')[0]
        for path in train_paths]

external_edge_augmentations = {}
external_edge_matrix = numpy.zeros(
        (max_shape_id+1, max_shape_id+1), dtype=numpy.long)

for file_name in scene_data:
    if file_name in test_file_names:
        continue
    
    if file_name in train_file_names:
        continue
    
    '''
    for brick_shape in info['shape_ids']:
        if brick_shape in scene_data[file_name]['brick_counts']:
            scenes_to_mine.append(file_name)
            break
    '''
    
    for edge in scene_data[file_name]['edge_data']:
        a, b = edge.split(',')
        if a in info['shape_ids'] and b in info['shape_ids']:
            if a not in external_edge_augmentations:
                external_edge_augmentations[a] = {}
            if b not in external_edge_augmentations[a]:
                external_edge_augmentations[a][b] = []
            external_edge_augmentations[a][b].extend(
                    scene_data[file_name]['edge_data'][edge])
            
            id_a = info['shape_ids'][a]
            id_b = info['shape_ids'][b]
            external_edge_matrix[id_a, id_b] = 1

draw_image = numpy.zeros((max_shape_id+1, max_shape_id+1, 3), numpy.uint8)
draw_image[:,:,0] = test_edge_matrix.astype(numpy.uint8)*255
draw_image[:,:,2] = train_edge_matrix.astype(numpy.uint8)*255
draw_image[:,:,1] = external_edge_matrix.astype(numpy.uint8)*255
Image.fromarray(draw_image).save('extant_edge_matrix.png')
Image.fromarray(draw_image[:,:,0]).save('test_edge_matrix.png')
Image.fromarray(draw_image[:,:,2]).save('train_edge_matrix.png')
Image.fromarray(draw_image[:,:,1]).save('augment_edge_matrix.png')

draw_image[:,:,1] = (test_edge_matrix + train_edge_matrix == 2).astype(
        numpy.uint8)*255
Image.fromarray(draw_image).save('train_test_edge_matrix.png')

train_edge_set = set(train_edge_counts.keys())
test_edge_set = set(test_edge_counts.keys())
print('Covered test edges: %i/%i'%(
        len(train_edge_set & test_edge_set), len(test_edge_set)))

'''
with open('augmentations.json', 'w') as f:
    json.dump(external_edge_augmentations, f)
'''

'''
print(len(brick_shapes))
for name in list(sorted(brick_shapes)):
    print(name)
'''
