#!/usr/bin/env python
import os
import random
import json

import tqdm

import ltron.settings as settings
from ltron.bricks.brick_scene import BrickScene

random.seed(1234567890)

existing_sets = {}
lte_500_path = settings.datasets['lte_500']
omr_ldraw = os.path.join(os.path.dirname(lte_500_path), 'ldraw')
all_sets = sorted(os.listdir(omr_ldraw))
good_sets = []
shape_ids = {}
max_edges = 0
all_colors = set()
for set_name in tqdm.tqdm(all_sets):
    try:
        scene = BrickScene(renderable=False, track_snaps=True)
        scene.import_ldraw(os.path.join(omr_ldraw, set_name))
    except:
        print('bad: %s'%set_name)
        continue
    
    num_parts = len(scene.instances)
    if num_parts <= 500:
        good_sets.append('ldraw/' + set_name)
        for brick_shape in scene.shape_library:
            if str(brick_shape) not in shape_ids:
                shape_ids[str(brick_shape)] = len(shape_ids)+1
        
        edges = scene.get_assembly_edges()
        try:
            max_edges = max(max_edges, edges.shape[1])
        except IndexError:
            pass
        
        colors = set(int(color) for color in scene.color_library)
        all_colors |= colors

train_sets = set(random.sample(good_sets, int(len(good_sets)*0.9)))
test_sets = set(good_sets) - train_sets

dataset_info = {
    'splits' : {
        'all' : list(sorted(good_sets)),
        'train' : list(sorted(train_sets)),
        'test' : list(sorted(test_sets)),
    },
    'max_instances_per_scene' : 500,
    'max_edges_per_scene' : max_edges,
    'shape_ids' : shape_ids,
    'colors' : sorted(list(all_colors))
}

with open(lte_500_path, 'w') as f:
    json.dump(dataset_info, f, indent=4)
