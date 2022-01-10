#!/usr/bin/env python
import os
import random
import json

import tqdm

import ltron.settings as settings
from ltron.bricks.brick_scene import BrickScene

dataset_scenes = ['8661-1 - Carbon Star.mpd']
carbon_star_path = settings.datasets['carbon_star']
omr_ldraw = os.path.join(os.path.dirname(carbon_star_path), 'ldraw')

scene = BrickScene()
instance_counts = {}
instances_per_scene = []
all_colors = set()
for dataset_scene in dataset_scenes:
    scene.clear_instances()
    scene.clear_assets()
    scene.import_ldraw(os.path.join(omr_ldraw, dataset_scene))
    instances_per_scene.append(len(scene.instances))
    print('%s has %i instances'%(dataset_scene, len(scene.instances)))
    for instance_id, instance in scene.instances.items():
        brick_shape = instance.brick_shape
        if str(brick_shape) not in instance_counts:
            instance_counts[str(brick_shape)] = 0
        instance_counts[str(brick_shape)] += 1
        all_colors.add(instance.color)

print('Average instances per model: %f'%(
        sum(instances_per_scene)/len(instances_per_scene)))
print('Min/Max instances per model: %i, %i'%(
        min(instances_per_scene), max(instances_per_scene)))

sorted_instance_counts = reversed(sorted(
        (value, key) for key, value in instance_counts.items()))

print('Part usage statistics:')
for count, brick_shape in sorted_instance_counts:
    print('%s: %i'%(brick_shape, count))

print('%i total brick shapes'%len(instance_counts))

random.seed(1234)

all_scenes = ['ldraw/' + dataset_scenes[0]]
dataset_info = {
    'splits' : {
        'all' : all_scenes,
    },
    'max_instances_per_scene' : max(instances_per_scene),
    'shape_ids':dict(
            zip(sorted(instance_counts.keys()),
            range(1, len(instance_counts)+1))),
    'all_colors':list(sorted(all_colors, key=int))
}

with open(carbon_star_path, 'w') as f:
    json.dump(dataset_info, f, indent=4)
