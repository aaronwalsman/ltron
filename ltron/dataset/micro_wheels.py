#!/usr/bin/env python
import ltron.settings as settings
from ltron.dataset.build_dataset import build_dataset

dataset_paths = [
    "4096 - Micro Wheels - AB Forklift.mpd",
    "4096 - Micro Wheels - AB Loader.mpd",
    "4096 - Micro Wheels - AB Truck and Trailer.mpd",
    "4096 - Micro Wheels - EB Combine Harvester.mpd",
    "4096 - Micro Wheels - EB Crane.mpd",
    "4096 - Micro Wheels - EB Tractor and Trailer.mpd",
    "4096 - Micro Wheels - EB Truck.mpd",
    "4096 - Micro Wheels - QB 4WD.mpd",
    "4096 - Micro Wheels - QB Cement Mixer.mpd",
    "4096 - Micro Wheels - QB Formula1.mpd",
    "4096 - Micro Wheels - QB Roadster 1.mpd",
    "4096 - Micro Wheels - QB Roadster 2.mpd",
    "4096 - Micro Wheels - QB Truck.mpd"
]

from ltron.bricks.brick_scene import BrickScene
import os
import random
brick_shapes = {}
for path in dataset_paths:
    scene = BrickScene()
    scene.import_ldraw(os.path.join(settings.paths['omr'], 'ldraw', path))
    brick_shapes[path] = set()
    for instance_id, instance in scene.instances.items():
        brick_shape = str(instance.brick_shape)
        brick_shapes[path].add(brick_shape)

while True:
    test_set = random.sample(dataset_paths, 5)
    train_set = set(dataset_paths) - set(test_set)
    
    test_parts = set.union(*[brick_shapes[path] for path in test_set])
    train_parts = set.union(*[brick_shapes[path] for path in train_set])
    
    if not len(test_parts - train_parts):
        print('Train:')
        for path in train_set:
            print(path)
        
        print('Test:')
        for path in test_set:
            print(path)
        
        break

test_set = [os.path.join('ldraw', path) for path in test_set]
build_dataset('micro_wheels', settings.paths['omr'], dataset_paths, test_set)
