#!/usr/bin/env python
import tqdm

from brick_gym.dataset.paths import get_dataset_paths
from brick_gym.bricks.brick_scene import BrickScene

paths = get_dataset_paths('tiny_turbos2', 'third_four', subset=64)

brick_types = set()

scene = BrickScene()

for path in tqdm.tqdm(paths):
    scene.import_ldraw(path)

print(len(scene.brick_library))
names = sorted(scene.brick_library.keys())
for name in names:
    print(name)
