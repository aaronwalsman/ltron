#!/usr/bin/env python
import tqdm

from ltron.dataset.build_dataset import build_dataset
from ltron.dataset.paths import get_dataset_paths
from ltron.bricks.brick_scene import BrickScene

paths = get_dataset_paths('tiny_turbos3', 'all')

brick_counts = []

for path in tqdm.tqdm(paths):
    scene = BrickScene()
    scene.import_ldraw(path)
    brick_counts.append((len(scene.instances), path))

lte_64 = [path for count, path in brick_counts if count <= 64]
build_dataset('tt_64', '/media/awalsman/data_drive/ltron/data/download/collections/OMR', lte_64, 20)

'''
for i in range(120):
    count = len([count for count in brick_counts if count <= i])
    p = '|' * (count // 2)
    if count % 2 == 1:
        p += '.'
    print('%i: %s'%(i,p))
'''
