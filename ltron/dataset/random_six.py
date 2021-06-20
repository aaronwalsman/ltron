#!/usr/bin/env python
import os

import tqdm

import ltron.settings as settings
from ltron.geometry.scene_sampler import sample_scene, SingleSubAssemblySampler

samplers = [
    SingleSubAssemblySampler('54383.dat'),
    SingleSubAssemblySampler('41770.dat'),
    SingleSubAssemblySampler('2450.dat'),
    SingleSubAssemblySampler('43722.dat'),
    SingleSubAssemblySampler('2436.dat'),
    SingleSubAssemblySampler('4081.dat'),
]

ldraw_path = os.path.join(
    settings.paths['data'], 'random_six', 'ldraw')
if not os.path.isdir(ldraw_path):
    os.makedirs(ldraw_path)

for i in tqdm.tqdm(range(10000)):
    scene = sample_scene(samplers, (4,8), ['1','4','14'], timeout=10)
    scene_path = os.path.join(ldraw_path, 'random_six_%06i.mpd'%i)
    scene.export_ldraw(scene_path)
