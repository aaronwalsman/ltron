#!/usr/bin/env python
import os
import random
random.seed(1234567890)

import tqdm

import ltron.settings as settings
import ltron.dataset.scales as scales
from ltron.bricks.brick_scene import BrickScene
from ltron.geometry.scene_sampler import sample_scene, SingleSubAssemblySampler

samplers_s006 = [
    SingleSubAssemblySampler('54383.dat'),
    SingleSubAssemblySampler('41770.dat'),
    SingleSubAssemblySampler('2450.dat'),
    SingleSubAssemblySampler('43722.dat'),
    SingleSubAssemblySampler('2436.dat'),
    SingleSubAssemblySampler('4081.dat'),
]

colors_c006 = ['1','4','7','14','22','25']


def make_mpd(
    collection,
    name,
    samplers,
    colors,
    start_scene,
    num_scenes,
    min_bricks,
    max_bricks
):
    ldraw_path = os.path.join(settings.collections[collection], 'ldraw_new')
    if not os.path.isdir(ldraw_path):
        os.makedirs(ldraw_path)
    
    scene = BrickScene(
        renderable=True, track_snaps=True, collision_checker=True)
    for i in tqdm.tqdm(range(start_scene, num_scenes+start_scene)):
        sample_scene(
            scene,
            samplers,
            (min_bricks, max_bricks),
            colors,
            debug=False,
            timeout=10,
        )
        scene_path = os.path.join(ldraw_path, '%s_%06i.mpd'%(name, i))
        scene.export_ldraw(scene_path)
        
        scene.clear_instances()

def make_scale(collection, scale, start_scene, num_scenes):
    num_bricks = getattr(scales, '%s_max_bricks'%scale)
    make_mpd(
        collection,
        scale,
        samplers_s006,
        colors_c006,
        start_scene,
        num_scenes,
        num_bricks,
        num_bricks,
    )

#make_scale('random_construction_6_6', 'pico', 50000, 5000)
#make_scale('random_construction_6_6', 'nano', 0, 55000)
make_scale('random_construction_6_6', 'micro', 0, 55000)
