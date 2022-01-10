#!/usr/bin/env python

'''
Notes:
2436a and 2436b both exist in dataset
30027a and 30027b both exist
4081a and 4081b both exist
4085a and 4085c both exit
'''

'''
reject = [
    '2335.dat', # flag no studs, could use 3957a.dat
    '2430.dat', # HINGE NEEDS SUBASSEMBLY
    '2460.dat', # HELLICOPTER SPINNER NEEDS SUBASSEMBLY
    '2479.dat', # HELICOPTER SPINNER NEEDS SUBASSEMBLY
    '2584.dat', # winch
    '2585.dat', # winch
    '2926.dat', # new axle
    '30028.dat', # new tire
    '30031.dat', # handlebar, combine with 2555.dat
    '30350a.dat', # bigger flag no studs, could use 3957a.dat
    '30374.dat', # bar, no studs
    '30383.dat', # hinge part
    '30389b.dat', # hinge part
    '30552.dat', # tow truck part subassembly with 30553.dat
    '30553.dat', # tow truck part 30552.dat
    '30648.dat', # giant tire
    '3139.dat', # tire
    '32013.dat', # technic connector
    '32014.dat', # technic connector
    '32062.dat', # technic axle
    '32184.dat', # more technic
    '32523.dat', # more technic
    '3641.dat', # tire
    '3707.dat', # technic axle
    '3749.dat', # technic pin
    '3937.dat', # hinge with V
    '3938.dat', # hinge with ^
    '41669.dat', # technic spike thing
    
]
'''

import os

import tqdm

import numpy

import ltron.settings as settings
from ltron.dataset.paths import get_dataset_info
from ltron.bricks.brick_scene import BrickScene
from ltron.geometry.scene_sampler import (
        sample_scene,
        SingleSubAssemblySampler)

'''
spacing = 100
if __name__ == '__main__':
    scene = BrickScene(renderable=False)
    info = get_dataset_info('tiny_turbos3')
    for i, name in enumerate(sorted(info['shape_ids'])):
        brick_shape = scene.add_brick_shape(name)
        t = numpy.eye(4)
        t[1,1] = -1.0
        t[2,2] = -1.0
        t[0,3] = i % 20 * spacing
        t[2,3] = i // 20 * spacing
        scene.add_instance(brick_shape, 4, t)
    
    scene.export_ldraw('./tmp.ldr')
'''

if __name__ == '__main__':
    scene = BrickScene(renderable=False)
    info = get_dataset_info('tiny_turbos3')
    samplers = [SingleSubAssemblySampler(class_name)
            for class_name in info['shape_ids']]
    
    ldraw_path = os.path.join(
            settings.paths['data'], 'rando_tt3', 'ldraw')
    if not os.path.isdir(ldraw_path):
        os.makedirs(ldraw_path)
    
    colors = [int(c) for c in info['all_colors']]
    
    for i in tqdm.tqdm(range(1000)):
        scene = sample_scene(samplers, (20, 60), colors, timeout=20)
        scene_path = os.path.join(ldraw_path, 'rando_tt3_%06i.mpd'%i)
        scene.export_ldraw(scene_path)
