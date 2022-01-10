#!/usr/bin/env python
import random
import math
import os

import tqdm

import ltron.settings as settings
from ltron.dataset.paths import get_dataset_info
from ltron.bricks.brick_scene import BrickScene
from ltron.geometry.scene_sampler import (
        sample_scene,
        SingleSubAssemblySampler,
        AntennaSampler,
        SpinnerPlateSampler,
        FolderSampler,
        RegularAxleWheelSampler,
        WideAxleWheelSampler)

micro_wheels_info = get_dataset_info('micro_wheels')
colors = [int(c) for c in micro_wheels_info['all_colors']]

brick_shapes = micro_wheels_info['shape_ids'].keys()
#samplers = [
#    SingleSubAssemblySampler(brick_shape) for brick_shape in brick_shapes]

samplers = [
    AntennaSampler,
    SpinnerPlateSampler,
    FolderSampler,
    # three different wheel shapes, so repeating three times
    RegularAxleWheelSampler,
    RegularAxleWheelSampler,
    RegularAxleWheelSampler,
    # three different wheel shapes, so repeating three times
    WideAxleWheelSampler,
    WideAxleWheelSampler,
    WideAxleWheelSampler,
    SingleSubAssemblySampler('2412b.dat'),
    SingleSubAssemblySampler('2436a.dat'),
    SingleSubAssemblySampler('2540.dat'),
    SingleSubAssemblySampler('2877.dat'),
    SingleSubAssemblySampler('3001.dat'),
    SingleSubAssemblySampler('3002.dat'),
    # WHEEL 30027b.dat
    # TIRE 30028.dat
    SingleSubAssemblySampler('3003.dat'),
    SingleSubAssemblySampler('3004.dat'),
    SingleSubAssemblySampler('3010.dat'),
    SingleSubAssemblySampler('3020.dat'),
    SingleSubAssemblySampler('3021.dat'),
    SingleSubAssemblySampler('3022.dat'),
    SingleSubAssemblySampler('3023.dat'),
    SingleSubAssemblySampler('3034.dat'),
    SingleSubAssemblySampler('3039.dat'),
    SingleSubAssemblySampler('3040b.dat'),
    SingleSubAssemblySampler('3062b.dat'),
    SingleSubAssemblySampler('3065.dat'), # VISUALLY THE SAME 2x1 brick as 3004
    SingleSubAssemblySampler('3068b.dat'),
    SingleSubAssemblySampler('3069b.dat'),
    SingleSubAssemblySampler('3176.dat'),
    SingleSubAssemblySampler('3460.dat'),
    SingleSubAssemblySampler('3622.dat'),
    SingleSubAssemblySampler('3623.dat'),
    # TIRE 3641.dat
    SingleSubAssemblySampler('3660.dat'),
    SingleSubAssemblySampler('3665a.dat'),
    SingleSubAssemblySampler('3666.dat'),
    # SPINNER_ROTOR 3679.dat
    # SPINNER_HOLDER 3680.dat
    SingleSubAssemblySampler('3710.dat'),
    SingleSubAssemblySampler('3788.dat'), # WHEEL COVERS, COULD GO WITH ASSEMBLY
    SingleSubAssemblySampler('3794a.dat'),
    SingleSubAssemblySampler('3795.dat'),
    # FOLDER_HOLDER 3937.dat
    # FOLDER_ROTOR 3938.dat
    SingleSubAssemblySampler('3941.dat'),
    SingleSubAssemblySampler('3942c.dat'),
    SingleSubAssemblySampler('4032a.dat'),
    SingleSubAssemblySampler('4070.dat'),
    SingleSubAssemblySampler('4081b.dat'),
    SingleSubAssemblySampler('4085c.dat'),
    SingleSubAssemblySampler('41769.dat'),
    SingleSubAssemblySampler('41770.dat'),
    SingleSubAssemblySampler('41855.dat'),
    SingleSubAssemblySampler('4286.dat'),
    SingleSubAssemblySampler('43719.dat'),
    SingleSubAssemblySampler('4589.dat'),
    # ANTENNA_HOLDER 4592.dat
    # ANTENNA 4593.dat
    # AXLE 4600.dat
    SingleSubAssemblySampler('4623.dat'),
    # WHEEL 4624.dat
    SingleSubAssemblySampler('4865a.dat'),
    # WHEEL 6014.dat
    # TIRE 6015.dat
    SingleSubAssemblySampler('6141.dat'),
    # AXLE 6157.dat
    SingleSubAssemblySampler('6231.dat'),
]

ldraw_path = os.path.join(
        settings.paths['data'], 'rando_micro_wheels', 'ldraw_new')
if not os.path.isdir(ldraw_path):
    os.makedirs(ldraw_path)

for i in tqdm.tqdm(range(1000)):
    scene = sample_scene(samplers, (20,30), colors, timeout=10)
    scene_path = os.path.join(ldraw_path, 'rando_micro_wheels_%06i.mpd'%i)
    scene.export_ldraw(scene_path)
