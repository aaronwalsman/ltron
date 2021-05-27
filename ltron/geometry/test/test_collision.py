#!/usr/bin/env python
import os

import numpy

import splendor.buffer_manager_egl as buffer_manager_egl

import brick_gym.config as config
from brick_gym.bricks.brick_scene import BrickScene
from brick_gym.geometry.collision import check_collision

manager = buffer_manager_egl.initialize_shared_buffer_manager()
scene = BrickScene(renderable=True)
scene.import_ldraw(os.path.join(
        config.paths['omr'], 'ldraw', '8661-1 - Carbon Star.mpd'))

instance = scene.instances['29']
#instance.transform[2,3] -= 10
scene.renderer.set_instance_transform(str(instance), instance.transform)
snap = instance.get_snaps()[1]

collision = check_collision(
        scene,
        [instance],
        snap.transform,
        'F',
        resolution=(512,512),
        dump_images='test')

print(collision)
