#!/usr/bin/env python
import os

import numpy

import splendor.contexts.egl as egl

import ltron.settings as settings
from ltron.bricks.brick_scene import BrickScene
from ltron.geometry.collision import check_collision

egl.initialize_plugin()
egl.initialize_device()

scene = BrickScene(
    renderable=True,
    render_args={
        'load_scene':'grey_cube',
    }
)
scene.import_ldraw(os.path.join(
        settings.collections['omr'], 'ldraw', '8661-1 - Carbon Star.mpd'))

instances = [scene.instances['29'], scene.instances['30']]
#instances[0].transform[2,3] -= 10
#scene.renderer.set_instance_transform(str(instances[0]), instances[0].transform)
snap = instances[0].get_snaps()[1]

collision = check_collision(
        scene,
        instances,
        snap.transform,
        'F',
        resolution=(512,512),
        dump_images='test')

print(collision)
