#!/usr/bin/env python
import os

import numpy

import splendor.contexts.egl as egl

import ltron.settings as settings
from ltron.bricks.brick_scene import BrickScene
from ltron.geometry.collision import check_collision, check_snap_collision

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

collision = check_snap_collision(
        scene,
        instances,
        snap,
        mode='attach',
        resolution=(64,64),
        dump_images='test')

print(collision)

collision = check_snap_collision(
        scene,
        instances,
        snap,
        mode='detach',
        resolution=(64,64),
        dump_images='test')

print(collision)

instance_transform = instances[0].transform
instance_transform[1,3] -= 8
scene.move_instance('29', instance_transform)
snap = instances[0].get_snaps()[1]

collision = check_snap_collision(
        scene,
        instances,
        snap,
        mode='attach',
        resolution=(64,64),
        dump_images='test')

print(collision)



