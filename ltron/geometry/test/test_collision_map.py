#!/usr/bin/env python
import os
import time

#from ltron.settings import collections
from ltron.bricks.brick_scene import BrickScene
from ltron.geometry.collision import build_collision_map

if __name__ == '__main__':
    #carbon_star_path = os.path.join(
    #    collections['omr'], 'ldraw', '8661-1 - Carbon Star.mpd')
    scene = BrickScene(
        renderable=True, track_snaps=True, collision_checker=True)
    scene.import_ldraw('./model2_no_wheels.mpd')
    
    t0 = time.time()
    a = scene.get_assembly()
    t1 = time.time()
    print('assembly_time: %f'%(t1-t0))
    
    t0 = time.time()
    collision_map = build_collision_map(scene)
    t1 = time.time()
    print('t: %f'%(t1-t0))
    
    import pdb
    pdb.set_trace()
