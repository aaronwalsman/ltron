#!/usr/bin/env python
import os
import time

from ltron.settings import collections
from ltron.bricks.brick_scene import BrickScene
from ltron.geometry.collision import build_collision_map

if __name__ == '__main__':
    carbon_star_path = os.path.join(
        collections['omr'], 'ldraw', '8661-1 - Carbon Star.mpd')
    scene = BrickScene(renderable=True, track_snaps=True)
    scene.import_ldraw(carbon_star_path)
    
    t0 = time.time()
    collision_map = build_collision_map(scene)
    t1 = time.time()
    print('t: %f'%(t1-t0))
    
    import pdb
    pdb.set_trace()
