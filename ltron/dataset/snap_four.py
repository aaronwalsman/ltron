#!/usr/bin/env python
import random
import math
import os
import json

import numpy

import tqdm

import ltron.settings as settings
from ltron.bricks.brick_scene import BrickScene

random.seed(1234)

num_scenes = 2000
scene_size = (10,10)

ldraw_path = os.path.join(settings.collections['snap_four'], 'ldraw')
if not os.path.exists(ldraw_path):
    os.makedirs(ldraw_path)

for i in tqdm.tqdm(range(num_scenes)):
    x = random.randint(0, scene_size[0]-1) * 20
    z = random.randint(0, scene_size[1]-1) * 20
    
    scene = BrickScene()
    
    theta = random.uniform(0, math.pi*2)
    c = math.cos(theta)
    s = math.sin(theta)
    wing_pose = numpy.array([
        [ c, 0,-s, x],
        [ 0, 1, 0, 0],
        [ s, 0, c, z],
        [ 0, 0, 0, 1]]) @ BrickScene.upright
    scene.add_instance('41770.dat', brick_color=4, transform=wing_pose)
    
    for j in range(4):
        x = random.randint(0, scene_size[0]-1) * 20
        z = random.randint(0, scene_size[1]-1) * 20
        dot_pose = numpy.eye(4)
        dot_pose[0,3] = x
        dot_pose[1,3] = 8*j
        dot_pose[2,3] = z
        dot_pose = dot_pose @ BrickScene.upright
        
        color = [15, 14, 25, 1][j]
        scene.add_instance('6141.dat', brick_color=color, transform=dot_pose)
    
    scene_path = os.path.join(ldraw_path, 'snap_four_%06i.mpd'%i)
    scene.export_ldraw(scene_path)

json_data = {
    'splits':{
        'all':[
            '{snap_four}/ldraw/snap_four*.mpd',
        ],
        'train':[
            '{snap_four}/ldraw/snap_four_0000*.mpd',
            '{snap_four}/ldraw/snap_four_0001*.mpd',
            '{snap_four}/ldraw/snap_four_0002*.mpd',
            '{snap_four}/ldraw/snap_four_0003*.mpd',
            '{snap_four}/ldraw/snap_four_0004*.mpd',
            '{snap_four}/ldraw/snap_four_0005*.mpd',
            '{snap_four}/ldraw/snap_four_0006*.mpd',
            '{snap_four}/ldraw/snap_four_0007*.mpd',
            '{snap_four}/ldraw/snap_four_0008*.mpd',
            '{snap_four}/ldraw/snap_four_0009*.mpd',
            '{snap_four}/ldraw/snap_four_0010*.mpd',
            '{snap_four}/ldraw/snap_four_0011*.mpd',
            '{snap_four}/ldraw/snap_four_0012*.mpd',
            '{snap_four}/ldraw/snap_four_0013*.mpd',
            '{snap_four}/ldraw/snap_four_0014*.mpd',
        ],
        'test':[
            '{snap_four}/ldraw/snap_four_0015*.mpd',
            '{snap_four}/ldraw/snap_four_0016*.mpd',
            '{snap_four}/ldraw/snap_four_0017*.mpd',
            '{snap_four}/ldraw/snap_four_0018*.mpd',
            '{snap_four}/ldraw/snap_four_0019*.mpd',
        ],
    },
    'max_instances_per_scene':2,
    'max_edges_per_scene':1,
    'shape_ids':{
        '41770.dat':1,
        '6141.dat':2,
    }
}

json_path = os.path.join(settings.collections['snap_four'], 'snap_four.json')
with open(json_path, 'w') as f:
    json.dump(json_data, f)
