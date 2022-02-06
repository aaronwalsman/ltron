#!/usr/bin/env python
import random
import math
import os
import json

import numpy

import tqdm

from splendor.frame_buffer import FrameBufferWrapper

import ltron.settings as settings
from ltron.bricks.brick_scene import BrickScene
from ltron.geometry.collision import check_collision

random.seed(1234)

num_scenes = 2000
scene_size = (10,10)

ldraw_path = os.path.join(settings.collections['conditional_snap_two'], 'ldraw')
if not os.path.exists(ldraw_path):
    os.makedirs(ldraw_path)

dummy_scene = BrickScene(renderable=True)    
frame_buffer = FrameBufferWrapper(64, 64, anti_alias=False)

for i in tqdm.tqdm(range(num_scenes)):
    x = random.randint(0, scene_size[0]-1) * 20
    z = random.randint(0, scene_size[1]-1) * 20
    
    scene = BrickScene(renderable=True)
    
    theta = random.uniform(0, math.pi*2)
    c = math.cos(theta)
    s = math.sin(theta)
    wing_pose = numpy.array([
        [ c, 0,-s, x],
        [ 0, 1, 0, 0],
        [ s, 0, c, z],
        [ 0, 0, 0, 1]]) @ BrickScene.upright
    wedge_instance = scene.add_instance('41770.dat', 4, wing_pose)
    
    x1 = random.randint(0, scene_size[0]-1) * 20
    z1 = random.randint(0, scene_size[1]-1) * 20
    
    x2 = random.randint(0, scene_size[0]-1) * 20
    z2 = random.randint(0, scene_size[1]-1) * 20
    while abs(x1-x2) < 40 and abs(z1-z2) < 40:
        x2 = random.randint(0, scene_size[0]-1) * 20
        z2 = random.randint(0, scene_size[1]-1) * 20
    
    theta = random.uniform(0, math.pi*2)
    c = math.cos(theta)
    s = math.sin(theta)
    slope_pose = numpy.array([
        [ c, 0,-s, x1],
        [ 0, 1, 0,  24],
        [ s, 0, c, z1],
        [ 0, 0, 0,  1]]) @ BrickScene.upright
    slope_instance_a = scene.add_instance('3040.dat', 15, slope_pose)
    
    theta = random.uniform(0, math.pi*2)
    c = math.cos(theta)
    s = math.sin(theta)
    slope_pose = numpy.array([
        [ c, 0,-s, x2],
        [ 0, 1, 0,  24],
        [ s, 0, c, z2],
        [ 0, 0, 0,  1]]) @ BrickScene.upright
    slope_instance_b = scene.add_instance('3040.dat', 14, slope_pose)
    
    #scene_path = os.path.join(ldraw_path, 'conditional_snap_two_%06i.mpd'%i)
    scene_path = os.path.join(ldraw_path, 'x_%06i.mpd'%i)
    scene.export_ldraw(scene_path)
    
    top_wedge_snaps = 5,6,7,8
    bottom_slope_snaps = 1,2
    thetas = [i*math.pi/2. for i in range(4)]
    
    # first snap
    wedge_snap = random.choice(top_wedge_snaps)
    slope_snap = random.choice(bottom_slope_snaps)
    theta = random.choice(thetas)
    slope_snap_transform = slope_instance_a.snaps[slope_snap].transform
    wedge_snap_transform = wedge_instance.snaps[wedge_snap].transform
    c = math.cos(theta)
    s = math.sin(theta)
    rotation = numpy.array([
        [ c, 0,-s, 0],
        [ 0, 1, 0, 0],
        [ s, 0, c, 0],
        [ 0, 0, 0, 1]])
    
    first_transform = (
        wedge_snap_transform @
        rotation @
        numpy.linalg.inv(slope_snap_transform) @
        slope_instance_a.transform)
    scene.move_instance(slope_instance_a, first_transform)
    
    # second snap
    while True:
        wedge_snap = random.choice(top_wedge_snaps)
        slope_snap = random.choice(bottom_slope_snaps)
        theta = random.choice(thetas)
        slope_snap_transform = slope_instance_b.snaps[slope_snap].transform
        wedge_snap_transform = wedge_instance.snaps[wedge_snap].transform
        c = math.cos(theta)
        s = math.sin(theta)
        rotation = numpy.array([
            [ c, 0,-s, 0],
            [ 0, 1, 0, 0],
            [ s, 0, c, 0],
            [ 0, 0, 0, 1]])
        
        second_transform = (
            wedge_snap_transform @
            rotation @
            numpy.linalg.inv(slope_snap_transform) @
            slope_instance_b.transform)
        
        scene.move_instance(slope_instance_b, second_transform)
        
        if not check_collision(
            scene,
            [slope_instance_b],
            slope_snap_transform,
            '-',
            frame_buffer=frame_buffer,
        ):
            break
    
    scene_path = os.path.join(ldraw_path, 'y_%06i.mpd'%i)
    scene.export_ldraw(scene_path)

json_data = {
    'splits':{
        'all_x':[
            '{conditional_snap_two}/ldraw/x*.mpd',
        ],
        'train_x':[
            '{conditional_snap_two}/ldraw/x_0000*.mpd',
            '{conditional_snap_two}/ldraw/x_0001*.mpd',
            '{conditional_snap_two}/ldraw/x_0002*.mpd',
            '{conditional_snap_two}/ldraw/x_0003*.mpd',
            '{conditional_snap_two}/ldraw/x_0004*.mpd',
            '{conditional_snap_two}/ldraw/x_0005*.mpd',
            '{conditional_snap_two}/ldraw/x_0006*.mpd',
            '{conditional_snap_two}/ldraw/x_0007*.mpd',
            '{conditional_snap_two}/ldraw/x_0008*.mpd',
            '{conditional_snap_two}/ldraw/x_0009*.mpd',
            '{conditional_snap_two}/ldraw/x_0010*.mpd',
            '{conditional_snap_two}/ldraw/x_0011*.mpd',
            '{conditional_snap_two}/ldraw/x_0012*.mpd',
            '{conditional_snap_two}/ldraw/x_0013*.mpd',
            '{conditional_snap_two}/ldraw/x_0014*.mpd',
        ],
        'test_x':[
            '{conditional_snap_two}/ldraw/x_0015*.mpd',
            '{conditional_snap_two}/ldraw/x_0016*.mpd',
            '{conditional_snap_two}/ldraw/x_0017*.mpd',
            '{conditional_snap_two}/ldraw/x_0018*.mpd',
            '{conditional_snap_two}/ldraw/x_0019*.mpd',
        ],
        'all_y':[
            '{conditional_snap_two}/ldraw/y*.mpd',
        ],
        'train_y':[
            '{conditional_snap_two}/ldraw/y_0000*.mpd',
            '{conditional_snap_two}/ldraw/y_0001*.mpd',
            '{conditional_snap_two}/ldraw/y_0002*.mpd',
            '{conditional_snap_two}/ldraw/y_0003*.mpd',
            '{conditional_snap_two}/ldraw/y_0004*.mpd',
            '{conditional_snap_two}/ldraw/y_0005*.mpd',
            '{conditional_snap_two}/ldraw/y_0006*.mpd',
            '{conditional_snap_two}/ldraw/y_0007*.mpd',
            '{conditional_snap_two}/ldraw/y_0008*.mpd',
            '{conditional_snap_two}/ldraw/y_0009*.mpd',
            '{conditional_snap_two}/ldraw/y_0010*.mpd',
            '{conditional_snap_two}/ldraw/y_0011*.mpd',
            '{conditional_snap_two}/ldraw/y_0012*.mpd',
            '{conditional_snap_two}/ldraw/y_0013*.mpd',
            '{conditional_snap_two}/ldraw/y_0014*.mpd',
        ],
        'test_y':[
            '{conditional_snap_two}/ldraw/y_0015*.mpd',
            '{conditional_snap_two}/ldraw/y_0016*.mpd',
            '{conditional_snap_two}/ldraw/y_0017*.mpd',
            '{conditional_snap_two}/ldraw/y_0018*.mpd',
            '{conditional_snap_two}/ldraw/y_0019*.mpd',
        ],
    },
    'max_instances_per_scene':3,
    'max_edges_per_scene':3,
    'shape_ids':{
        '41770.dat':1,
        '3040.dat':2,
    }
}

json_path = os.path.join(
    settings.collections['conditional_snap_two'], 'conditional_snap_two.json')
with open(json_path, 'w') as f:
    json.dump(json_data, f)
