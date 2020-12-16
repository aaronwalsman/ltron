#!/usr/bin/env python
import math
import os
import multiprocessing

import renderpy.buffer_manager_glut as buffer_manager
import renderpy.core as core

import brick_gym.config as config
import brick_gym.dataset.ldraw_environment as ldraw_environment
import brick_gym.viewpoint as viewpoint

def env_process(
        connection,
        rank,
        width, height,
        viewpoint_class_name, viewpoint_args):
    
    viewpoint_class = getattr(viewpoint, viewpoint_class_name)
    viewpoint_control = viewpoint_class(**viewpoint_args)
    environment = ldraw_environment.LDrawEnvironment(
            viewpoint_control,
            width=width,
            height=height)
    
    while True:
        instruction, args = connection.recv()
        if instruction == 'load_scene':
            print('You just told me to load scene: %s'%args)
            environment.load_path(args)
        
        if instruction == 'hide_id':
            print('You just told me to hide id: %i'%args)
        
        if instruction == 'observe':
            print('You just told me to observe:')
            print(args)
            observations = []
            for mode in args:
                observation = environment.observe(mode)
                observations.append(observation)
            connection.send(observations)
        
        if instruction == 'shutdown':
            print('You just told me to shutdown')
            break

if __name__ == '__main__':
    num_processes = 4
    parent_connections = []
    processes = []
    for i in range(num_processes):
        parent_connection, child_connection = multiprocessing.Pipe()
        parent_connections.append(parent_connection)
        process = multiprocessing.Process(target=env_process, args=(
                child_connection, i, 256, 256,
                'FixedAzimuthalViewpoint',
                {'azimuth':math.radians(30), 'elevation':-math.radians(45)}))
        process.start()
        processes.append(process)
    
    for i in range(num_processes):
        parent_connections[i].send(
                ('load_scene',
                os.path.join(config.paths['omr'], '8661-1 - Carbon Star.mpd')))
    
    for i in range(num_processes):
        parent_connections[i].send(
                ('observe', ('color', 'instances')))
    
    for i in range(num_processes):
        color, instance_mask = parent_connections[i].recv()
        print(color.shape)
        print(instance_mask.shape)
    
    for i in range(num_processes):
        parent_connections[i].send(('shutdown', None))
        processes[i].join()
