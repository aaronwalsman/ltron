import random
import os
import json
import multiprocessing

import numpy

from gym import spaces

import renderpy.buffer_manager_egl as buffer_manager
import renderpy.core as core
import renderpy.masks as masks
import renderpy.examples as renderpy_examples

import ltron.spaces as bg_spaces
import ltron.ldraw.ldraw_renderpy as ldraw_renderpy

default_image_light = renderpy_examples.image_lights['grey_cube']

# Deprecated.  Use BrickEnv instead.
class LDrawEnvironment:
    def __init__(self,
            model_library,
            action_mode = ('',),
            observation_mode = ('color',),
            max_edges_per_scene=None,
            height = 256,
            width = 256,
            scene_path = None):
        
        # store local information
        self.observation_mode = observation_mode
        self.action_mode = action_mode
        
        # build observation space
        observation_space = {}
        for mode in observation_mode:
            if mode == 'color':
                color_space = bg_spaces.ImageSpace(height, width)
                observation_space['color'] = color_space
            elif mode == 'segmentation':
                segmentation_space = bg_spaces.SegmentationSpace(
                        height, width)
                observation_space['segmentation'] = segmentation_space
            elif mode == 'graph':
                observation_space['graph'] = bg_spaces.GraphSpace(
                        model_library.num_classes,
                        model_library.max_bricks_per_scene)
            elif mode == 'sparse_graph':
                observation_space['sparse_graph'] = bg_spaces.SparseGraphSpace(
                        model_library.num_classes,
                        model_library.max_bricks_per_scene,
                        model_library.max_edges_per_scene)
        self.observation_space = spaces.Dict(observation_space)
        
        # build action space
        self.action_space = spaces.Tuple((
                self.viewpoint_control.action_space,
                self.visibility_control.action_space))
        
        action_space = {}
        for mode in action_mode:
            if mode == 'hide':
        
        
        # initialize renderpy
        self.manager = buffer_manager.initialize_shared_buffer_manager()
        try:
            self.manager.add_frame(
                    'color', width=width, height=height, anti_aliasing=True)
        except buffer_manager.FrameExistsError:
            pass
        try:
            self.manager.add_frame(
                    'mask', width=width, height=height, anti_aliasing=False)
        except buffer_manager.FrameExistsError:
            pass
        self.renderer = core.Renderpy()
        self.edges = []
        
        # load initial path
        self.loaded_scene_path = None
        if scene_path is not None:
            self.load_path(scene_path)
    
    def load_path(self, scene_path, force=False):
        if scene_path != self.loaded_scene_path or force:
            # convert the mpd file to a renderpy scene
            with open(scene_path, 'r') as scene_file:
                scene_data = ldraw_renderpy.mpd_to_renderpy(
                        scene_file,
                        default_image_light)
            
            # clear and load just the new instances
            # we don't need to reload meshes
            # this is awkward, renderpy should add a function to do this
            if self.loaded_scene_path is None or force:
                self.renderer.load_scene(scene_data, clear_existing=True)
            else:
                self.renderer.clear_instances()
                for mesh, mesh_data in scene_data['meshes'].items():
                    if not self.renderer.mesh_exists(mesh):
                        self.renderer.load_mesh(mesh, **mesh_data)
                self.renderer.load_scene(
                        {'instances' : scene_data['instances'],
                         'camera' : scene_data['camera']},
                        clear_existing = False)
            
            bbox = self.renderer.get_instance_center_bbox()
            self.viewpoint_control.set_bbox(bbox)
            self.visibility_control.reset()
            
            self.loaded_scene_path = scene_path
            
        for instance_name in self.renderer.list_instances():
            self.renderer.show_instance(instance_name)
        self.hidden_indices = set()
    
    '''
    def get_brick_at_pixel(self, x, y):
        self.manager.enable_frame('mask')
        self.renderer.mask_render()
        mask = self.manager.read_pixels('mask')
        indices = masks.color_byte_to_index(mask)
        brick_index = indices[y,x]
        return brick_index
    '''
    
    '''
    def hide_brick(self, brick_index):
        if brick_index != 0:
            brick_name = 'instance_%i'%brick_index
            try:
                self.renderer.hide_instance(brick_name)
                self.hidden_indices.add(brick_index)
                return brick_name
            except KeyError:
                return None
        
        return None
    '''
    
    '''
    def hide_brick_at_pixel(self, x, y):
        brick_index = self.get_brick_at_pixel(x, y)
        instance_name = self.hide_brick(brick_index)
        
        return instance_name
    '''
    
    def get_instance_brick_types(self):
        instance_types = {}
        for instance_name in self.renderer.list_instances():
            instance_id = int(instance_name.split('_')[-1])
            instance_types[instance_id] = self.renderer.get_instance_mesh_name(
                    instance_name)
        return instance_types
    
    def reset_state(self):
        self.viewpoint_control.reset()
        self.hide_control.reset()
    
    def reset(self):
        self.reset_state()
        return self.observe()
    
    def observe(self, mode='color'):
        self.renderer.set_camera_pose(self.viewpoint_control.observe())
        
        if mode == 'color':
            self.manager.enable_frame('color')
            self.renderer.color_render()
            image = self.manager.read_pixels('color')
        elif mode == 'instance_labels':
            self.manager.enable_frame('mask')
            self.renderer.mask_render()
            image = self.manager.read_pixels('mask')
            image = masks.color_byte_to_index(image)
        
        return image
    
    def step(self, action):
        self.viewpoint_control.step(action)
        return self.observe('color'), 0.0, False, {}
    
    def render(self, mode='human', close=False):
        self.manager.show_window()
        self.manager.enable_window()
        self.manager.color_render()

# DEPRECATED, USE MULTICLASS INTEAD
'''
def ldraw_process(
        connection,
        width, height,
        viewpoint_control):
    
    environment = LDrawEnvironment(
            viewpoint_control,
            width=width,
            height=height)
    
    while True:
        instruction, args = connection.recv()
        if instruction == 'load_path':
            #print('Loading scene: %s'%args)
            environment.load_path(args)
        
        elif instruction == 'hide_brick':
            #print('Hiding brick: %i'%args)
            environment.hide_brick(args)
        
        elif instruction == 'observe':
            #print('Observing: ' + ' '.join(args))
            observations = []
            for mode in args:
                observation = environment.observe(mode)
                observations.append(observation)
            connection.send(observations)
        
        elif instruction == 'get_instance_brick_types':
            #print('Getting instance brick types')
            connection.send(environment.get_instance_brick_types())
        
        elif instruction == 'edges':
            connection.send(environment.edges)
        
        elif instruction == 'shutdown':
            #print('Shutting down')
            break

# DEPRECATE THIS, USE MULTICLASS instead
class MultiLDrawEnvironment:
    def __init__(self,
            num_processes,
            width, height,
            viewpoint_control):
        
        self.num_processes = num_processes
        self.width = width
        self.height = height
        self.viewpoint_control = viewpoint_control
        self.connections = []
        self.processes = []
    
    def start_processes(self):
        for i in range(self.num_processes):
            parent_connection, child_connection = multiprocessing.Pipe()
            self.connections.append(parent_connection)
            process_args = (
                    child_connection,
                    self.width,
                    self.height,
                    self.viewpoint_control)
            process = multiprocessing.Process(
                    target = ldraw_process, args = process_args)
            process.start()
            self.processes.append(process)
    
    def shutdown_processes(self):
        for connection in self.connections:
            connection.send(('shutdown',None))
        
        for process in self.processes:
            process.join()
        
        self.connections = []
        self.processes = []
    
    def load_paths(self, paths):
        assert(len(paths) <= self.num_processes)
        for path, connection in zip(paths, self.connections):
            connection.send(('load_path', path))
    
    def observe(self, modes):
        for connection in self.connections:
            connection.send(('observe', modes))
        observations = []
        for connection in self.connections:
            observation = connection.recv()
            observations.append(observation)
        
        return observations
    
    def get_instance_brick_types(self):
        for connection in self.connections:
            connection.send(('get_instance_brick_types', None))
        instance_brick_types = []
        for connection in self.connections:
            instance_brick_types.append(connection.recv())
        
        return instance_brick_types
    
    def get_edges(self):
        for connection in self.connections:
            connection.send(('edges', None))
        edges = []
        for connection in self.connections:
            edges.append(connection.recv())
        
        return edges
    
    def hide_bricks(self, bricks):
        assert(len(bricks) <= self.num_processes)
        for brick, connection in zip(bricks, self.connections):
            connection.send(('hide_brick', brick))
    
    def __del__(self):
        self.shutdown_processes()
    
    def __enter__(self):
        self.start_processes()
    
    def __exit__(self, type, value, traceback):
        self.shutdown_processes()
'''
