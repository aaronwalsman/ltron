import random
import os

import numpy

from gym import spaces

import renderpy.buffer_manager_glut as buffer_manager
import renderpy.core as core

import brick_gym.ldraw.ldraw_renderpy as ldraw_renderpy

default_image_light = '/home/awalsman/Development/renderpy/renderpy/            example_image_lights/grey_cube'

class MPDSequence:
    def __init__(self,
            directory,
            split,
            subset = None,
            model_selection_mode = 'random',
            width = 256,
            height = 256):
        
        self.observation_space = spaces.Box(
                low=0, high=255, shape=(height, width, 3), dtype=numpy.uint8)
        
        # find all model files
        self.model_directory = os.path.join(directory, split)
        self.model_files = sorted(
                model_file for model_file in os.listdir(self.model_directory)
                if model_file[-4:] == '.mpd')
        if subset is not None:
            self.model_files = self.model_files[:subset]
        
        # initialize the model_id
        self.model_selection_mode = model_selection_mode
        if self.model_selection_mode == 'random':
            self.model_id = random.randint(0, len(self.model_files)-1)
        elif model_selection_mode == 'sequential':
            self.model_id = 0
        elif isinstance(model_selection_mode, int):
            self.model_id = model_selection_mode
        
        # initialize renderpy
        self.manager = buffer_manager.initialize_shared_buffer_manager(
                width, height)
        self.manager.add_frame(
                'color', width=width, height=height, anti_aliasing=True)
        self.manager.add_frame(
                'mask', width=width, height=height, anti_aliasing=True)
        self.renderer = core.Renderpy()
        self.first_load = False
    
    def load_model_file(self):
        # convert the mpd file to a renderpy scene
        model_path = self.model_files[self.model_id]
        with open(model_path, 'r') as model_file:
            scene_data = ldraw_renderpy.mpd_to_renderpy(
                    model_file,
                    default_image_light)
        
        # clear and load just the new instances, we don't need to reload meshes
        # this is awkward, renderpy should add a function to do this
        if not self.first_load:
            renderer.load_scene(scene_data, clear_existing=True)
        else:
            renderer.clear_instances()
            for mesh, mesh_data in scene_data['meshes'].items():
                if not renderer.mesh_exists(mesh):
                    renderer.load_mesh(mesh, **mesh_data)
            renderer.load_scene(
                    {'instances' : scene_data['instances'],
                     'camera' : scene_data['camera']},
                    clear_existing = False)
    
    def increment_scene(self):
        if model_selection_mode == 'random':
            self.model_id = random.randint(0, len(self.model_files-1))
            self.load_model_file()
        
        elif model_selection_mode == 'sequential':
            self.model_id = (self.model_id + 1)%len(self.model_files)
            self.load_model_file()
    
    def set_camera_pose(self, camera_pose):
        self.renderer.set_camera_pose(camera_pose)
    
    def observe(self):
        self.manager.enable_frame('color')
        self.renderer.color_render()
        color_image = self.manager.read_pixels('color')
        
        self.manager.enable_frame('mask')
        self.renderer.mask_render()
        mask_image = self.manager.read_pixels('mask')
        
        return color_image, mask_image
    
    def render(self, mode='human', close=False):
        # TODO : Make this happen
        pass
