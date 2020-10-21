import random
import os

import numpy

from gym import spaces

import renderpy.buffer_manager_glut as buffer_manager
import renderpy.core as core

import brick_gym.ldraw.ldraw_renderpy as ldraw_renderpy
import brick_gym.random_stack.dataset as random_stack_dataset

default_image_light = '/home/awalsman/Development/renderpy/renderpy/example_image_lights/grey_cube'

mesh_indices = {
    '3005' : 1,
    '3004' : 2,
    '3003' : 3,
    '3002' : 4,
    '3001' : 5,
    '2456' : 6}

class MPDSequence:
    def __init__(self,
            directory,
            split,
            subset = None,
            model_selection_mode = 'random',
            width = 256,
            height = 256):
        
        self.scene_center = (0,0,0)
        self.scene_distance = 0
        
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
        elif self.model_selection_mode == 'sequential':
            self.model_id = 0
        elif isinstance(self.model_selection_mode, int):
            self.model_id = self.model_selection_mode
        
        # initialize renderpy
        self.manager = buffer_manager.initialize_shared_buffer_manager(
                width, height)
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
        self.first_load = False
    
    def load_model_file(self):
        # convert the mpd file to a renderpy scene
        model_path = os.path.join(
                self.model_directory, self.model_files[self.model_id])
        with open(model_path, 'r') as model_file:
            scene_data = ldraw_renderpy.mpd_to_renderpy(
                    model_file,
                    default_image_light)
        
        # clear and load just the new instances, we don't need to reload meshes
        # this is awkward, renderpy should add a function to do this
        if not self.first_load:
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
        
        bbox_min, bbox_max = self.renderer.get_instance_center_bbox()
        bbox_range = bbox_max - bbox_min
        self.scene_center = bbox_min + bbox_range * 0.5
        self.scene_distance = numpy.max(bbox_range) * 3
        
        self.renderer.set_instance_masks_to_mesh_indices(mesh_indices)
    
    def increment_scene(self):
        if self.model_selection_mode == 'random':
            self.model_id = random.randint(0, len(self.model_files)-1)
            self.load_model_file()
        
        elif self.model_selection_mode == 'sequential':
            self.model_id = (self.model_id + 1)%len(self.model_files)
            self.load_model_file()
    
    def set_camera_pose(self, camera_pose):
        self.renderer.set_camera_pose(camera_pose)
    
    def observe(self):
        self.manager.enable_frame('color')
        self.renderer.color_render()
        color_image = self.manager.read_pixels('color')
        self.recent_color = color_image
        
        self.manager.enable_frame('mask')
        self.renderer.mask_render()
        mask_image = self.manager.read_pixels('mask')
        self.recent_mask = mask_image
        
        return color_image
    
    def render(self, mode='human', close=False):
        # TODO : Make this happen
        pass
