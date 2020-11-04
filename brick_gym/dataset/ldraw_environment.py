import random
import os

import numpy

from gym import spaces

import renderpy.buffer_manager_glut as buffer_manager
import renderpy.core as core
import renderpy.masks as masks

#import brick_gym.dataset.path_list as path_list
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

class LDrawEnvironment:
    def __init__(self,
            viewpoint_control,
            width = 256,
            height = 256):
        
        self.viewpoint_control = viewpoint_control
        self.action_space = self.viewpoint_control.action_space
        self.observation_space = spaces.Box(
                low=0, high=255, shape=(height, width, 3), dtype=numpy.uint8)
        
        '''
        # find all model files
        self.model_directory = os.path.join(directory, split)
        self.model_files = sorted(
                model_file for model_file in os.listdir(self.model_directory)
                if model_file[-4:] == '.mpd')
        if subset is not None:
            self.model_files = self.model_files[:subset]
        '''
        # initialize the loaded_model_path
        #self.reset_mode = reset_mode
        self.loaded_model_path = None
        
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
    
    def load_path(self, model_path, force=False):
        if model_path != self.loaded_model_path or force:
            # convert the mpd file to a renderpy scene
            with open(model_path, 'r') as model_file:
                scene_data = ldraw_renderpy.mpd_to_renderpy(
                        model_file,
                        default_image_light)
            
            # clear and load just the new instances,
            # we don't need to reload meshes
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
            
            bbox = self.renderer.get_instance_center_bbox()
            self.viewpoint_control.set_bbox(bbox)
            
            self.renderer.set_instance_masks_to_mesh_indices(mesh_indices)
            
            self.loaded_model_path = model_path
    
    def get_brick_at_pixel(self, x, y):
        self.manager.enable_frame('mask')
        instance_indices = {
                name:int(name.split('_')[-1])
                for name in self.renderer.list_instances()}
        self.renderer.set_instance_masks_to_instance_indices(instance_indices)
        self.renderer.mask_render()
        mask = self.manager.read_pixels('mask')
        self.renderer.set_instance_masks_to_mesh_indices(mesh_indices)
        
        indices = masks.color_byte_to_index(mask)
        instance_index = indices[y,x]
        if instance_index == 0:
            return None
        else:
            instance_name = 'instance_%i'%instance_index
            return instance_name
    
    def hide_brick_at_pixel(self, x, y):
        instance_name = self.get_brick_at_pixel(x, y)
        if instance_name is not None:
            self.renderer.hide_instance(instance_name)
    
    def reset_state(self):
        self.viewpoint_control.reset()
        '''
        if self.reset_mode == 'random':
            model_id = random.randint(0, len(self.model_files)-1)
        elif self.reset_mode == 'sequential':
            if not self.first_load:
                model_id = 0
            else:
                model_id = (self.loaded_model_id + 1)%len(self.model_files)
        elif isinstance(self.reset_mode, int):
            model_id = self.reset_mode
        
        self.set_state((model_id, None))
        '''
    
    def reset(self):
        self.reset_state()
        return self.observe()
    
    def observe(self, mode='color'):
        self.renderer.set_camera_pose(self.viewpoint_control.observe())
        
        if mode == 'color':
            self.manager.enable_frame('color')
            self.renderer.color_render()
            image = self.manager.read_pixels('color')
        
        elif mode == 'mask':
            self.manager.enable_frame('mask')
            self.renderer.mask_render()
            image = self.manager.read_pixels('mask')
        
        return image
    
    def step(self, action):
        self.viewpoint_control.step(action)
        return self.observe('color'), 0.0, False, {}
    
    def render(self, mode='human', close=False):
        self.manager.show_window()
        self.manager.enable_window()
        self.manager.color_render()
