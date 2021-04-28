import numpy

from renderpy.frame_buffer import FrameBufferWrapper
import renderpy.masks as masks

import ltron.gym.spaces as bg_spaces
from ltron.gym.components.brick_env_component import BrickEnvComponent
'''
class RendererComponent(BrickEnvComponent):
    def __init__(self,
            buffer_manager_mode='egl',
            load_path=None,
            scene_path_key='scene_path',
            renderer_key='renderer'):
        
        # store keys
        self.scene_path_key = scene_path_key
        self.renderer_key = renderer_key
        
        # initialize manager
        if buffer_manager_mode == 'egl':
            buffer_manager = buffer_manager_egl
        elif buffer_manager_mode == 'glut':
            buffer_manager = buffer_manager_glut
        self.manager = buffer_manager.initialize_shared_buffer_manager()
        
        # initialize renderer
        self.renderer = core.Renderpy()

        # load initial path
        self.loaded_path = None
        if load_path is not None:
            self.load_path(load_path)
        
    def initialize_state(self, state):
        state[self.renderer_key] = self.renderer
    
    def load_path(self, scene_path, force=False):
        if scene_path != self.loaded_path or force:
            # convert the mpd file to a renderpy scene
            with open(scene_path, 'r') as scene_file:
                scene_data = ldraw_renderpy.mpd_to_renderpy(
                        scene_file,
                        default_image_light)
            
            self.renderer.clear_instances()
            self.renderer.load_scene(
                    scene_data, clear_scene=False, reload_assets=False)
            
            self.loaded_path = scene_path
    
    def reset_state(self, state):
        if self.scene_path_key is not None:
            scene_path = state[self.scene_path_key]
            self.load_path(scene_path)

class FrameRenderComponent(BrickEnvComponent):
    def __init__(self,
            width,
            height,
            scene_component = 'brick_scene',
            anti_alias = True):
        
        self.height = height
        self.width = width
        self.anti_alias = anti_alias
        self.frame_buffer = FrameBufferWrapper(
                self.width, self.height, self.anti_aliasing)
        self.scene_component = scene_component
    
    def compute_observation(self):
        raise NotImplementedError
    
    def reset(self):
        self.image = self.compute_observation()
        return self.image
    
    def step(self, action):
        self.image = self.compute_observation()
        return self.image, 0., False, None
'''
class ColorRenderComponent(BrickEnvComponent):
    def __init__(self,
            width,
            height,
            scene_component,
            anti_alias=True):
        
        self.width = width
        self.height = height
        self.scene_component = scene_component
        self.scene_component.brick_scene.make_renderable()
        self.anti_alias = anti_alias
        self.frame_buffer = FrameBufferWrapper(
                self.width, self.height, self.anti_alias)
        
        self.observation_space = bg_spaces.ImageSpace(
                self.width, self.height)
    
    def compute_observation(self):
        scene = self.scene_component.brick_scene
        self.frame_buffer.enable()
        scene.color_render()
        self.observation = self.frame_buffer.read_pixels()
    
    def reset(self):
        self.compute_observation()
        return self.observation
    
    def step(self, action):
        self.compute_observation()
        return self.observation, 0., False, None
    
    def set_state(self, state):
        self.compute_observation()

class SegmentationRenderComponent(BrickEnvComponent):
    def __init__(self,
            height,
            width,
            scene_component,
            terminate_on_empty=True):
        
        self.width = width
        self.height = height
        self.scene_component = scene_component
        self.terminate_on_empty = terminate_on_empty
        self.scene_component.brick_scene.make_renderable()
        self.frame_buffer = FrameBufferWrapper(
                self.width, self.height, anti_alias=False)
        
        self.observation_space = bg_spaces.SegmentationSpace(
                self.height, self.width)
    
    def compute_observation(self):
        scene = self.scene_component.brick_scene
        self.frame_buffer.enable()
        scene.mask_render()
        mask = self.frame_buffer.read_pixels()
        self.observation = masks.color_byte_to_index(mask)
    
    def reset(self):
        self.compute_observation()
        return self.observation
    
    def step(self, action):
        self.compute_observation()
        terminal = False
        if self.terminate_on_empty:
            terminal = numpy.all(self.observation == 0)
        return self.observation, 0., terminal, None
    
    def set_state(self, state):
        self.compute_observation()
