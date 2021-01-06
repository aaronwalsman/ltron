import renderpy.buffer_manager_glut as buffer_manager_glut
import renderpy.buffer_manager_egl as buffer_manager_egl
from renderpy.frame_buffer import FrameBufferWrapper
import renderpy.core as core
import renderpy.masks as masks
import renderpy.examples as rpy_examples

import brick_gym.gym.spaces as bg_spaces
from brick_gym.gym.components.brick_env_component import BrickEnvComponent
import brick_gym.ldraw.ldraw_renderpy as ldraw_renderpy

default_image_light = rpy_examples.image_lights['grey_cube']

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
            height,
            width,
            frame_key,
            anti_aliasing = True,
            renderer_key = 'renderer'):
        
        self.height = height
        self.width = width
        self.frame_buffer = None
        self.frame_key = frame_key
        self.anti_aliasing = anti_aliasing
        self.renderer_key = renderer_key
    
    def initialize_state(self, state):
        self.frame_buffer = FrameBufferWrapper(
                self.width, self.height, self.anti_aliasing)
        state[self.frame_key] = None
    
    def render(self, state):
        raise NotImplementedError
    
    def reset_state(self, state):
        self.render(state)
    
    def update_state(self, state, action):
        self.render(state)
    
    def compute_observation(self, state, observation):
        observation[self.frame_key] = state[self.frame_key]

class ColorRenderComponent(FrameRenderComponent):
    def __init__(self,
            height,
            width,
            frame_key='color',
            anti_aliasing=True,
            renderer_key='renderer'):
        super(ColorRenderComponent, self).__init__(
                height=height,
                width=width,
                frame_key=frame_key,
                anti_aliasing=anti_aliasing,
                renderer_key=renderer_key)
    
    def update_observation_space(self, observation_space):
        observation_space[self.frame_key] = bg_spaces.ImageSpace(
                self.height, self.width)
    
    def render(self, state):
        renderer = state[self.renderer_key]
        self.frame_buffer.enable()
        renderer.color_render()
        image = self.frame_buffer.read_pixels()
        state[self.frame_key] = image

class MaskRenderComponent(FrameRenderComponent):
    def __init__(self,
            height,
            width,
            frame_key='mask',
            renderer_key='renderer'):
        super(MaskRenderComponent, self).__init__(
                height=height,
                width=width,
                frame_key=frame_key,
                anti_aliasing=False,
                renderer_key=renderer_key)
    
    def update_observation_space(self, observation_space):
        observation_space[self.frame_key] = bg_spaces.SegmentationSpace(
                self.height, self.width)
    
    def render(self, state):
        renderer = state[self.renderer_key]
        self.frame_buffer.enable()
        renderer.mask_render()
        mask = self.frame_buffer.read_pixels()
        segmentation = masks.color_byte_to_index(mask)
        state[self.frame_key] = segmentation
