import time

import numpy

from splendor.frame_buffer import FrameBufferWrapper
import splendor.masks as masks

import ltron.gym.spaces as ltron_spaces
from ltron.gym.components.ltron_gym_component import LtronGymComponent

class ColorRenderComponent(LtronGymComponent):
    def __init__(self,
            width,
            height,
            scene_component,
            anti_alias=True):
        
        self.width = width
        self.height = height
        self.scene_component = scene_component
        scene = self.scene_component.brick_scene
        self.scene_component.brick_scene.make_renderable()
        self.anti_alias = anti_alias
        self.frame_buffer = FrameBufferWrapper(
                self.width, self.height, self.anti_alias)
        
        self.observation_space = ltron_spaces.ImageSpace(
                self.width, self.height)
    
    def observe(self):
        scene = self.scene_component.brick_scene
        self.frame_buffer.enable()
        scene.viewport_scissor(0,0,self.width,self.height)
        scene.color_render()
        self.observation = self.frame_buffer.read_pixels()
    
    def reset(self):
        self.observe()
        return self.observation
    
    def step(self, action):
        self.observe()
        return self.observation, 0., False, None
    
    def set_state(self, state):
        self.observe()
        return self.observation

class SegmentationRenderComponent(LtronGymComponent):
    def __init__(self,
        width,
        height,
        scene_component,
    ):
        
        self.width = width
        self.height = height
        self.scene_component = scene_component
        self.scene_component.brick_scene.make_renderable()
        self.frame_buffer = FrameBufferWrapper(
                self.width, self.height, anti_alias=False)
        
        self.observation_space = ltron_spaces.SegmentationSpace(
                self.width, self.height)
    
    def observe(self):
        scene = self.scene_component.brick_scene
        self.frame_buffer.enable()
        scene.viewport_scissor(0,0,self.width,self.height)
        scene.mask_render()
        mask = self.frame_buffer.read_pixels()
        self.observation = masks.color_byte_to_index(mask)
    
    def reset(self):
        self.observe()
        return self.observation
    
    def step(self, action):
        self.observe()
        return self.observation, 0., False, None
    
    def set_state(self, state):
        self.observe()
        return self.observation

class SnapRenderComponent(LtronGymComponent):
    def __init__(self,
        width,
        height,
        scene_component,
        polarity=None,
        style=None,
    ):
        
        self.width = width
        self.height = height
        self.scene_component = scene_component
        self.polarity=polarity
        self.style=style
        self.scene_component.brick_scene.make_renderable()
        self.frame_buffer = FrameBufferWrapper(
            self.width, self.height, anti_alias=False)
        
        self.observation_space = ltron_spaces.SnapSegmentationSpace(
            self.width, self.height)
        
        self.observation = numpy.zeros(
            (self.height, self.width, 2), dtype=numpy.long)
    
    def observe(self):
        scene = self.scene_component.brick_scene
        self.frame_buffer.enable()
        scene.viewport_scissor(0,0,self.width,self.height)
        
        # get the snap names
        snaps = scene.get_matching_snaps(
            polarity=self.polarity, style=self.style)
        
        # render instance ids
        scene.snap_render_instance_id(snaps)
        instance_id_mask = self.frame_buffer.read_pixels()
        instance_ids = masks.color_byte_to_index(instance_id_mask)
        
        # render snap ids
        scene.snap_render_snap_id(snaps)
        snap_id_mask = self.frame_buffer.read_pixels()
        snap_ids = masks.color_byte_to_index(snap_id_mask)
        
        self.observation = numpy.stack((instance_ids, snap_ids), axis=-1)
    
    def reset(self):
        self.observe()
        return self.observation
    
    def step(self, action):
        self.observe()
        return self.observation, 0., False, None
    
    def set_state(self, state):
        self.observe()
        return self.observation
