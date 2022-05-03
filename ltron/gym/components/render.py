import numpy

from splendor.frame_buffer import FrameBufferWrapper
import splendor.masks as masks

from ltron.gym.spaces import ImageSpace, IndexMaskSpace, SnapSegmentationSpace
from ltron.gym.components.ltron_gym_component import LtronGymComponent

class RenderComponent(LtronGymComponent):
    def __init__(self, width, height, render_frequency='step', observable=True):
        self.width = width
        self.height = height
        assert render_frequency in ('reset', 'step', 'on_demand')
        self.render_frequency = render_frequency
        self.observable=observable
        self.stale = True
    
    def observe(self):
        if self.stale:
            self.render_image()
            self.stale = False
        return self.observation
    
    def reset(self):
        self.stale = True
        if self.render_frequency in ('step', 'reset'):
            self.observe()
        if self.observable:
            return self.observation
        else:
            return None
    
    def step(self, action):
        self.stale = True
        if self.render_frequency in ('step',):
            self.observe()
        if self.observable:
            return self.observation, 0., False, None
        else:
            return None, 0., False, None
    
    def get_state(self):
        return (self.observation, self.stale)
    
    def set_state(self, state):
        self.observation, self.stale = state
        return self.observation

class ColorRenderComponent(RenderComponent):
    def __init__(self,
        width,
        height,
        scene_component,
        render_frequency='step',
        anti_alias=True,
        observable=True,
    ):
        
        super().__init__(
            width,
            height,
            render_frequency=render_frequency,
            observable=observable
        )
        
        self.scene_component = scene_component
        self.render_frequency = render_frequency
        
        scene = self.scene_component.brick_scene
        self.scene_component.brick_scene.make_renderable()
        self.anti_alias = anti_alias
        self.frame_buffer = FrameBufferWrapper(
                self.width, self.height, self.anti_alias)
        
        if observable:
            self.observation_space = ImageSpace(
                    self.width, self.height)
    
    def render_image(self):
        scene = self.scene_component.brick_scene
        self.frame_buffer.enable()
        scene.viewport_scissor(0,0,self.width,self.height)
        scene.color_render()
        self.observation = self.frame_buffer.read_pixels()

class InstanceRenderComponent(RenderComponent):
    def __init__(self,
        width,
        height,
        scene_component,
        render_frequency='step',
        observable=True,
    ):
        
        super().__init__(
            width,
            height,
            render_frequency=render_frequency,
            observable=observable,
        )
        
        self.width = width
        self.height = height
        self.scene_component = scene_component
        self.scene_component.brick_scene.make_renderable()
        self.frame_buffer = FrameBufferWrapper(
                self.width, self.height, anti_alias=False)
        
        if observable:
            self.observation_space = IndexMaskSpace(
                    self.width, self.height)
    
    def render_image(self):
        scene = self.scene_component.brick_scene
        self.frame_buffer.enable()
        # it seems like this should be done at a lower level than this
        scene.viewport_scissor(0,0,self.width,self.height)
        scene.mask_render()
        mask = self.frame_buffer.read_pixels()
        self.observation = masks.color_byte_to_index(mask)

class SnapRenderComponent(RenderComponent):
    def __init__(self,
        width,
        height,
        scene_component,
        render_frequency='step',
        polarity=None,
        style=None,
        observable=True,
    ):
        
        super().__init__(
            width,
            height,
            render_frequency=render_frequency,
            observable=observable,
        )
        
        self.scene_component = scene_component
        self.polarity=polarity
        self.style=style
        self.scene_component.brick_scene.make_renderable()
        self.frame_buffer = FrameBufferWrapper(
            self.width, self.height, anti_alias=False)
        
        if observable:
            self.observation_space = SnapSegmentationSpace(
                self.width, self.height)
        
        self.observation = numpy.zeros(
            (self.height, self.width, 2), dtype=numpy.long)
    
    def render_image(self):
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
