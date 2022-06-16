import numpy

from splendor.frame_buffer import FrameBufferWrapper
import splendor.masks as masks

from ltron.gym.spaces import ImageSpace, InstanceMaskSpace, SnapMaskSpace
from ltron.gym.components.sensor_component import SensorComponent

class ColorRenderComponent(SensorComponent):
    def __init__(self,
        width,
        height,
        scene_component,
        anti_alias=True,
        update_frequency='step',
        observable=True,
    ):
        super().__init__(
            update_frequency=update_frequency,
            observable=observable,
        )
        
        self.width = width
        self.height = height
        self.scene_component = scene_component
        
        scene = self.scene_component.brick_scene
        self.scene_component.brick_scene.make_renderable()
        self.anti_alias = anti_alias
        
        self.frame_buffer = FrameBufferWrapper(
                self.width, self.height, self.anti_alias)
        
        if observable:
            self.observation_space = ImageSpace(self.width, self.height)
        
        self.observation = numpy.zeros(
            (self.height, self.width, 3), dtype=numpy.uint8)
    
    def update_observation(self):
        scene = self.scene_component.brick_scene
        self.frame_buffer.enable()
        scene.viewport_scissor(0, 0, self.width, self.height)
        scene.color_render()
        self.observation = self.frame_buffer.read_pixels()

class InstanceRenderComponent(SensorComponent):
    def __init__(self,
        width,
        height,
        scene_component,
        update_frequency='step',
        observable=True,
    ):
        
        super().__init__(
            update_frequency=update_frequency,
            observable=observable,
        )
        
        self.width = width
        self.height = height
        self.scene_component = scene_component
        self.scene_component.brick_scene.make_renderable()
        
        self.frame_buffer = FrameBufferWrapper(
                self.width, self.height, anti_alias=False)
        
        if observable:
            self.observation_space = InstanceMaskSpace(
                    self.width, self.height)
        
        self.observation = numpy.zeros(
            (self.height, self.width), dtype=numpy.long)
    
    def update_observation(self):
        scene = self.scene_component.brick_scene
        self.frame_buffer.enable()
        # it seems like this should be done at a lower level than this
        scene.viewport_scissor(0,0,self.width,self.height)
        scene.mask_render()
        mask = self.frame_buffer.read_pixels()
        self.observation = masks.color_byte_to_index(mask)

class SnapRenderComponent(SensorComponent):
    def __init__(self,
        width,
        height,
        scene_component,
        polarity=None,
        style=None,
        update_frequency='step',
        observable=True,
    ):
        
        super().__init__(
            update_frequency=update_frequency,
            observable=observable,
        )
        
        self.width = width
        self.height = height
        self.scene_component = scene_component
        self.scene_component.brick_scene.make_renderable()
        self.polarity=polarity
        self.style=style
        
        self.frame_buffer = FrameBufferWrapper(
            self.width, self.height, anti_alias=False)
        
        if observable:
            self.observation_space = SnapMaskSpace(
                self.width, self.height)
        
        self.observation = numpy.zeros(
            (self.height, self.width, 2), dtype=numpy.long)
    
    def update_observation(self):
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
