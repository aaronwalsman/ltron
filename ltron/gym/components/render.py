import numpy

from splendor.frame_buffer import FrameBufferWrapper
from splendor.masks import color_byte_to_index

from supermecha.gym.components.sensor_component import SensorComponent
from supermecha.gym.spaces import ImageSpace

from ltron.gym.spaces import InstanceMaskSpace, SnapMaskSpace

class RenderComponent(SensorComponent):
    def __init__(self,
        scene_component,
        width,
        height,
        anti_alias=False,
        update_on_init=False,
        update_on_reset=False,
        update_on_step=False,
        cache_observation=True,
        observable=False,
    ):
        # setup the sensor
        super().__init__(
            update_on_init=update_on_init,
            update_on_reset=update_on_reset,
            update_on_step=update_on_step,
            cache_observation=cache_observation,
            observable=observable,
        )
        
        # store the inputs
        self.scene_component = scene_component
        self.width = width
        self.height = height
        self.anti_alias = anti_alias
        
        # make the scene renderable
        self.scene_component.brick_scene.make_renderable()
        
        # make the rendering framebuffer
        self.frame_buffer = FrameBufferWrapper(
                self.width, self.height, self.anti_alias)

class ColorRenderComponent(RenderComponent):
    def __init__(self,
        scene_component,
        width,
        height,
        anti_alias=True,
        update_on_init=False,
        update_on_reset=False,
        update_on_step=False,
        cache_observation=False,
        observable=False,
    ):
        super().__init__(
            scene_component,
            width,
            height,
            anti_alias=anti_alias,
            update_on_init=update_on_init,
            update_on_reset=update_on_reset,
            update_on_step=update_on_step,
            cache_observation=cache_observation,
            observable=observable,
        )
        
        if observable:
            self.observation_space = ImageSpace(self.width, self.height)
    
    def compute_observation(self):
        scene = self.scene_component.brick_scene
        self.frame_buffer.enable()
        scene.viewport_scissor(0, 0, self.width, self.height)
        scene.color_render()
        observation = self.frame_buffer.read_pixels()
        return observation, None

class InstanceRenderComponent(RenderComponent):
    def __init__(self,
        scene_component,
        width,
        height,
        update_on_init=False,
        update_on_reset=False,
        update_on_step=False,
        cache_observation=False,
        observable=False,
    ):
        
        super().__init__(
            scene_component,
            width,
            height,
            anti_alias=False,
            update_on_init=update_on_init,
            update_on_reset=update_on_reset,
            update_on_step=update_on_step,
            cache_observation=cache_observation,
            observable=observable,
        )
        
        if observable:
            self.observation_space = InstanceMaskSpace(self.width, self.height)
    
    def compute_observation(self):
        scene = self.scene_component.brick_scene
        self.frame_buffer.enable()
        scene.viewport_scissor(0,0,self.width,self.height)
        scene.mask_render()
        mask = self.frame_buffer.read_pixels()
        observation = color_byte_to_index(mask)
        return observation, None

class SnapRenderComponent(RenderComponent):
    def __init__(self,
        scene_component,
        width,
        height,
        polarity=None,
        style=None,
        update_on_init=False,
        update_on_reset=False,
        update_on_step=False,
        cache_observation=False,
        observable=False,
    ):
        
        super().__init__(
            scene_component,
            width,
            height,
            anti_alias=False,
            update_on_init=update_on_init,
            update_on_reset=update_on_reset,
            update_on_step=update_on_step,
            cache_observation=cache_observation,
            observable=observable,
        )
        
        self.polarity=polarity
        self.style=style
        
        if observable:
            self.observation_space = SnapMaskSpace(self.width, self.height)
    
    def compute_observation(self, polarity=None, style=None):
        if polarity is None:
            polarity = self.polarity
        if style is None:
            style = self.style
        
        scene = self.scene_component.brick_scene
        self.frame_buffer.enable()
        scene.viewport_scissor(0,0,self.width,self.height)
        
        # get the snap names
        snaps = scene.get_matching_snaps(
            polarity=polarity, style=style)
        
        # render instance ids
        scene.snap_render_instance_id(snaps)
        instance_id_mask = self.frame_buffer.read_pixels()
        instance_ids = color_byte_to_index(instance_id_mask)
        
        # render snap ids
        scene.snap_render_snap_id(snaps)
        snap_id_mask = self.frame_buffer.read_pixels()
        snap_ids = color_byte_to_index(snap_id_mask)
        
        self.observation = numpy.stack((instance_ids, snap_ids), axis=-1)
        
        return self.observation, None
