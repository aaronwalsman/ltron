import numpy

from ltron.gym.components.ltron_gym_component import LtronGymComponent
from ltron.gym.spaces import SegmentationSpace

class DeduplicateTileMaskComponent(LtronGymComponent):
    def __init__(self,
        tile_width,
        tile_height,
        render_component,
        background=102,
    ):
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.render_component = render_component
        
        assert self.render_component.width % self.tile_width == 0
        assert self.render_component.height % self.tile_height == 0
        self.width = self.render_component.width // self.tile_width
        self.height = self.render_component.height // self.tile_height
        
        self.observation_space = SegmentationSpace(self.width, self.height, 1)
        self.background = background
    
    def observe(self):
        frame = self.render_component.observation
        h, w, c = frame.shape
        modified_channels = frame != self.previous_frame
        modified_tiles = modified_channels.reshape(
            self.height, self.tile_height, self.width, self.tile_width, c)
        modified_tiles = numpy.moveaxis(modified_tiles, 2, 1)
        modified_tiles = modified_tiles.reshape(self.height, self.width, -1)
        self.observation = numpy.any(modified_tiles, axis=-1).reshape(
            self.height, self.width).astype(numpy.long)
        
        #import pdb
        #pdb.set_trace()
        
        self.previous_frame = frame
    
    def reset(self):
        self.previous_frame = self.background
        self.observe()
        return self.observation
    
    def step(self, action):
        self.observe()
        return self.observation, 0., False, None
    
    def get_state(self):
        return self.observation
    
    def set_state(self, state):
        self.observation = state
        return self.observation
