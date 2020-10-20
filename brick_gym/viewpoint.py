import math

from gym import spaces

import renderpy.camera as camera

class AzimuthElevationViewpoint:
    def __init__(self,
            distance = 500,
            azimuth_steps = 24,
            elevation_steps = 7,
            elevation_range = (-math.pi/3, math.pi/3)):
        
        self.azimuth_steps = azimuth_steps
        self.elevation_steps = elevation_steps
        self.elevation_range = elevation_range
        
        self.azimuth_index = 0
        self.elevation_index = 0
        
        self.action_space = spaces.Discrete(5)
    
    def get_transform(self):
        self.azimuth = (self.azimuth_index / self.azimuth_steps) * math.pi * 2.
        self.elevation = (
                (self.elevation_index / (self.elevation_steps-1)) *
                (self.elevation_range[1] - self.elevation_range[0]) +
                self.elevation_range[0])
        
        return camera.whatever
    
    def get_state(self):
        return self.azimuth_index, self.elevation_index
    
    def set_state(self, state):
        azimuth_index, elevation_index = state
        self.azimuth_index = azimuth_index
        self.elevation_index = elevation_index
    
    def step(self, action):
        if action == 0:
            self.offset_state((-1, 0))
        elif action == 1:
            self.offset_state(( 0,-1))
        elif action == 2:
            self.offset_state(( 1, 0))
        elif action == 3:
            self.offset_state(( 0, 1))
        elif action == 4:
            pass
    
    def reset(self, state=None):
        if state is not None:
            self.azimuth_index = state[0]
            self.elevation_index = state[1]
        else:
            if self.viewpoint_reset_mode == 'random':
                self.azimuth_index = random.randint(0, self.azimuth_steps-1)
                self.elevation_index = random.randint(0, self.elevation_steps-1)
            elif isinstance(self.viewpoint_reset_mode, tuple):
                self.azimuth_index = self.viewpoint_reset_mode[0]
                self.elevation_index = self.viewpoint_reset_mode[1]
    
    def offset_state(self, offset):
        azimuth_offset, elevation_offset = offset
        self.azimuth_index  = (
                (self.azimuth_index + azimuth_offset) % self.azimuth_steps)
        self.elevation_index += elevation_offset
        self.elevation_index = max(0, self.elevation_index)
        self.elevation_index = min(self.elevation_steps-1, self.elevation_index)
