import random
import math

import numpy

from gym import spaces

import renderpy.camera as camera

# TODO: Inherit from superclass ViewpointControl that contains camera projection

class AzimuthElevationViewpointControl:
    def __init__(self,
            bbox = ((0,0,0), (0,0,0)),
            azimuth_steps = 24,
            elevation_steps = 7,
            elevation_range = (-math.pi/3, math.pi/3),
            reset_mode = 'centered'):
        
        self.action_space = spaces.Discrete(5)
        self.observation_space = None # TODO
        
        self.set_bbox(bbox)
        self.azimuth_steps = azimuth_steps
        self.elevation_steps = elevation_steps
        self.elevation_range = elevation_range
        self.reset_mode = reset_mode
        self.reset()
    
    def get_transform(self):
        self.azimuth = (self.azimuth_index / self.azimuth_steps) * math.pi * 2.
        self.elevation = (
                (self.elevation_index / (self.elevation_steps-1)) *
                (self.elevation_range[1] - self.elevation_range[0]) +
                self.elevation_range[0])
        
        return camera.azimuthal_pose_to_matrix(
                [self.azimuth, self.elevation, 0.0, self.distance, 0.0, 0.0],
                center=self.center)
    
    def get_state(self):
        return self.azimuth_index, self.elevation_index
    
    def set_state(self, state):
        azimuth_index, elevation_index = state
        self.azimuth_index = azimuth_index
        self.elevation_index = elevation_index
    
    def set_bbox(self, bbox):
        self.bbox = bbox
        bbox_min, bbox_max = bbox
        bbox_range = numpy.array(bbox_max) - numpy.array(bbox_min)
        self.center = bbox_min + bbox_range * 0.5
        self.distance = numpy.max(bbox_range) * 3
    
    def offset_state(self, offset):
        azimuth_offset, elevation_offset = offset
        self.azimuth_index  = (
                (self.azimuth_index + azimuth_offset) % self.azimuth_steps)
        self.elevation_index += elevation_offset
        self.elevation_index = max(0, self.elevation_index)
        self.elevation_index = min(self.elevation_steps-1, self.elevation_index)
    
    def observe(self):
        return self.get_transform(), self.get_state()
    
    def step(self, action):
        if action == 0:
            pass
        elif action == 1:
            self.offset_state((-1, 0))
        elif action == 2:
            self.offset_state(( 0,-1))
        elif action == 3:
            self.offset_state(( 1, 0))
        elif action == 4:
            self.offset_state(( 0, 1))
        
        return self.observe(), 0.0, False, {}
    
    def reset(self):
        if self.reset_mode == 'random':
            self.azimuth_index = random.randint(0, self.azimuth_steps-1)
            self.elevation_index = random.randint(0, self.elevation_steps-1)
        elif self.reset_mode == 'centered':
            self.azimuth_index = 0
            self.elevation_index = math.ceil(self.elevation_steps/2)
        elif isinstance(self.reset_mode, tuple):
            self.azimuth_index = self.reset_mode[0]
            self.elevation_index = self.reset_mode[1]
        
        return self.observe()
    
