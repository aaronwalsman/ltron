import random
import math

import numpy

from gym import spaces

import renderpy.camera as camera

# TODO: Inherit from superclass ViewpointControl that contains camera projection

class AzimuthalViewpoint:
    def __init__(self,
            bbox = ((0,0,0), (0,0,0))):
        
        self.action_space = spaces.Discrete(1)
        self.observation_space = None # TODO
        
        self.set_bbox(bbox)
    
    def set_bbox(self, bbox):
        self.bbox = bbox
        bbox_min, bbox_max = bbox
        bbox_range = numpy.array(bbox_max) - numpy.array(bbox_min)
        self.center = bbox_min + bbox_range * 0.5
        self.distance = numpy.max(bbox_range) * 3.0
    
    def get_azimuth_elevation(self):
        return NotImplementedError
    
    def observe(self):
        azimuth, elevation = self.get_azimuth_elevation()
        return camera.azimuthal_pose_to_matrix(
                [azimuth, elevation, 0.0, self.distance, 0.0, 0.0],
                center = self.center)
    
    def apply_action(self, action):
        pass
    
    def step(self, action):
        self.apply_action(action)
        return self.observe(), 0.0, False, {}
    
    def reset(self):
        return self.observe()

class FixedAzimuthalViewpoint(AzimuthalViewpoint):
    def __init__(self,
            bbox = ((0,0,0), (0,0,0)),
            azimuth = 0,
            elevation = 0):
        
        super(FixedAzimuthalViewpoint, self).__init__(bbox)
        self.azimuth = azimuth
        self.elevation = elevation
    
    def get_azimuth_elevation(self):
        return self.azimuth, self.elevation

class StepControlAzimuthalViewpoint(AzimuthalViewpoint):
    def __init__(self,
            bbox = ((0,0,0), (0,0,0)),
            azimuth_steps = 24,
            elevation_steps = 7,
            elevation_range = (-math.pi/3, math.pi/3),
            reset_mode = 'centered'):
        
        super(StepControlAzimuthalViewpoint, self).__init__(bbox)
        
        self.action_space = spaces.Discrete(5)
        
        self.azimuth_steps = azimuth_steps
        self.elevation_steps = elevation_steps
        self.elevation_range = elevation_range
        self.reset_mode = reset_mode
    
    def get_azimuth_elevation(self):
        azimuth = (self.azimuth_index / self.azimuth_steps) * math.pi * 2.
        elevation = (
                (self.elevation_index / (self.elevation_steps-1)) *
                (self.elevation_range[1] - self.elevation_range[0]) +
                self.elevation_range[0])
        
        return azimuth, elevation
    
    def apply_action(self, action):
        if action == 0:
            offset = ( 0, 0)
        elif action == 1:
            offset = (-1, 0)
        elif action == 2:
            offset = ( 0,-1)
        elif action == 3:
            offset = ( 1, 0)
        elif action == 4:
            offset = ( 0, 1)
        azimuth_offset, elevation_offset = offset
        self.azimuth_index  = (
                (self.azimuth_index + azimuth_offset) % self.azimuth_steps)
        self.elevation_index += elevation_offset
        self.elevation_index = max(0, self.elevation_index)
        self.elevation_index = min(self.elevation_steps-1, self.elevation_index)
    
    def reset(self):
        if self.reset_mode == 'random':
            self.azimuth_index = random.randint(0, self.azimuth_steps-1)
            self.elevation_index = random.randint(0, self.elevation_steps-1)
        elif self.reset_mode == 'centered':
            self.azimuth_index = 0
            self.elevation_index = math.floor(self.elevation_steps/2)
        elif isinstance(self.reset_mode, tuple):
            self.azimuth_index = self.reset_mode[0]
            self.elevation_index = self.reset_mode[1]
        
        return self.observe()
