import numpy

import gym.spaces as gym_spaces

from ltron.gym.components.ltron_gym_component import LtronGymComponent

class DenseMapComponent(LtronGymComponent):
    def __init__(self,
        instance_data_component,
        segmentation_component,
        instance_data_key=(),
    ):
        self.instance_data_component = instance_data_component
        self.segmentation_component = segmentation_component
        self.instance_data_key = instance_data_key
        
        instance_observation_space = instance_data_component.observation_space
        for k in self.instance_data_key:
            instance_observation_space = instance_observation_space[k]
        
        dtype = instance_observation_space.dtype
        box_shape = (
            segmentation_component.height,
            segmentation_component.width,
            *instance_observation_space.shape[1:])
        
        low = numpy.zeros(box_shape, dtype)
        low[:,:] = instance_observation_space.low[0]
        high = numpy.zeros(box_shape, dtype)
        high[:,:] = instance_observation_space.high[0]
        
        self.observation_space = gym_spaces.Box(
            low, high, box_shape, dtype)
    
    def compute_observation(self):
        instance_data = self.instance_data_component.observation
        for k in self.instance_data_key:
            instance_data = instance_data[k]
        self.observation = instance_data[
            self.segmentation_component.observation]
    
    def reset(self):
        self.compute_observation()
        return self.observation
    
    def step(self, action):
        self.compute_observation()
        return self.observation, 0., False, None
