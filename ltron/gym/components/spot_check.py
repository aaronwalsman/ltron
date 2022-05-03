import random

import numpy

from gym.spaces import MultiDiscrete

from ltron.gym.spaces import ImageSpace
from ltron.gym.components.ltron_gym_component import LtronGymComponent

class SpotCheck(LtronGymComponent):
    def __init__(self, width, height, render_frequency='step'):
        self.width = width
        self.height = height
        assert render_frequency in ('reset', 'step', 'on_demand')
        self.render_frequency = render_frequency
        self.stale = True
        
        self.observation_space = MultiDiscrete((height, width))
    
    def observe(self):
        if self.stale:
            self.render_image()
            self.stale = False
        return self.observation
    
    def reset(self):
        self.stale = True
        self.observe()
        return self.position
    
    def step(self, action):
        self.stale = True
        if self.render_frequency in ('step',):
            self.observe()
        return self.position, 0., False, None
    
    def get_state(self):
        return (self.observation, self.stale)

    def set_state(self, state):
        self.observation, self.stale = state
        return self.observation
    
    def render_image(self):
        self.observation = numpy.zeros(
            (self.height, self.width, 3), dtype=numpy.uint8)
        self.observation[:] = 102
        #self.position = [
        #    random.randint(0, self.height-1), random.randint(0, self.width-1)]
        self.position = [random.randint(0, 255), random.randint(0, 255)]
        #self.position = [random.randint(1,3),0] #[random.randint(1,7), 0]
        y, x = self.position
        #self.observation[y*16+4:(y+1)*16-4, x*16+4:(x+1)*16-4] = 255
        #self.observation[self.position[0], self.position[1]] = 255
        #radius = random.randint(1,5)
        radius = 16
        #min_y = max(0, y-radius)
        max_y = min(y+radius, 256)
        #min_x = max(0, x-radius)
        max_x = min(x+radius, 256)
        self.observation[y:max_y, x:max_x] = 255

class ConstantImage(LtronGymComponent):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
        self.observation = numpy.zeros(
            (self.height, self.width, 3), dtype=numpy.uint8)
        self.observation[:] = 102
        
    def step(self, action):
        return None, 0., False, None
    
    def reset(self):
        return None
