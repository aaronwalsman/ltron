from gym.spaces import MultiDiscrete

from ltron.gym.components.ltron_gym_component import LtronGymComponent

class Cusror(LtronGymComponent):
    def __init__(self, resolution):
        self.resolution = resolution
        self.action_space = MultiDiscrete(*self.resolution)
        self.observation_space = MultiDiscrete(*self.resolution)
        self.set_cursor((0,)*len(self.resolution))
    
    def set_cursor(self, y, x):
        self.cursor = y,x
    
    def get_observation(self):
        return self.cursor
    
    def reset(self):
        self.set_cursor(0,0)
        return self.get_observation()
    
    def step(self, action):
        y,x = action
        self.set_cursor(y,x)
        observation = self.get_observation()
        return observation, 0, False, {}
