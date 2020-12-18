import gym
from gym import spaces

# no to this, it's too complex to keep track of what's been updated and what hasn't
'''
class DeferError(Exception):
    pass

def run_deferred(functions):
    incomplete_functions = functions
    while incomplete_functions:
        deferred_functions = []
        for function in incomplete_functions:
            defer = function()
            if defer:
                deferred_functions.append(function)
        if len(deferred_functions) == len(incomplete_functions):
            raise DeferError('No progress made in defered evaluation')
'''

class BrickEnv(gym.Env):
    def __init__(self, components):
        self.components = components
        self.initialize_state()
        
    def initialize_state(self):
        self.state = {}
        
        #run_deferred([lambda : component.initialize_state(self.state)
        #        for component in components])
        for component in component:
            component.initialize_state(self.state)
        
        observation_space = {}
        for component in components:
            component.update_observation_space(observation_space)
        self.observation_space = spaces.Dict(observation_space)
        
        action_space = {}
        for component in components:
            component.update_action_space(action_space)
        self.action_space = spaces.Dict(action_space)
    
    def set_state(self, state):
        self.state = state
    
    def compute_observation(self):
        observation = {}
        for component in self.components:
            component.compute_observation(self.state, observation)
        return observation
    
    def reset_state(self):
        #run_deferred([lambda : component.reset_state(self.state)
        #        for component in self.components])
        for component in self.components:
            component.reset_state(self.state)
    
    def reset(self):
        self.reset_state()
        return self.compute_observation()
    
    def update_state(self, action):
        #run_deferred([lambda : component.update_state(self.state, action)
        #        for component in self.components])
        for component in self.components:
            component.update_state(self.state, action)
    
    def reward(self):
        reward = 0.
        for component in self.components:
            reward += component.compute_reward(self.state)
        return reward
    
    def check_terminal(self):
        for component in self.components:
            if component.check_terminal(self.state):
                return True
        return False
    
    def info(self):
        info = {}
        for component in self.components:
            component.update_info(self.state, info)
    
    def step(self, action):
        self.update_state(action)
        observation = self.compute_observation()
        reward = self.reward()
        terminal = self.check_terminal()
        info = self.info()
        
        return observation, reward, terminal, info
    
    def render(self, mode='human', close=False):
        for component in self.components:
            component.render(self.state)
