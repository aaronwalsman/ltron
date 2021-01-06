import sys
import traceback
import multiprocessing

import gym
from gym.vector.async_vector_env import AsyncVectorEnv
from gym import spaces

class BrickEnv(gym.Env):
    def __init__(self, components, print_traceback=False):
        self.print_traceback = print_traceback
        self.components = components
        self.initialize_state()
        
    def initialize_state(self):
        try:
            self.state = {}
            
            for component in self.components:
                component.initialize_state(self.state)
            
            observation_space = {}
            for component in self.components:
                component.update_observation_space(observation_space)
            self.observation_space = spaces.Dict(observation_space)
            
            action_space = {}
            for component in self.components:
                component.update_action_space(action_space)
            self.action_space = spaces.Dict(action_space)
        except:
            if self.print_traceback:
                exc_class, exc, exc_traceback = sys.exc_info()
                print(''.join(traceback.format_tb(exc_traceback)))
            raise
    
    def set_state(self, state):
        self.state = state
    
    def compute_observation(self):
        observation = {}
        for component in self.components:
            component.compute_observation(self.state, observation)
        return observation
    
    def reset_state(self):
        for component in self.components:
            component.reset_state(self.state)
    
    def reset(self):
        try:
            self.reset_state()
            return self.compute_observation()
        except:
            if self.print_traceback:
                exc_class, exc, exc_traceback = sys.exc_info()
                print(''.join(traceback.format_tb(exc_traceback)))
            raise
    
    def update_state(self, action):
        for component in self.components:
            component.update_state(self.state, action)
    
    def reward(self, action):
        reward = 0.
        for component in self.components:
            reward += component.compute_reward(self.state, action)
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
    
    def check_action(self, action):
        for key in self.action_space:
            if key not in action:
                raise KeyError('Expected key "%s" in action'%key)
    
    def step(self, action):
        try:
            self.check_action(action)
            self.update_state(action)
            observation = self.compute_observation()
            reward = self.reward(action)
            terminal = self.check_terminal()
            info = self.info()
            
            return observation, reward, terminal, info
        except:
            if self.print_traceback:
                exc_class, exc, exc_traceback = sys.exc_info()
                print(''.join(traceback.format_tb(exc_traceback)))
            raise
    
    def render(self, mode='human', close=False):
        try:
            for component in self.components:
                component.render(self.state)
        except:
            if self.print_traceback:
                exc_class, exc, exc_traceback = sys.exc_info()
                print(''.join(traceback.format_tb(exc_traceback)))
            raise
    
    def cleanup(self):
        for component in self.components:
            component.cleanup(self.state)
    
    def close(self):
        try:
            self.cleanup()
        except:
            if self.print_traceback:
                exc_class, exc, exc_traceback = sys.exc_info()
                print(''.join(traceback.format_tb(exc_traceback)))
            raise

def async_brick_env(
        num_processes, env_constructor, *args, **kwargs):
    def constructor_wrapper(i):
        def constructor():
            env = env_constructor(
                    *args, rank=i, size=num_processes, **kwargs)
            return env
        return constructor
    constructors = [constructor_wrapper(i) for i in range(num_processes)]
    vector_env = AsyncVectorEnv(constructors, context='spawn')
    
    return vector_env
