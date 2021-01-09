import sys
import traceback
import multiprocessing

import gym
from gym.vector.async_vector_env import AsyncVectorEnv
from gym import spaces

from brick_gym.bricks.brick_scene import BrickScene

class BrickEnv(gym.Env):
    def __init__(self, components, print_traceback=False):
        self.print_traceback = print_traceback
        try:
            self.components = components
            
            observation_space = {}
            action_space = {}
            for component_name, component in self.components.items():
                if hasattr(component, 'observation_space'):
                    observation_space[component_name] = (
                            component.observation_space)
                if hasattr(component, 'action_space'):
                    action_space[component_name] = component.action_space
            self.observation_space = spaces.Dict(observation_space)
            self.action_space = spaces.Dict(action_space)
        
        except:
            if self.print_traceback:
                exc_class, exc, exc_traceback = sys.exc_info()
                print(''.join(traceback.format_tb(exc_traceback)))
            raise
    
    '''
    def compute_observation(self):
        observation = {}
        for component_name, component in self.components.items():
            if component_name in self.observation_space:
                observation[component_name] = component.compute_observation()
        return observation
    '''
    
    def reset(self):
        try:
            observation = {}
            for component_name, component in self.components.items():
                component_observation = component.reset()
                if component_name in self.observation_space.spaces:
                    observation[component_name] = component_observation
            return observation
        except:
            if self.print_traceback:
                exc_class, exc, exc_traceback = sys.exc_info()
                print(''.join(traceback.format_tb(exc_traceback)))
            raise
    
    '''
    def update_state(self, action):
        for component_name, component in self.components.items():
            component.update_state(self.state, action[component_name])
    
    def reward(self, action):
        reward = 0.
        for component_name, component in self.components.items():
            reward += component.compute_reward(
                    self.state, action[component_name])
        return reward
    
    def check_terminal(self):
        for component in self.components.values():
            if component.check_terminal(self.state):
                return True
        return False
    
    def info(self):
        info = {}
        for component in self.components.values():
            component.update_info(self.state, info)
    
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
    '''
    def check_action(self, action):
        for key in self.action_space:
            if key not in action:
                raise KeyError('Expected key "%s" in action'%key)
    
    def step(self, action):
        try:
            self.check_action(action)
            observation = {}
            reward = 0.
            terminal = False
            info = {}
            for component_name, component in self.components.items():
                if component_name in self.action_space.spaces:
                    component_action = action[component_name]
                else:
                    component_action = None
                o,r,t,i = component.step(component_action)
                if component_name in self.observation_space.spaces:
                    observation[component_name] = o
                reward += r
                terminal |= t
                if i is not None:
                    info[component_name] = i
            
            return observation, reward, terminal, info
        except:
            if self.print_traceback:
                exc_class, exc, exc_traceback = sys.exc_info()
                print(''.join(traceback.format_tb(exc_traceback)))
            raise
    
    def render(self, mode='human', close=False):
        try:
            for component in self.components.values():
                component.render(self.state)
        except:
            if self.print_traceback:
                exc_class, exc, exc_traceback = sys.exc_info()
                print(''.join(traceback.format_tb(exc_traceback)))
            raise
    
    def get_state(self):
        try:
            state = {}
            for component_name, component in self.components.items():
                s = component.get_state()
                if s is not None:
                    state[component_name] = s
        except:
            if self.print_traceback:
                exc_class, exc, exc_traceback = sys.exc_info()
                print(''.join(traceback.format_tb(exc_traceback)))
            raise
    
    def set_state(self, state):
        try:
            for component_name, component in self.components.items():
                component_state = state.get(component_name, None)
                component.set_state(component_state)
        except:
            if self.print_traceback:
                exc_class, exc, exc_traceback = sys.exc_info()
                print(''.join(traceback.format_tb(exc_traceback)))
            raise
    
    def close(self):
        try:
            for component in self.components.values():
                component.close()
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
