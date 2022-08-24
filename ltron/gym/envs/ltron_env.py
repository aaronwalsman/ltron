from collections import OrderedDict
import sys
import traceback
import multiprocessing

import gym
from gym.vector.async_vector_env import AsyncVectorEnv
from gym.vector.sync_vector_env import SyncVectorEnv
from gym.spaces import Dict, Discrete, MultiDiscrete

from ltron.config import Config
from ltron.bricks.brick_scene import BrickScene
from ltron.gym.spaces import DiscreteChain

def traceback_decorator(f):
    def wrapper(self, *args, **kwargs):
        try:
            return f(self, *args, **kwargs)
        except:
            if hasattr(self, 'print_traceback') and self.print_traceback:
                exc_class, exc, exc_traceback = sys.exc_info()
                print(''.join(traceback.format_tb(exc_traceback)))
            raise
    
    return wrapper

class LtronEnv(gym.Env):
    @traceback_decorator
    def __init__(
        self,
        components,
        combine_action_space='dict',
        early_termination=False,
        expert_component=None,
        print_traceback=False,
    ):
        self.components = components
        self.combine_action_space = combine_action_space
        self.early_termination = early_termination
        self.expert_component = expert_component
        if early_termination:
            assert self.expert_component is not None
        self.print_traceback = print_traceback
        
        # build the observation space
        observation_space = OrderedDict()
        for component_name, component in self.components.items():
            if hasattr(component, 'observation_space'):
                observation_space[component_name] = (
                        component.observation_space)
        self.observation_space = Dict(observation_space)
        
        # build the action space
        if self.combine_action_space == 'dict':
            action_space = {}
            for component_name, component in self.components.items():
                if hasattr(component, 'action_space'):
                    action_space[component_name] = component.action_space
            self.action_space = Dict(action_space)
        
        elif self.combine_action_space == 'discrete_chain':
            subspaces = {}
            #subspaces = {'NO_OP':Discrete(1)}
            #ignore = {'NO_OP':[]}
            for component_name, component in self.components.items():
                if hasattr(component, 'action_space'):
                    subspaces[component_name] = component.action_space
                    #ignore[component_name] = [component.no_op_action()]
            self.action_space = DiscreteChain(subspaces) #, ignore)
        elif self.combine_action_space == 'single':
            action_components = [
                c for c in self.components.values()
                if hasattr(c, 'action_space')
            ]
            assert len(action_components) == 1
            self.action_component = action_components[0]
            self.action_space = self.action_component.action_space
        else:
            raise ValueError('combine_action_space argument must be either '
                '"dict", "discrete_chain" or "single"')
        
        # build the metadata
        self.metadata = {}
        for component_name, component in self.components.items():
            if hasattr(component, 'metadata'):
                self.metadata[component_name] = component.metadata
        
        # add the observation and action space to the metadata so that we will
        # always have access to our custom spaces when working with vector envs
        # TODO TODO TODO: It's time to make our own subclass of the vector envs
        # so that we don't have to do this.
        self.metadata['observation_space'] = self.observation_space
        self.metadata['action_space'] = self.action_space
        self.metadata['no_op_action'] = self.no_op_action()
    
    @traceback_decorator
    def reset(self):
        observation = {}
        for component_name, component in self.components.items():
            component_observation = component.reset()
            if component_name in self.observation_space.spaces:
                observation[component_name] = component_observation
        
        if self.early_termination:
            self.early_termination_mistakes = 0
            self.update_early_termination_actions(observation)
        
        return observation
    
    @traceback_decorator
    def update_early_termination_actions(self, observation):
        self.expert_actions = observation[self.expert_component]
    
    @traceback_decorator
    def check_early_termination(self, action):
        self.early_termination_mistakes += action not in self.expert_actions
        if self.early_termination_mistakes >= self.early_termination:
            return True
        
        return False
    
    @traceback_decorator
    def check_action(self, action):
        if isinstance(self.action_space, Dict):
            for key in self.action_space:
                if key not in action:
                    raise KeyError('Expected key "%s" in action'%key)
        elif isinstance(self.action_space, DiscreteChain):
            if action >= self.action_space.n:
                raise ValueError(
                    'Expected action to be less than %i'%self.action_space.n)
    
    def dict_component_actions(self, action):
        component_actions = {}
        for component_name, component in self.components.items():
            if hasattr(component, 'action_space'):
                component_actions[component_name] = action[component_name]
            else:
                component_actions[component_name] = None
        
        return component_actions
    
    def discrete_chain_component_actions(self, action):
        component_actions = {}
        active_name, *ijk = self.action_space.unravel(action)
        for component_name, component in self.components.items():
            if hasattr(component, 'action_space'):
                if active_name == component_name:
                    if hasattr(component.action_space, 'layout'):
                        a = component.action_space.layout.ravel(*ijk)
                    elif isinstance(component.action_space, MultiDiscrete):
                        a = ijk
                    else:
                        a = ijk[0]
                    component_actions[component_name] = a
                else:
                    component_actions[component_name] = component.no_op_action()
            else:
                component_actions[component_name] = None
        
        return component_actions
    
    def single_component_actions(self, action):
        component_actions = {}
        for component_name, component in self.components.items():
            if hasattr(component, 'action_space'):
                component_actions[component_name] = action
            else:
                component_actions[component_name] = None
        
        return component_actions
    
    @traceback_decorator
    def step(self, action):
        self.check_action(action)
        if self.combine_action_space == 'dict':
            component_actions = self.dict_component_actions(action)
        elif self.combine_action_space == 'discrete_chain':
            component_actions = self.discrete_chain_component_actions(action)
        elif self.combine_action_space == 'single':
            component_actions = self.single_component_actions(action)
        
        observation = {}
        reward = 0.
        terminal = False
        info = {}
        for component_name, component in self.components.items():
            component_action = component_actions[component_name]
            try:
                o,r,t,i = component.step(component_action)
            except:
                print('step failed for %s'%component_name)
                raise
            if component_name in self.observation_space.spaces:
                observation[component_name] = o
            reward += r
            terminal |= t
            if i is not None:
                info[component_name] = i
        
        if self.early_termination:
            terminate_early = self.check_early_termination(action)
            self.update_early_termination_actions(observation)
            terminal |= terminate_early
        
        return observation, reward, terminal, info
    
    @traceback_decorator
    def render(self, mode='human', close=False):
        for component in self.components.values():
            component.render(self.state)
    
    @traceback_decorator
    def get_state(self):
        state = {}
        for component_name, component in self.components.items():
            s = component.get_state()
            state[component_name] = s
        
        return state
    
    @traceback_decorator
    def set_state(self, state):
        observation = {}
        for component_name, component_state in state.items():
            o = self.components[component_name].set_state(component_state)
            if component_name in self.observation_space.spaces:
                observation[component_name] = o
        
        return observation
    
    @traceback_decorator
    def no_op_action(self):
        action = {}
        if self.combine_action_space == 'dict':
            for component_name, component in self.components.items():
                if hasattr(component, 'action_space'):
                    action[component_name] = component.no_op_action()
            return action
        elif self.combine_action_space == 'chained_discrete':
            return 0
        else:
            for component_name, component in self.components.items():
                if hasattr(component, 'action_space'):
                    return component.no_op_action()
    
    @traceback_decorator
    def close(self):
        for component in self.components.values():
            component.close()

def async_ltron(num_processes, env_constructor, *args, **kwargs):
    def constructor_wrapper(i):
        def constructor():
            env = env_constructor(
                *args, rank=i, size=num_processes, **kwargs)
            return env
        return constructor
    constructors = [constructor_wrapper(i) for i in range(num_processes)]
    vector_env = AsyncVectorEnv(constructors, context='spawn')
    
    return vector_env

def sync_ltron(num_processes, env_constructor, *args, **kwargs):
    def constructor_wrapper(i):
        def constructor():
            env = env_constructor(
                *args, rank=1, size=num_processes, **kwargs)
            return env
        return constructor
    constructors = [constructor_wrapper(i) for i in range(num_processes)]
    vector_env = SyncVectorEnv(constructors)
    
    return vector_env
