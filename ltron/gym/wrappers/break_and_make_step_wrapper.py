from copy import deepcopy

import numpy

from gymnasium import Wrapper
from gymnasium.spaces import Discrete, Dict

from splendor.image import save_image

from ltron.gym.envs.break_and_make_env import BreakAndMakeEnv

BRICK_DONE_BONUS = 0.1
BRICK_DONE_PENALTY = -1

'''
Removes "phase" action for "brick_done" action.
Each time "brick_done" is pressed:
1. If the wrapped env's "phase" is "break":
    1.1. Number of bricks must be 1 less than the previous reset/"brick_done"
        pressed.  If not, terminate with negative reward.
    1.2. If there are no bricks left, automatically switch phase to "make"
2. Else:
    2.1. Number of bricks must be 1 more than the previous reset/"brick_done"
        pressed.  If not, terminate with a negative reward.
    2.2. 
    
'''

class BreakAndMakeStepWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
        # modify the observation space
        observation_space = deepcopy(self.env.observation_space)
        observation_space['target_image'] = deepcopy(observation_space['image'])
        observation_space['target_assembly'] = deepcopy(
            observation_space['assembly'])
        self.observation_space = observation_space
        
        # modify the action space
        action_space = deepcopy(self.env.action_space)
        action_space['brick_done'] = Discrete(2)
        action_space = Dict(
            {k:v for k,v in action_space.items() if k != 'phase'})
        self.action_space = action_space
    
    def no_op_action(self):
        action = self.env.no_op_action()
        del(action['phase'])
        action['brick_done'] = 0
        return action
    
    def observation(self, o):
        o = deepcopy(o)
        if self.env.components['phase'].phase == 0:
            o['target_image'] = numpy.zeros_like(o['image'])
            o['target_assembly'] = {}
            o['target_assembly']['shape'] = numpy.zeros_like(
                o['assembly']['shape'])
            o['target_assembly']['color'] = numpy.zeros_like(
                o['assembly']['color'])
            o['target_assembly']['pose'] = numpy.zeros_like(
                o['assembly']['pose'])
            o['target_assembly']['edges'] = numpy.zeros_like(
                o['assembly']['edges'])
        else:
            o['target_image'] = self.target_images[self.brick_step-1]
            o['target_assembly'] = self.target_assemblies[self.brick_step-1]
        
        return o
    
    def save_debug(self, o):
        image = numpy.concatenate((o['image'], o['target_image']), axis=1)
        save_image(image, 'debug_%04i.png'%self.total_steps)
    
    def reset(self, seed=None, options=None):
        o,i = super().reset(seed=seed, options=options)
        
        # modify the observation
        o = self.observation(o)
        
        self.num_bricks = len(
            self.env.components['scene'].brick_scene.instances)
        self.orig_bricks = self.num_bricks
        self.brick_step = 0
        self.total_steps = 0
        self.target_images = [o['image']]
        self.target_assemblies = [o['assembly']]
        
        self.save_debug(o)
        
        return o, i
    
    def step(self, action):
        
        brick_done_reward = 0
        terminal = False
        switch_phase = False
        if action['brick_done']:
            num_bricks = len(
                self.env.components['scene'].brick_scene.instances)
            if self.env.components['phase'].phase == 0:
                target_bricks = self.num_bricks - 1
                if num_bricks == 0:
                    switch_phase = True
                self.brick_step += 1
            else:
                target_bricks = self.num_bricks + 1
                if num_bricks == self.orig_bricks:
                    switch_phase = True
                self.brick_step -= 1
            
            if num_bricks != target_bricks:
                brick_done_reward += BRICK_DONE_PENALTY
                terminal = True
            else:
                brick_done_reward += BRICK_DONE_BONUS
            
            self.num_bricks = num_bricks
        
        env_action = deepcopy(action)
        del(env_action['brick_done'])
        env_action['phase'] = switch_phase
        
        o,r,t,u,i = self.env.step(env_action)
        
        o = self.observation(o)
        
        self.total_steps += 1
        self.save_debug(o)
        
        if self.env.components['phase'].phase == 0:
            if action['brick_done']:
                self.target_images.append(o['image'])
                self.target_assemblies.append(o['assembly'])
        
        r += brick_done_reward
        t |= terminal
        return o,r,t,u,i

def break_and_make_step_wrapper_env(config, train=True):
    break_and_make_env = BreakAndMakeEnv(config, train)
    wrapped_env = BreakAndMakeStepWrapper(break_and_make_env)
    return wrapped_env
