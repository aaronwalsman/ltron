import random

import numpy

from gym.spaces import Box

from ltron.gym.components.ltron_gym_component import LtronGymComponent
from ltron.score import edit_distance

class ClassifyExpert(LtronGymComponent):
    def __init__(self,
        env,
        target_assembly_component,
        estimate_assembly_component,
        shape_ids,
        max_actions=1000000,
        max_instructions=2048,
        shuffle_instructions=True,
    ):
        # store variables
        self.env = env
        self.target_assembly_component = target_assembly_component
        self.estimate_assembly_component = estimate_assembly_component
        self.shape_ids = shape_ids
        self.max_instructions = max_instructions
        self.shuffle_instructions = shuffle_instructions
        
        # build observation_space
        self.observation_space = Box(
            low=numpy.zeros(self.max_instructions, dtype=numpy.long),
            high=numpy.full(
                self.max_instructions, max_actions, dtype=numpy.long),
            shape=(self.max_instructions,),
            dtype=numpy.long,
        )
    
    def reset(self):
        return self.observe()
    
    def step(self, action):
        observation = self.observe()
        return observation, 0., False, {}
    
    def observe(self):
        time_step = self.env.components['step'].episode_step
        #if time_step:
        target_assembly = self.target_assembly_component.observe()
        estimate_assembly = self.estimate_assembly_component.observe()
        d, a_to_b = edit_distance(
            estimate_assembly,
            target_assembly,
            {v:k for k,v in self.shape_ids.items()},
        )
        #if a_to_b:
        target_indices = set(numpy.where(target_assembly['shape'])[0])
        remaining_indices = target_indices - set(a_to_b.values())
        if len(remaining_indices):
            #i = random.choice(list(remaining_indices))
            #target_shape = target_assembly['shape'][i]
            #target_color = target_assembly['color'][i]
            target_shape = target_assembly['shape'][time_step+1]
            target_color = target_assembly['color'][time_step+1]
            
            actions = [
                self.env.action_to_insert_brick(target_shape, target_color)]
        else:
            actions = self.env.finish_actions()
            #target_shape = 0
            #target_color = 0
        
        actions = numpy.array(actions)
        if self.shuffle_instructions:
            permutation = numpy.random.permutation(len(actions))
            actions = actions[permutation]
        
        actions = actions[:self.max_instructions]
        obs = numpy.zeros(self.max_instructions, dtype=numpy.long)
        obs[:len(actions)] = actions
        
        #else:
        #    action = self.env.action_to_insert_brick(0,0)
        return obs
