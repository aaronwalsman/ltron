import random
import math

import numpy

from gym.spaces import Box, Dict

from ltron.matching import (
    match_assemblies,
)
from ltron.gym.spaces import MultiSE3Space
from ltron.gym.components.ltron_gym_component import LtronGymComponent
from ltron.score import edit_distance

class EstimateExpert(LtronGymComponent):
    def __init__(self,
        env,
        
        table_scene_component,
        estimate_scene_component,
        
        target_assembly_component,
        table_assembly_component,
        estimate_assembly_component,
        
        shape_ids,
        include_pose=False,
        max_instructions=2048,
        shuffle_instructions=True,
        always_add_viewpoint_actions=False,
        terminate_on_empty=False,
        max_actions=1000000,
    ):
        # store variables
        self.env = env
        
        self.table_scene_component = table_scene_component
        self.estimate_scene_component = estimate_scene_component
        
        self.target_assembly_component = target_assembly_component
        self.table_assembly_component = table_assembly_component
        self.estimate_assembly_component = estimate_assembly_component
        
        self.shape_ids = shape_ids
        self.include_pose = include_pose
        self.shape_names = {v:k for k,v in self.shape_ids.items()}
        self.max_instructions = max_instructions
        self.shuffle_instructions = shuffle_instructions
        self.always_add_viewpoint_actions = always_add_viewpoint_actions
        self.terminate_on_empty = terminate_on_empty
        
        # build observation space
        if self.include_pose:
            self.observation_space = Dict({
                'mode':Box(low=0,
                    high=max_actions,
                    shape=(self.max_instructions,),
                    dtype=numpy.long,
                ),
                'pose':MultiSE3Space(max_instructions),
            })
        else:
            self.observation_space = Box(
                low=0,
                high=max_actions,
                shape=(self.max_instructions,),
                dtype=numpy.long,
            )
    
    def reset(self):
        observation = self.observe()
        return observation
    
    def step(self, action):
        observation = self.observe()
        if self.terminate_on_empty:
            #terminal = numpy.sum(observation['mode']) == 0
            terminal = numpy.sum(observation) == 0
        else:
            terminal = False
        return observation, 0., terminal, {}
    
    def observe(self):
        # get assemblies
        estimate_assembly = self.estimate_assembly_component.observe()
        target_assembly = self.target_assembly_component.observe()
        
        # compute the expert actions
        modes, poses = self.expert_actions(estimate_assembly, target_assembly)
        
        modes = numpy.array(modes)
        poses = numpy.array(poses)
        
        # shuffle
        if self.shuffle_instructions:
            permutation = numpy.random.permutation(len(modes))
            modes = modes[permutation]
            poses = poses[permutation]
        
        # truncate
        modes = modes[:self.max_instructions]
        poses = poses[:self.max_instructions]
        
        mode_obs = numpy.zeros(self.max_instructions, dtype=numpy.long)
        mode_obs[:len(modes)] = modes
        pose_obs = numpy.zeros((self.max_instructions, 4, 4))
        pose_obs[:len(poses)] = poses
        
        # dictify
        if self.include_pose:
            self.observation = {'mode':mode_obs, 'pose':pose_obs}
        else:
            self.observation = mode_obs
        
        # return
        #print(estimate_assembly['shape'])
        #print(target_assembly['shape'])
        #print('XPERT:', self.observation)
        return self.observation
    
    def expert_actions(self,
        estimate_assembly,
        target_assembly,
    ):
        '''
        Match everything, then:
        1. If everything is matched: FINISH
        2. If the scene is empty: predict any of the unmatched bricks
        3. If the scene is not empty: remove something or change viewpoint
            3.A. Actually defer this until later
        '''
        
        # match the estimate and target assemblies
        d, a_to_b = edit_distance(
            estimate_assembly,
            target_assembly,
            self.shape_names,
        )
        
        # if everything is matched: FINISH
        num_target_instances = numpy.sum(target_assembly['shape'] != 0)
        if len(a_to_b) == num_target_instances:
            finish_actions = self.env.finish_actions()
            modes = numpy.array(finish_actions, dtype=numpy.long)
            poses = numpy.zeros((len(finish_actions), 4, 4))
            return modes, poses
        
        # PREDICT
        remaining_target_indices = list(
            set(numpy.where(target_assembly['shape'])[0]) -
            set(a_to_b.values())
        )
        
        # get remaining shapes/colors/poses
        remaining_shapes = target_assembly['shape'][remaining_target_indices]
        remaining_colors = target_assembly['color'][remaining_target_indices]
        remaining_poses = target_assembly['pose'][remaining_target_indices]
        
        # map shapes/colors to modes
        if self.include_pose:
            modes_poses = [
                self.env.actions_to_insert_brick(shape, color, pose)
                for shape, color, pose
                in zip(remaining_shapes, remaining_colors, remaining_poses)
            ]
            modes, poses = zip(*modes_poses)
            return modes, poses
        else:
            modes = [
                self.env.actions_to_insert_brick(shape, color)
                for shape, color in zip(remaining_shapes, remaining_colors)
            ]
            poses = numpy.zeros((len(modes), 4, 4))
            return modes, poses
