import random
import math
import json

import numpy

from gym.spaces import Dict, Discrete

from ltron.hierarchy import hierarchy_branch
from ltron.score import score_configurations
from ltron.gym.components.ltron_gym_component import LtronGymComponent
from ltron.gym.spaces import ConfigurationSpace, InstanceMatchingSpace

class Reassembly(LtronGymComponent):
    def __init__(self,
        #class_ids,
        #color_ids,
        #max_instances,
        #max_edges,
        workspace_scene_component,
        workspace_viewpoint_component=None,
        handspace_scene_component=None,
        dataset_component=None,
        reassembly_mode='clear',
        skip_reassembly=False,
        train=False,
    ):
        #self.class_ids = class_ids
        #self.color_ids = color_ids
        #self.max_instances = max_instances
        self.workspace_scene_component = workspace_scene_component
        self.workspace_viewpoint_component = workspace_viewpoint_component
        self.handspace_scene_component = handspace_scene_component
        self.dataset_component = dataset_component
        self.reassembly_mode = reassembly_mode
        self.skip_reassembly = skip_reassembly
        self.train = train
        
        #self.action_space = Dict({'start':Discrete(2), 'end':Discrete(2)})
        self.action_space = Discrete(3)
        observation_space = {'reassembling':Discrete(2)}
        '''
        if self.train:
            observation_space['target_configuration'] = ConfigurationSpace(
                #num_classes,
                #num_colors,
                class_ids,
                color_ids,
                max_instances,
                max_edges,
            )
            observation_space['workspace_configuration'] = ConfigurationSpace(
                #num_classes,
                #num_colors,
                class_ids,
                color_ids,
                max_instances,
                max_edges,
            )
            observation_space['handspace_configuration'] = ConfigurationSpace(
                #num_classes,
                #num_colors,
                class_ids,
                color_ids,
                1,
                0,
            )
            observation_space['target_workspace_matching'] = (
                InstanceMatchingSpace(max_instances))
        '''
        self.observation_space = Dict(observation_space)
        self.reassembling=False
    
    def observe(self):
        #scene = self.workspace_scene_component.brick_scene
        #handspace_scene = self.handspace_scene_component.brick_scene
        #workspace_space = self.observation_space['workspace_configuration']
        #workspace_configuration = workspace_space.from_scene(scene)
        #target_configuration = self.workspace_scene_component.initial_config
        #workspace_configuration = self.workspace_scene_component.config
        
        #self.score, matching = score_configurations(
        #    target_configuration,
        #    workspace_configuration,
        #)
        
        self.observation = {'reassembling':self.reassembling}
        '''
        if self.train:
            self.observation['target_configuration'] = self.target_configuration
            self.observation['workspace_configuration'] = (
                workspace_configuration)
            handspace_space = self.observation_space['handspace_configuration']
            handspace_configuration = handspace_space.from_scene(
                handspace_scene)
            self.observation['handspace_configuration'] = (
                handspace_configuration)
            matching_array = numpy.zeros(
                (self.max_instances, 2), dtype=numpy.long)
            if len(matching):
                matching_array[:len(matching)] = list(matching)
            self.observation['target_workspace_matching'] = matching_array
        '''
    
    def reset(self):
        '''
        workspace_scene = self.workspace_scene_component.brick_scene
        if self.train:
            target_space = self.observation_space['target_configuration']
            self.target_configuration = target_space.from_scene(workspace_scene)
        
        if self.handspace_scene_component is not None:
            handspace_scene = self.handspace_scene_component.brick_scene
            handspace_scene.clear_instances()
        '''
        self.reassembling=False
        self.observe()
        return self.observation
    
    def step(self, action):
        
        #if action['start'] and not self.reassembling:
        if action == 1 and not self.reassembling:
            self.reassembling=True
            workspace_scene = self.workspace_scene_component.brick_scene
            workspace_scene.clear_instances()
            
            if self.workspace_viewpoint_component is not None:
                self.workspace_viewpoint_component.center = (0,0,0)
            
            if self.handspace_scene_component is not None:
                handspace_scene = self.handspace_scene_component.brick_scene
                handspace_scene.clear_instances()
            
            if self.reassembly_mode == 'clear':
                pass
            
            elif self.reassembly_mode == 'square':
                square = math.ceil(len(self.target_bricks)**0.5)
                brick_order = list(range(len(self.target_bricks)))
                spacing=140
                for i, brick_id in enumerate(brick_order):
                    target_brick = self.target_bricks[brick_id]
                    x = i % square
                    z = i // square
                    transform = scene.upright.copy()
                    transform[0,3] = (x-square/2.) * spacing
                    transform[2,3] = (z-square/2.) * spacing
                    scene.add_instance(
                        target_brick.brick_type,
                        target_brick.color,
                        transform,
                    )
            
            else:
                raise NotImplementedError
        
        self.observe()
        #if self.reassembling:
        #    score = self.score
        #else:
        #    score = 0.
        
        #return self.observation, 0., action['end'], {}
        
        if self.skip_reassembly:
            terminal = (action == 1) or (action == 2)
        else:
            terminal = (action == 2)
        
        return self.observation, 0., terminal, {}
    
    def set_state(self, state):
        self.reassembling = state['reassembling']
        return self.observation
    
    def get_state(self):
        return {'reassembling':self.reassembling}

class ReassemblyScoreComponent(LtronGymComponent):
    def __init__(self,
        initial_config_component,
        current_config_component,
        reassembly_component,
        disassembly_score=0.,
    ):
        self.initial_config_component = initial_config_component
        self.current_config_component = current_config_component
        self.reassembly_component = reassembly_component
        self.disassembly_score = disassembly_score
        self.recent_disassembly_score = 0.
    
    def observe(self):
        if self.reassembly_component.reassembling:
            initial_config = self.initial_config_component.config
            current_config = self.current_config_component.config
            
            self.score, matching = score_configurations(
                initial_config,
                current_config,
            )
            self.score += self.recent_disassembly_score
        else:
            if self.disassembly_score:
                initial_config = self.initial_config_component.config
                current_config = self.current_config_component.config
                initial_instances = numpy.sum(initial_config['class'] != 0)
                current_instances = numpy.sum(current_config['class'] != 0)
                score = 1. - (current_instances/initial_instances)
                score *= self.disassembly_score
                self.recent_disassembly_score = score
                self.score = score
                
            else:
                self.score = 0.
    
    def reset(self):
        self.observe()
        self.recent_disassembly_score = 0.
        return None
    
    def step(self, action):
        self.observe()
        return None, self.score, False, {}
    
    def set_state(self, state):
        self.observe()
        return None
