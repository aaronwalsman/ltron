import random
import math
import json

import numpy

from gym.spaces import Dict, Discrete

from ltron.hierarchy import hierarchy_branch
from ltron.score import score_assemblies
from ltron.gym.components.ltron_gym_component import LtronGymComponent
from ltron.gym.spaces import AssemblySpace, InstanceMatchingSpace

class BreakAndMakePhaseSwitch(LtronGymComponent):
    def __init__(self,
        workspace_scene_component,
        task='break_and_make',
        workspace_viewpoint_component=None,
        handspace_scene_component=None,
        dataset_component=None,
        start_make_mode='clear',
        train=False,
    ):
        self.task = task
        self.workspace_scene_component = workspace_scene_component
        self.workspace_viewpoint_component = workspace_viewpoint_component
        self.handspace_scene_component = handspace_scene_component
        self.dataset_component = dataset_component
        self.start_make_mode = start_make_mode
        self.train = train
        
        self.action_space = Discrete(3)
        self.observation_space = Discrete(2)
        self.phase = 0
    
    def observe(self):
        
        self.observation = self.phase
    
    def reset(self):
        self.phase = 0
        self.observe()
        return self.observation
    
    def step(self, action):
        if action == 1 and not self.phase:
            self.phase = 1
            workspace_scene = self.workspace_scene_component.brick_scene
            workspace_scene.clear_instances()
            
            if self.workspace_viewpoint_component is not None:
                self.workspace_viewpoint_component.center = (0,0,0)
            
            if self.handspace_scene_component is not None:
                handspace_scene = self.handspace_scene_component.brick_scene
                handspace_scene.clear_instances()
            
            if self.start_make_mode == 'clear':
                pass
            
            elif self.start_make_mode == 'square':
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
        
        if self.task == 'break_only':
            terminal = (action == 1) or (action == 2)
        else:
            terminal = (action == 2)
        
        return self.observation, 0., terminal, {}
    
    def set_state(self, state):
        self.phase = state['phase']
        return self.observation
    
    def get_state(self):
        return {'phase':self.phase}
    
    def no_op_action(self):
        return 0

class BreakAndMakeScore(LtronGymComponent):
    def __init__(self,
        initial_assembly_component,
        current_assembly_component,
        phase_switch_component,
    ):
        self.initial_assembly_component = initial_assembly_component
        self.current_assembly_component = current_assembly_component
        self.phase_switch_component = phase_switch_component
    
    def observe(self):
        if self.phase_switch_component.phase:
            initial_assembly = self.initial_assembly_component.assembly
            current_assembly = self.current_assembly_component.assembly
            
            self.score, matching = score_assemblies(
                initial_assembly,
                current_assembly,
            )
        else:
            #if self.disassembly_score:
            #    initial_assembly = self.initial_assembly_component.assembly
            #    current_assembly = self.current_assembly_component.assembly
            #    initial_instances = numpy.sum(initial_assembly['class'] != 0)
            #    current_instances = numpy.sum(current_assembly['class'] != 0)
            #    score = 1. - (current_instances/initial_instances)
            #    score *= self.disassembly_score
            #    self.recent_disassembly_score = score
            #    self.score = score
            #
            #else:
            #    self.score = 0.
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

class BreakOnlyScore(LtronGymComponent):
    def __init__(self, initial_assembly_component, current_assembly_component):
        self.initial_assembly_component = initial_assembly_component
        self.current_assembly_component = current_assembly_component
    
    def observe(self):
        initial_assembly = self.initial_assembly_component.assembly
        FINISH_THIS_UP
