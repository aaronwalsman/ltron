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
        table_scene_component,
        #task='break_and_make',
        table_viewpoint_component=None,
        hand_scene_component=None,
        dataset_component=None,
        start_make_mode='clear',
        train=False,
    ):
        #self.task = task
        self.table_scene_component = table_scene_component
        self.table_viewpoint_component = table_viewpoint_component
        self.hand_scene_component = hand_scene_component
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
    
    def no_op_action(self):
        return 0
    
    def step(self, action):
        if action == 1 and not self.phase:
            self.phase = 1
            table_scene = self.table_scene_component.brick_scene
            table_scene.clear_instances()
            
            if self.table_viewpoint_component is not None:
                self.table_viewpoint_component.center = (0,0,0)
            
            if self.hand_scene_component is not None:
                hand_scene = self.hand_scene_component.brick_scene
                hand_scene.clear_instances()
            
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
                        target_brick.brick_shape,
                        target_brick.color,
                        transform,
                    )
            
            else:
                raise NotImplementedError
        
        self.observe()
        
        #if self.task == 'break_only':
        #    terminal = (action == 1) or (action == 2)
        #else:
        #    terminal = (action == 2)
        
        return self.observation, 0., (action == 2), {}
    
    def set_state(self, state):
        self.phase = state['phase']
        return self.observation
    
    def get_state(self):
        return {'phase':self.phase}

class BreakOnlyPhaseSwitch(LtronGymComponent):
    def __init__(self):
        self.action_space = Discrete(3)
        self.observation_space = Discrete(2)
    
    def reset(self):
        return 0
    
    def step(self, action):
        return 0, 0., (action != 0), {}
    
    def no_op_action(self):
        return 0

class BreakAndMakeScore(LtronGymComponent):
    def __init__(self,
        initial_assembly_component,
        current_assembly_component,
        phase_component,
        shape_ids,
    ):
        self.initial_assembly_component = initial_assembly_component
        self.current_assembly_component = current_assembly_component
        self.phase_component = phase_component
        self.part_names = {value:key for key, value in shape_ids.items()}
    
    def observe(self):
        if self.phase_component.phase:
            initial_assembly = self.initial_assembly_component.assembly
            current_assembly = self.current_assembly_component.assembly
            
            self.score, matching = score_assemblies(
                initial_assembly,
                current_assembly,
                self.part_names,
            )
        else:
            self.score = 0.
        
    def reset(self):
        self.observe()
        #self.recent_disassembly_score = 0.
        return None
    
    def step(self, action):
        self.observe()
        return None, self.score, False, {}
    
    def set_state(self, state):
        self.observe()
        return None

'''
class BreakAndCountScore(LtronGymComponent):
    def __init__(self,
        initial_assembly_component,
        hand_scene_component,
        phase_component,
    ):
        self.initial_assembly_component = initial_assembly_component
        self.hand_scene_component = hand_scene_component
        self.phase_component = phase_component
    
    def observe(self):
        if self.phase_component.phase:
            # THIS IS ALL WRONG, YOU HAVE TO DO IT ONCE WHEN THE ITEM IS ADDED
            # NOT EVERY FRAME
            hand_assembly = self.hand_scene_component.brick_scene.get_assembly(
                shape_ids = self.initial_assembly_component.shape_ids,
                color_ids = self.initial_assembly_component.color_ids,
            )
            instance_id = numpy.where(hand_assembly['shape'] != 0)[0]
            shape_id = hand_assembly['shape'][instance_id]
            color_id = hand_assembly['color'][instance_id]
            if (shape_id)
            
        else:
            self.score = 0.
    
    def reset(self):
        # lists are dumb for this, but we won't have big enough ones to matter
        # sets do not allow for multisets which is what we need, and I don't
        # want to deal with making a dictionary multiset thing
        self.true_positive_instances = []
        self.false_positive_instances = []
        initial_assembly = self.initial_assembly_component.assembly
        instance_ids = numpy.where(initial_assembly['shape'] != 0)[0]
        self.false_negative_instances = [
            (initial_assembly['shape'][i], initial_assembly['color'][i]
            for i in instance_ids
        ]
'''

class BreakOnlyScore(LtronGymComponent):
    def __init__(self, initial_assembly_component, current_assembly_component):
        self.initial_assembly_component = initial_assembly_component
        self.current_assembly_component = current_assembly_component
    
    def observe(self):
        initial_assembly = self.initial_assembly_component.assembly
        current_assembly = self.current_assembly_component.assembly
        initial_count = numpy.sum(initial_assembly['shape'] != 0)
        current_count = numpy.sum(current_assembly['shape'] != 0)
        if initial_count:
            self.score = 1. - (current_count / initial_count)
        else:
            self.score = 0.
    
    def reset(self):
        self.observe()
        return None
    
    def step(self, action):
        self.observe()
        return None, self.score, False, {}
    
    def set_state(self, state):
        self.observe()
        return None
