import random
import math

import numpy

from gym.spaces import Dict, Discrete

from ltron.score import score_configurations
from ltron.gym.components.ltron_gym_component import LtronGymComponent

class Reassembly(LtronGymComponent):
    def __init__(self,
        workspace_scene_component,
        handspace_scene_component=None,
        reassembly_mode='clear',
    ):
        self.workspace_scene_component = workspace_scene_component
        self.handspace_scene_component = handspace_scene_component
        self.reassembly_mode = reassembly_mode
        
        self.action_space = Dict({'start':Discrete(2)})
        self.observation_space = Dict({'reassembling':Discrete(2)})
        self.reassembling=False
    
    def reset(self):
        workspace_scene = self.workspace_scene_component.brick_scene
        target_bricks, target_neighbors = workspace_scene.get_brick_neighbors()
        self.target_bricks = [brick.clone() for brick in target_bricks]
        self.target_neighbors = [
            [neighbor.clone() for neighbor in brick_neighbors]
            for brick_neighbors in target_neighbors
        ]
        
        if self.handspace_scene_component is not None:
            handspace_scene = self.handspace_scene_component.brick_scene
            handspace_scene.clear_instances()
        
        self.reassembling=False
        
        return {'reassembling':self.reassembling}
    
    def step(self, action):
        
        if action['start'] and not self.reassembling:
            self.reassembling=True
            workspace_scene = self.workspace_scene_component.brick_scene
            workspace_scene.clear_instances()
            
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
        
        if self.reassembling:
            scene = self.workspace_scene_component.brick_scene
            current_bricks, current_neighbors = scene.get_brick_neighbors()
            
            score = score_configurations(
                self.target_bricks,
                self.target_neighbors,
                current_bricks,
                current_neighbors,
            )
        else:
            score = 0.
        
        return {'reassembling':self.reassembling}, score, False, {}
