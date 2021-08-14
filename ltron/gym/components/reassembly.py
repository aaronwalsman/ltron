import random
import math

import numpy

from gym.spaces import Dict, Discrete

from ltron.score import score_configurations
from ltron.gym.components.ltron_gym_component import LtronGymComponent

class Reassembly(LtronGymComponent):
    def __init__(self,
        scene_component,
    ):
        self.scene_component = scene_component
        self.action_space = Dict({'start':Discrete(2)})
        self.observation_space = Dict({'reassembling':Discrete(2)})
    
    def reset(self):
        scene = self.scene_component.brick_scene
        target_bricks, target_neighbors = scene.get_brick_neighbors()
        self.target_bricks = [brick.clone() for brick in target_bricks]
        self.target_neighbors = [
            [neighbor.clone() for neighbor in brick_neighbors]
            for brick_neighbors in target_neighbors
        ]
        self.reassembling=False
        
        return {'reassembling':self.reassembling}
    
    def step(self, action):
        if action['start'] and not self.reassembling:
            self.reassembling=True
            scene = self.scene_component.brick_scene
            scene.clear_instances()
            square = math.ceil(len(self.target_bricks)**0.5)
            random.shuffle(self.target_bricks)
            spacing=140
            for i, target_brick in enumerate(self.target_bricks):
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
        
        if self.reassembling:
            scene = self.scene_component.brick_scene
            current_bricks, current_neighbors = scene.get_brick_neighbors()
            x_score, y_score = score_configurations(
                self.target_bricks,
                self.target_neighbors,
                current_bricks,
                current_neighbors,
            )
            print(x_score)
            print(y_score)
        else:
            score = 0.
        
        return {'reassembling':self.reassembling}, 0., False, {}
